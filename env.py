from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
import numpy as np

from config import ExpCfg, CFG
from world import Scene
from obs import ObservationBuilder


@dataclass
class MoveResult:
    clipped: np.ndarray
    oob_mask: np.ndarray
    oob_count: int
    overflow_dist: float
    oob_reward: float


@dataclass
class StepMetrics:
    covered_any: np.ndarray
    pkts_before: np.ndarray
    max_before: float
    per_uav_max: List[float]
    team_max_sum: float


class CollectDataEnv:
    """
    Structured environment for UAV data collection.

    Pipeline of one step:
        1) move UAVs
        2) advance packet dynamics
        3) compute metrics before clearing
        4) clear covered users
        5) compute reward
        6) record logs and return obs/info

    Notes:
    - The env still outputs a flat observation vector.
    - Observation semantics are defined by ObservationBuilder.
    - Higher-level model-side parsing should be handled outside the env
      (for example in condition.py).
    """

    def __init__(self, cfg: Optional[ExpCfg] = None):
        self.cfg: ExpCfg = cfg or CFG
        self.cfg.validate()

        self.scene: Optional[Scene] = None
        self.obs_builder = ObservationBuilder(self.cfg)

        # trajectory / logging
        self.traj: List[List[np.ndarray]] = []
        self.hist_max_pkt: List[float] = []
        self.hist_cov_max_pkt: List[float] = []
        self.step_idx: List[int] = []

    # ============================================================
    # Public API
    # ============================================================
    def reset(self) -> np.ndarray:
        """
        First call:
            - build fixed map once
        Every call:
            - reset only episode states
        """
        if self.scene is None:
            self.scene = Scene()
            self.scene.build_map_once(self.cfg)
        else:
            self.scene.reset_episode(self.cfg)

        uav_pos = self.scene.uavs_pos()
        self.traj = [[pos.copy()] for pos in uav_pos]

        self.hist_max_pkt = []
        self.hist_cov_max_pkt = []
        self.step_idx = []

        return self._build_obs()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        assert self.scene is not None, "Call reset() before step()."

        move_res = self._move_uavs(action)
        self._advance_packets()
        metrics = self._compute_metrics()
        self._clear_covered(metrics.covered_any)
        reward = self._compute_reward(metrics, move_res)

        obs = self._build_obs()
        self._record_step(metrics)
        info = self._build_info(metrics, move_res, reward)

        done = False
        return obs, float(reward), done, info

    def obs_dim(self) -> int:
        """
        Return current flat observation dimension.
        """
        return int(self.obs_builder.obs_dim(self.scene))

    def act_dim(self) -> int:
        """
        Flat action dimension used by agent side:
            act_dim = 2 * n_uavs
        """
        return 2 * int(self.cfg.env.n_uavs)

    def get_scene(self) -> Scene:
        """
        Expose the current scene for higher-level utilities.
        Useful for debugging or future adapters that want richer state access.
        """
        assert self.scene is not None, "Call reset() before get_scene()."
        return self.scene

    def get_obs_builder(self) -> ObservationBuilder:
        """
        Expose the observation builder for adapter-side utilities.
        """
        return self.obs_builder

    # ============================================================
    # Stage 1: Move
    # ============================================================
    def _move_uavs(self, action: np.ndarray) -> MoveResult:
        assert self.scene is not None

        M = len(self.scene.uavs)
        assert action.shape == (M, 2), f"action shape should be ({M}, 2)"

        dxdy = action.astype(np.float32, copy=True)

        cur_pos = self.scene.uavs_pos().copy()
        proposed = cur_pos + dxdy
        clipped = self._clip_world(proposed.copy())

        oob_mask = (np.abs(proposed - clipped) > 1e-6).any(axis=1)
        oob_count = int(oob_mask.sum())

        if oob_count > 0:
            overflow_dist = float(np.linalg.norm((proposed - clipped)[oob_mask], axis=1).sum())
        else:
            overflow_dist = 0.0

        oob_penalty = float(getattr(self.cfg.env, "oob_penalty", 0.0))
        oob_penalty_scale = float(getattr(self.cfg.env, "oob_penalty_scale", 0.0))
        oob_reward = oob_penalty * oob_count - oob_penalty_scale * overflow_dist

        for i, uav in enumerate(self.scene.uavs):
            uav.prev_pos = uav.pos.copy()
            uav.pos = clipped[i]
            self.traj[i].append(clipped[i].copy())

        return MoveResult(
            clipped=clipped,
            oob_mask=oob_mask,
            oob_count=oob_count,
            overflow_dist=overflow_dist,
            oob_reward=oob_reward,
        )

    def _clip_world(self, xy: np.ndarray) -> np.ndarray:
        w = float(self.cfg.env.world_size)
        np.clip(xy[:, 0], 0.0, w, out=xy[:, 0])
        np.clip(xy[:, 1], 0.0, w, out=xy[:, 1])
        return xy

    # ============================================================
    # Stage 2: Dynamics
    # ============================================================
    def _advance_packets(self) -> None:
        assert self.scene is not None
        self.scene.update_pkts(float(self.cfg.env.step_move_sec))
        self.scene.update_pkts(float(self.cfg.env.step_collect_sec))

    # ============================================================
    # Stage 3: Metrics before clearing
    # ============================================================
    def _compute_metrics(self) -> StepMetrics:
        assert self.scene is not None

        users = self.scene.users_pos()  # (N,2)
        uavs = self.scene.uavs_pos()    # (M,2)
        M = len(self.scene.uavs)

        if users.shape[0] == 0 or uavs.shape[0] == 0:
            covered_any = np.zeros((users.shape[0],), dtype=bool)
            pkts_before = np.zeros((users.shape[0],), dtype=np.float32)
            return StepMetrics(
                covered_any=covered_any,
                pkts_before=pkts_before,
                max_before=0.0,
                per_uav_max=[0.0 for _ in range(M)],
                team_max_sum=0.0,
            )

        r2 = float(self.cfg.env.cov_radius) ** 2
        diff = users[:, None, :] - uavs[None, :, :]
        dist2 = np.sum(diff * diff, axis=-1)
        covered_mask = (dist2 <= r2)
        covered_any = covered_mask.any(axis=1)

        pkts_before = self.scene.pkts_num_array().astype(np.float32)
        max_before = float(pkts_before.max()) if pkts_before.size > 0 else 0.0

        if not covered_any.any():
            return StepMetrics(
                covered_any=covered_any,
                pkts_before=pkts_before,
                max_before=max_before,
                per_uav_max=[0.0 for _ in range(M)],
                team_max_sum=0.0,
            )

        dist2_masked = np.where(covered_mask, dist2, np.inf)
        owner = dist2_masked.argmin(axis=1)

        covered_idx = np.nonzero(covered_any)[0]
        owners_cov = owner[covered_any]

        per_uav_max: List[float] = []
        for j in range(M):
            idx_j = covered_idx[owners_cov == j]
            per_uav_max.append(float(pkts_before[idx_j].max()) if idx_j.size > 0 else 0.0)

        team_max_sum = float(np.sum(per_uav_max))

        return StepMetrics(
            covered_any=covered_any,
            pkts_before=pkts_before,
            max_before=max_before,
            per_uav_max=per_uav_max,
            team_max_sum=team_max_sum,
        )

    # ============================================================
    # Stage 4: Clear covered users
    # ============================================================
    def _clear_covered(self, covered_any: np.ndarray) -> None:
        assert self.scene is not None
        if covered_any.size == 0 or not covered_any.any():
            return

        idxs = np.nonzero(covered_any)[0]
        for idx in idxs:
            u = self.scene.users[int(idx)]
            u.pkts_num = 0
            u.last_visit = 0.0

    # ============================================================
    # Stage 5: Reward
    # ============================================================
    def _compute_reward(self, metrics: StepMetrics, move_res: MoveResult) -> float:
        if not self.cfg.env.ratio_reward:
            return float(
                self.cfg.env.team_max_sum_weight * metrics.team_max_sum
                - self.cfg.env.max_before_weight * metrics.max_before
                + move_res.oob_reward
            )

        ratio = (
            self.cfg.env.ratio_weight * metrics.team_max_sum / metrics.max_before
            if metrics.max_before > 1e-6 else 0.0
        )
        return float(
            self.cfg.env.team_max_sum_weight * ratio
            - self.cfg.env.max_before_weight * metrics.max_before
            + move_res.oob_reward
        )

    # ============================================================
    # Stage 6: Obs / logging / info
    # ============================================================
    def _build_obs(self) -> np.ndarray:
        assert self.scene is not None
        return self.obs_builder.build(self.scene)

    def _record_step(self, metrics: StepMetrics) -> None:
        assert self.scene is not None

        step_id = len(self.step_idx)
        self.step_idx.append(step_id)

        cur_all_pkts = self.scene.pkts_num_array().astype(np.float32)
        self.hist_max_pkt.append(float(cur_all_pkts.max()) if cur_all_pkts.size > 0 else 0.0)

        self.hist_cov_max_pkt.append(float(max(metrics.per_uav_max)) if metrics.per_uav_max else 0.0)

    def _build_info(self, metrics: StepMetrics, move_res: MoveResult, reward: float) -> Dict[str, Any]:
        assert self.scene is not None

        return {
            "time": float(self.scene.now),
            "covered_count": int(metrics.covered_any.sum()) if metrics.covered_any.size > 0 else 0,
            "max_before": float(metrics.max_before),
            "per_uav_max_pre_clear": metrics.per_uav_max,
            "team_max_sum_pre_clear": float(metrics.team_max_sum),
            "oob_count": int(move_res.oob_count),
            "oob_overflow_dist": float(move_res.overflow_dist),
            "oob_penalty": float(move_res.oob_reward),
            "reward_final": float(reward),
        }

    # ============================================================
    # Render
    # ============================================================
    def render(
        self,
        show: bool = True,
        save_path: Optional[str] = None,
        preset: str = "ob",
        dpi: Optional[int] = None,
        figsize_traj: Optional[Tuple[float, float]] = None,
        figsize_pkts: Optional[Tuple[float, float]] = None,
    ):
        """
        Left:
            Sensor points + UAV positions + coverage circles + trajectories
            + historical coverage discs
        Right:
            Slot-wise bar plot:
                1) max backlog
                2) max covered packets
        """
        if self.scene is None:
            print("Call reset() first.")
            return

        import os
        import matplotlib.pyplot as plt

        preset = (preset or "paper").lower().strip()
        if preset not in ("ob", "paper"):
            raise ValueError(f"preset must be 'ob' or 'paper', got: {preset}")

        if preset == "paper":
            rc = {
                "font.size": 16,
                "axes.titlesize": 24,
                "axes.labelsize": 24,
                "xtick.labelsize": 20,
                "ytick.labelsize": 20,
                "legend.fontsize": 18,
                "figure.dpi": 200,
            }
            default_figsize = (14, 6)
            default_dpi = 200
            tick_fs = 20
            label_fs = 24
            title_fs = 24
            traj_linewidth = 1.5
            traj_markersize = 3.5
            uav_scatter_size = 60
            user_scatter_size = 10
            hist_alpha = 0.05
            final_cov_alpha = 0.25
            legend_loc_left = "lower left"
            legend_loc_right = "upper left"
        else:
            rc = {
                "font.size": 10,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
                "figure.dpi": 140,
            }
            default_figsize = (10.5, 4.4)
            default_dpi = 160
            tick_fs = 10
            label_fs = 12
            title_fs = 14
            traj_linewidth = 1.2
            traj_markersize = 2.8
            uav_scatter_size = 45
            user_scatter_size = 8
            hist_alpha = 0.04
            final_cov_alpha = 0.22
            legend_loc_left = "upper right"
            legend_loc_right = "upper left"

        plt.rcParams.update(rc)

        if dpi is None:
            dpi = default_dpi

        users = self.scene.users_pos()
        uavs = self.scene.uavs_pos()
        w = float(self.cfg.env.world_size)

        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(max(10, len(uavs)))]

        save_combined = None
        if save_path is not None:
            if os.path.isdir(save_path):
                save_combined = os.path.join(save_path, "uav_case_display.png")
            else:
                root, ext = os.path.splitext(save_path)
                if ext == "":
                    save_combined = f"{save_path}.png"
                else:
                    save_combined = save_path

        fig = plt.figure(figsize=default_figsize, dpi=dpi)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 0.90])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])

        fig.subplots_adjust(
            left=0.06,
            right=0.99,
            bottom=0.12,
            top=0.86,
            wspace=0.02 if preset == "paper" else 0.08,
        )

        ax1.set_title("Sensor Distribution and\nUAV coverage & Trajectories", fontsize=title_fs, pad=10)
        ax1.set_xlim(0, w)
        ax1.set_ylim(0, w)
        ax1.set_aspect("equal", adjustable="box")
        ax1.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
        ax1.tick_params(axis="both", which="major", labelsize=tick_fs)

        for i in range(len(uavs)):
            c = colors[i]
            traj_i = np.asarray(self.traj[i], dtype=np.float32) if len(self.traj[i]) > 0 else None

            if traj_i is not None and len(traj_i) > 0:
                for (x, y) in traj_i[:-1]:
                    circ_hist = plt.Circle(
                        (x, y),
                        radius=float(self.cfg.env.cov_radius),
                        color=c,
                        fill=True,
                        alpha=hist_alpha,
                        linewidth=0,
                    )
                    ax1.add_patch(circ_hist)

            if traj_i is not None and len(traj_i) > 1:
                ax1.plot(
                    traj_i[:, 0],
                    traj_i[:, 1],
                    "-",
                    linewidth=traj_linewidth,
                    color=c,
                    alpha=0.9,
                    zorder=3,
                )
                ax1.plot(
                    traj_i[:, 0],
                    traj_i[:, 1],
                    "o",
                    markersize=traj_markersize,
                    color=c,
                    alpha=0.9,
                    zorder=4,
                )

            ax1.scatter(
                [uavs[i, 0]],
                [uavs[i, 1]],
                s=uav_scatter_size,
                color=c,
                edgecolors="k",
                linewidths=0.8,
                label=f"UAV{i}",
                zorder=5,
            )

            circ_final = plt.Circle(
                (uavs[i, 0], uavs[i, 1]),
                radius=float(self.cfg.env.cov_radius),
                edgecolor=c,
                facecolor=c,
                alpha=final_cov_alpha,
                linewidth=1.2,
            )
            ax1.add_patch(circ_final)

        if len(users) > 0:
            ax1.scatter(
                users[:, 0],
                users[:, 1],
                s=user_scatter_size,
                c="black",
                label="Sensors",
                alpha=0.8,
                zorder=1,
            )

        ax1.legend(loc=legend_loc_left, fontsize=rc["legend.fontsize"], framealpha=0.5)

        ax2.set_title("slot-wise max backlog vs.\nmax covered packets", fontsize=title_fs, pad=10)

        n = min(len(self.hist_max_pkt), len(self.hist_cov_max_pkt), len(self.step_idx))
        steps = np.arange(n)
        env_max = np.asarray(self.hist_max_pkt[:n], dtype=np.float32)
        cov_max = np.asarray(self.hist_cov_max_pkt[:n], dtype=np.float32)

        width = 0.4
        ax2.bar(
            steps - width / 2,
            env_max,
            width=width,
            label="max backlog",
            alpha=0.95,
        )
        ax2.bar(
            steps + width / 2,
            cov_max,
            width=width,
            label="max covered packets",
            alpha=0.95,
        )

        ax2.set_xlabel("slots", fontsize=label_fs)
        ax2.set_ylabel("packets", fontsize=label_fs)
        ax2.tick_params(axis="both", which="major", labelsize=tick_fs)
        ax2.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
        ax2.legend(loc=legend_loc_right, fontsize=rc["legend.fontsize"], framealpha=0.9)

        if save_combined:
            fig.savefig(save_combined, dpi=dpi, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)