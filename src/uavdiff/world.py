from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np

from uavdiff.config import ExpCfg, derive_seed

Vec2 = np.ndarray


# =========================
# Entities
# =========================
@dataclass
class User:
    pos: Vec2
    lam: float = 0.5
    last_visit: float = 0.0
    pkts_num: int = 0


@dataclass
class UAV:
    pos: Vec2
    battery: float = 1.0
    cov_radius: float = 86.6
    prev_pos: Optional[Vec2] = None

    def __post_init__(self) -> None:
        if self.prev_pos is None:
            self.prev_pos = self.pos.copy()


@dataclass
class Airship:
    pos: Vec2 = field(default_factory=lambda: np.array([200.0, 200.0], dtype=np.float32))


# =========================
# Scene (world state container)
# =========================
@dataclass
class Scene:
    users: List[User] = field(default_factory=list)
    uavs: List[UAV] = field(default_factory=list)
    airship: Airship = field(default_factory=Airship)

    now: float = 0.0
    world_size: float = 400.0

    topo_rng: np.random.Generator = field(default_factory=np.random.default_rng)
    dyn_rng: np.random.Generator = field(default_factory=np.random.default_rng)

    _users_pos_fixed: Optional[np.ndarray] = None

    # ---------- array helpers ----------
    def users_pos(self) -> np.ndarray:
        if self._users_pos_fixed is not None:
            return self._users_pos_fixed
        if not self.users:
            return np.zeros((0, 2), dtype=np.float32)
        return np.stack([u.pos for u in self.users], axis=0).astype(np.float32)

    def uavs_pos(self) -> np.ndarray:
        if not self.uavs:
            return np.zeros((0, 2), dtype=np.float32)
        return np.stack([u.pos for u in self.uavs], axis=0).astype(np.float32)

    def lam_array(self) -> np.ndarray:
        if not self.users:
            return np.zeros((0,), dtype=np.float32)
        return np.array([u.lam for u in self.users], dtype=np.float32)

    def pkts_num_array(self) -> np.ndarray:
        if not self.users:
            return np.zeros((0,), dtype=np.int32)
        return np.array([u.pkts_num for u in self.users], dtype=np.int32)

    def last_visit_array(self) -> np.ndarray:
        if not self.users:
            return np.zeros((0,), dtype=np.float32)
        return np.array([u.last_visit for u in self.users], dtype=np.float32)

    # ============================================================
    # Core API
    # ============================================================
    def build_map_once(self, cfg: ExpCfg) -> None:
        """
        Build fixed user distribution ONCE.
        """
        cfg.validate()

        self.world_size = float(cfg.env.world_size)

        # seed streams
        self.topo_rng = np.random.default_rng(int(cfg.seeds.map_seed))
        dyn_seed = derive_seed(cfg.seeds.exp_seed, cfg.seeds.env_dyn_offset)
        self.dyn_rng = np.random.default_rng(int(dyn_seed))

        # airship center
        center = np.array([self.world_size / 2.0, self.world_size / 2.0], dtype=np.float32)
        self.airship.pos = center.copy()

        # fixed users
        pos = self._generate_user_positions(cfg)
        self._users_pos_fixed = pos

        lam = float(cfg.env.lam)
        self.users = [User(pos=pos[i], lam=lam, last_visit=0.0, pkts_num=0) for i in range(pos.shape[0])]

        self.uavs = [
            UAV(pos=center.copy(), prev_pos=center.copy(), battery=1.0, cov_radius=float(cfg.env.cov_radius))
            for _ in range(int(cfg.env.n_uavs))
        ]

        self.reset_episode(cfg)

    def reset_episode(self, cfg: ExpCfg) -> None:
        """
        Reset per-episode states WITHOUT changing user distribution.
        """
        cfg.validate()

        self.now = 0.0

        center = np.array([self.world_size / 2.0, self.world_size / 2.0], dtype=np.float32)
        self.airship.pos = center.copy()

        for uav in self.uavs:
            uav.prev_pos = uav.pos.copy()
            uav.pos = center.copy()

        lam = float(cfg.env.lam)
        for u in self.users:
            u.pkts_num = 0
            u.last_visit = 0.0
            u.lam = lam

        cold_dt = float(cfg.env.cold_start_dt)
        if cold_dt > 0.0:
            self.update_pkts(cold_dt)

    def update_pkts(self, dt: float) -> None:
        """
        Poisson packet arrival:
            inc ~ Poisson(lam * dt)
        """
        dt = float(dt)
        if dt <= 0.0:
            return

        if not self.users:
            self.now += dt
            return

        lam_arr = self.lam_array()
        inc = self.dyn_rng.poisson(lam_arr * dt, size=lam_arr.shape).astype(np.int32)

        for i, u in enumerate(self.users):
            u.pkts_num += int(inc[i])
            u.last_visit += dt

        self.now += dt

    def collect_pkts(self, dt: float) -> None:
        # compatibility alias
        self.update_pkts(dt)

    # ============================================================
    # Internal helpers
    # ============================================================
    def _generate_user_positions(self, cfg: ExpCfg) -> np.ndarray:
        """
        Cluster + uniform generation.
        """
        ug = cfg.user_gen
        ug.validate()

        n_users = int(ug.n_users)
        w = float(cfg.env.world_size)

        if n_users <= 0:
            return np.zeros((0, 2), dtype=np.float32)

        n_clustered = int(round(ug.cluster_ratio * n_users))
        n_clustered = max(0, min(n_users, n_clustered))
        n_uniform = n_users - n_clustered

        pts = []

        # clustered
        if n_clustered > 0:
            n_clusters = int(ug.n_clusters)
            n_clusters = max(1, n_clusters)

            centers = np.column_stack([
                self.topo_rng.uniform(50, w - 50, n_clusters),
                self.topo_rng.uniform(50, w - 50, n_clusters),
            ]).astype(np.float32)

            counts = np.full(n_clusters, n_clustered // n_clusters, dtype=int)
            counts[: (n_clustered % n_clusters)] += 1

            for k in range(n_clusters):
                nk = int(counts[k])
                if nk <= 0:
                    continue
                cx, cy = centers[k]
                x = self.topo_rng.normal(cx, ug.cluster_std, nk)
                y = self.topo_rng.normal(cy, ug.cluster_std, nk)
                pts.append(np.column_stack([x, y]))

        # uniform
        if n_uniform > 0:
            xu = self.topo_rng.uniform(0, w, n_uniform)
            yu = self.topo_rng.uniform(0, w, n_uniform)
            pts.append(np.column_stack([xu, yu]))

        pos = np.vstack(pts).astype(np.float32) if pts else np.zeros((0, 2), dtype=np.float32)
        pos[:, 0] = np.clip(pos[:, 0], 0, w)
        pos[:, 1] = np.clip(pos[:, 1], 0, w)
        return pos