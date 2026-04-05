from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from uavdiff.config import ExpCfg
from uavdiff.world import Scene


@dataclass
class ObservationBuilder:
    """
    Build flat observation vector from Scene.

    Supported modes (cfg.obs.mode):
    1) "lv+uav":
        [ last_visit(all users), uav_pos(flat) ]

    2) "lv+uav+topk":
        [ last_visit(all users), uav_pos(flat), topK_oldest_user_pos(flat) ]
    """

    cfg: ExpCfg

    def obs_dim(self, scene: Scene | None = None) -> int:
        """
        Dimension of observation vector.
        Prefer using config (fixed n_users/n_uavs), but also works with scene.
        """
        n_users = int(self.cfg.user_gen.n_users) if scene is None else len(scene.users)
        n_uavs = int(self.cfg.env.n_uavs) if scene is None else len(scene.uavs)

        base = n_users + 2 * n_uavs  # last_visit + uav_pos(flat)

        if self.cfg.obs.mode == "lv+uav":
            return base

        if self.cfg.obs.mode == "lv+uav+topk":
            k = int(self.cfg.obs.topk_oldest)
            return base + 2 * k

        raise ValueError(f"Unknown obs mode: {self.cfg.obs.mode}")

    def build(self, scene: Scene) -> np.ndarray:
        """
        Build observation vector (float32).
        """
        # --- last_visit(all users) ---
        lv = scene.last_visit_array().astype(np.float32)  # (N,)

        # --- uav_pos(flat) ---
        uav_pos = scene.uavs_pos().reshape(-1).astype(np.float32)  # (2*M,)

        if self.cfg.obs.mode == "lv+uav":
            obs = np.concatenate([lv, uav_pos], axis=0).astype(np.float32)
            return obs

        if self.cfg.obs.mode == "lv+uav+topk":
            k = int(self.cfg.obs.topk_oldest)
            topk_pos = self._topk_oldest_user_pos(scene, k)  # (2*k,)
            obs = np.concatenate([lv, uav_pos, topk_pos], axis=0).astype(np.float32)
            return obs

        raise ValueError(f"Unknown obs mode: {self.cfg.obs.mode}")

    def _topk_oldest_user_pos(self, scene: Scene, k: int) -> np.ndarray:
        """
        Return flattened positions (2*k,) of the top-k oldest users (largest last_visit).

        If users < k, pad with zeros.
        """
        if k <= 0:
            return np.zeros((0,), dtype=np.float32)

        N = len(scene.users)
        if N == 0:
            return np.zeros((2 * k,), dtype=np.float32)

        lv = scene.last_visit_array().astype(np.float32)  # (N,)

        # get top-k indices by last_visit (largest)
        if N <= k:
            idx = np.argsort(-lv)  # descending
        else:
            # argpartition: O(N) selection, then sort those k
            part = np.argpartition(-lv, kth=k - 1)[:k]
            idx = part[np.argsort(-lv[part])]

        pos = scene.users_pos().astype(np.float32)  # (N,2)
        picked = pos[idx]  # (min(k,N),2)

        out = np.zeros((k, 2), dtype=np.float32)
        out[: picked.shape[0], :] = picked
        return out.reshape(-1)