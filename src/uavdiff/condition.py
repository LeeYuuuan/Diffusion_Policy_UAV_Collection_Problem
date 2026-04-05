from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, List

import numpy as np
import torch

from uavdiff.config import ExpCfg, CFG


@dataclass
class ParsedObs:
    """
    Parsed semantic view of one flat observation.

    Shapes below are for a single observation (not batched):
        sensor_values: (N,)
        uav_pos:       (M, 2)
        topk_pos:      (K, 2) or None
    """
    sensor_values: np.ndarray
    uav_pos: np.ndarray
    topk_pos: Optional[np.ndarray]
    flat_obs: np.ndarray


@dataclass
class ConditionBatch:
    """
    Model-facing structured condition batch.

    All tensors are on the target device.

    Shapes:
        flat_obs:       (B, obs_dim)
        sensor_tokens:  (B, N, Ds) or None
        uav_tokens:     (B, M, Du) or None
        topk_tokens:    (B, K, Dt) or None
        global_token:   (B, 1, Dg) or None
        critic_obs:     (B, critic_in_dim)

    Notes:
    - Feature dimensions Ds / Du / Dt / Dg are adapter-defined and may differ.
    - The actor can later project them into a common embedding space.
    """
    flat_obs: torch.Tensor
    sensor_tokens: Optional[torch.Tensor]
    uav_tokens: Optional[torch.Tensor]
    topk_tokens: Optional[torch.Tensor]
    global_token: Optional[torch.Tensor]
    critic_obs: torch.Tensor


class ObsLayout:
    """
    Layout helper for parsing flat observations.

    This class knows how current flat obs is organized according to:
        - cfg.obs.mode
        - cfg.user_gen.n_users
        - cfg.env.n_uavs
        - cfg.obs.topk_oldest

    Current supported flat formats:
        1) lv+uav
        2) lv+uav+topk
    """

    def __init__(self, cfg: ExpCfg):
        self.cfg = cfg
        self.n_users = int(cfg.user_gen.n_users)
        self.n_uavs = int(cfg.env.n_uavs)
        self.obs_mode = cfg.obs.mode
        self.topk = int(cfg.obs.topk_oldest)

        self._build_layout()

    def _build_layout(self) -> None:
        base = self.n_users + 2 * self.n_uavs

        if self.obs_mode == "lv+uav":
            self.obs_dim = base
            self.idx_sensor_start = 0
            self.idx_sensor_end = self.n_users

            self.idx_uav_start = self.idx_sensor_end
            self.idx_uav_end = self.idx_uav_start + 2 * self.n_uavs

            self.idx_topk_start = self.idx_uav_end
            self.idx_topk_end = self.idx_topk_start
            return

        if self.obs_mode == "lv+uav+topk":
            self.obs_dim = base + 2 * self.topk
            self.idx_sensor_start = 0
            self.idx_sensor_end = self.n_users

            self.idx_uav_start = self.idx_sensor_end
            self.idx_uav_end = self.idx_uav_start + 2 * self.n_uavs

            self.idx_topk_start = self.idx_uav_end
            self.idx_topk_end = self.idx_topk_start + 2 * self.topk
            return

        raise ValueError(f"Unknown obs mode: {self.obs_mode}")

    def parse_single_np(self, obs: np.ndarray) -> ParsedObs:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        assert obs.shape[0] == self.obs_dim, (
            f"obs dim mismatch: got {obs.shape[0]}, expected {self.obs_dim}"
        )

        sensor_values = obs[self.idx_sensor_start:self.idx_sensor_end].copy()
        uav_pos = obs[self.idx_uav_start:self.idx_uav_end].reshape(self.n_uavs, 2).copy()

        topk_pos = None
        if self.obs_mode == "lv+uav+topk":
            topk_flat = obs[self.idx_topk_start:self.idx_topk_end]
            topk_pos = topk_flat.reshape(self.topk, 2).copy()

        return ParsedObs(
            sensor_values=sensor_values,
            uav_pos=uav_pos,
            topk_pos=topk_pos,
            flat_obs=obs.copy(),
        )

    def parse_batch_np(self, obs_batch: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Parse batch flat observations.

        Input:
            obs_batch: (B, obs_dim)

        Output dict:
            sensor_values: (B, N)
            uav_pos:       (B, M, 2)
            topk_pos:      (B, K, 2) if available
            flat_obs:      (B, obs_dim)
        """
        obs_batch = np.asarray(obs_batch, dtype=np.float32)
        if obs_batch.ndim == 1:
            obs_batch = obs_batch[None, :]
        assert obs_batch.ndim == 2
        assert obs_batch.shape[1] == self.obs_dim, (
            f"obs dim mismatch: got {obs_batch.shape[1]}, expected {self.obs_dim}"
        )

        out: Dict[str, np.ndarray] = {}
        out["flat_obs"] = obs_batch.copy()
        out["sensor_values"] = obs_batch[:, self.idx_sensor_start:self.idx_sensor_end].copy()
        out["uav_pos"] = obs_batch[:, self.idx_uav_start:self.idx_uav_end].reshape(-1, self.n_uavs, 2).copy()

        if self.obs_mode == "lv+uav+topk":
            out["topk_pos"] = obs_batch[:, self.idx_topk_start:self.idx_topk_end].reshape(-1, self.topk, 2).copy()

        return out


class ConditionBuilder:
    """
    Convert flat env observations into structured condition tensors.

    Design goals:
    1) Keep env side unchanged: env still outputs flat obs.
    2) Concentrate obs parsing logic here.
    3) Make future obs changes local:
        - modify obs.py to emit new flat obs
        - modify this file to parse/build condition
        - avoid touching actor / critic internals as much as possible
    """

    def __init__(self, cfg: Optional[ExpCfg] = None, device: str = "cpu"):
        self.cfg = cfg or CFG
        self.cfg.validate()
        self.layout = ObsLayout(self.cfg)
        self.device = torch.device(device)

        self.world_size = float(self.cfg.env.world_size)
        self.n_users = int(self.cfg.user_gen.n_users)
        self.n_uavs = int(self.cfg.env.n_uavs)
        self.topk = int(self.cfg.obs.topk_oldest)

    # ============================================================
    # Public helpers
    # ============================================================
    def obs_dim(self) -> int:
        return int(self.layout.obs_dim)

    def parse_single(self, obs: np.ndarray) -> ParsedObs:
        return self.layout.parse_single_np(obs)

    def parse_batch(self, obs_batch: np.ndarray) -> Dict[str, np.ndarray]:
        return self.layout.parse_batch_np(obs_batch)

    # ============================================================
    # Actor-side condition construction
    # ============================================================
    def build_condition(
        self,
        obs_batch: np.ndarray | torch.Tensor,
        obs_history: Optional[np.ndarray | torch.Tensor] = None,
        critic_obs_history: Optional[np.ndarray | torch.Tensor] = None,
    ) -> ConditionBatch:
        """
        Build model-facing structured condition.

        Parameters
        ----------
        obs_batch:
            Current flat observation batch, shape (B, obs_dim) or (obs_dim,)

        obs_history:
            Optional actor-side observation history.
            Reserved for future extension. Current first version does not yet
            encode multi-step actor history into tokens, but the interface is kept.

        critic_obs_history:
            Optional observation history for critic input.
            If provided, this is preferred over obs_batch when building critic_obs.

        Returns
        -------
        ConditionBatch
        """
        obs_np = self._to_numpy_2d(obs_batch)
        parsed = self.parse_batch(obs_np)

        flat_obs_t = self._to_torch(parsed["flat_obs"])

        sensor_tokens = None
        if self.cfg.adapter.include_sensor_tokens:
            sensor_tokens_np = self._build_sensor_tokens(parsed)
            sensor_tokens = self._to_torch(sensor_tokens_np)

        uav_tokens = None
        if self.cfg.adapter.include_uav_tokens:
            uav_tokens_np = self._build_uav_tokens(parsed)
            uav_tokens = self._to_torch(uav_tokens_np)

        topk_tokens = None
        if self.cfg.adapter.include_topk_tokens and "topk_pos" in parsed:
            topk_tokens_np = self._build_topk_tokens(parsed)
            topk_tokens = self._to_torch(topk_tokens_np)

        global_token = None
        if self.cfg.adapter.include_global_token:
            global_token_np = self._build_global_token(parsed)
            global_token = self._to_torch(global_token_np)

        critic_obs = self.build_critic_obs(
            obs_history=critic_obs_history if critic_obs_history is not None else obs_batch
        )

        return ConditionBatch(
            flat_obs=flat_obs_t,
            sensor_tokens=sensor_tokens,
            uav_tokens=uav_tokens,
            topk_tokens=topk_tokens,
            global_token=global_token,
            critic_obs=critic_obs,
        )

    # ============================================================
    # Critic-side input construction
    # ============================================================
    def critic_input_dim(self) -> int:
        """
        Critic input dimension after history stacking.

        Current strategy:
            critic_obs = concatenation of the last K flat observations
        where:
            K = cfg.critic.state_horizon
        """
        k = int(self.cfg.critic.state_horizon)
        return k * self.obs_dim()

    def build_critic_obs(
        self,
        obs_history: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """
        Build critic input using a configurable number of recent states.

        Supported inputs:
        1) Single obs:
            shape (obs_dim,) or (B, obs_dim)
        2) History:
            shape (T, obs_dim) or (B, T, obs_dim)

        Output:
            shape (B, K * obs_dim)

        Rule:
        - Use the last K observations.
        - If history length < K, pad by repeating the earliest available obs.
        """
        k = int(self.cfg.critic.state_horizon)
        obs_np = np.asarray(self._to_numpy(obs_history), dtype=np.float32)

        if obs_np.ndim == 1:
            obs_np = obs_np[None, None, :]  # (1,1,D)
        elif obs_np.ndim == 2:
            if obs_np.shape[1] == self.obs_dim():
                # interpret as either (B,D) or (T,D)
                # by convention here we treat it as batch current obs
                obs_np = obs_np[:, None, :]  # (B,1,D)
            else:
                raise ValueError(
                    f"Unexpected 2D obs_history shape: {obs_np.shape}, expected (*, {self.obs_dim()})"
                )
        elif obs_np.ndim == 3:
            pass
        else:
            raise ValueError(f"Unsupported obs_history ndim: {obs_np.ndim}")

        assert obs_np.shape[-1] == self.obs_dim(), (
            f"obs dim mismatch in critic history: got {obs_np.shape[-1]}, expected {self.obs_dim()}"
        )

        bsz, tlen, odim = obs_np.shape
        out = np.zeros((bsz, k, odim), dtype=np.float32)

        for b in range(bsz):
            hist = obs_np[b]  # (T,D)
            if tlen >= k:
                picked = hist[-k:]
            else:
                pad_n = k - tlen
                pad = np.repeat(hist[:1], repeats=pad_n, axis=0)
                picked = np.concatenate([pad, hist], axis=0)
            out[b] = picked

        out = out.reshape(bsz, k * odim)
        return self._to_torch(out)

    # ============================================================
    # Token builders
    # ============================================================
    def _build_sensor_tokens(self, parsed: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Build sensor tokens from current flat obs.

        Current first-version token fields:
            [ normalized_last_visit ]

        Shape:
            (B, N, 1)

        Notes:
        - Current flat obs does not contain explicit sensor coordinates except top-k.
        - Later, if obs includes more sensor attributes, extend here.
        """
        sensor_values = parsed["sensor_values"]  # (B,N)

        # Simple scaling. We keep it conservative and easy to swap later.
        scale = 1.0 + np.maximum(sensor_values.max(axis=1, keepdims=True), 1.0)
        norm_lv = sensor_values / scale

        return norm_lv[..., None].astype(np.float32)

    def _build_uav_tokens(self, parsed: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Build UAV tokens.

        Current token fields:
            [ x_norm, y_norm ]

        Shape:
            (B, M, 2)
        """
        uav_pos = parsed["uav_pos"]  # (B,M,2)
        out = uav_pos / max(self.world_size, 1.0)
        return out.astype(np.float32)

    def _build_topk_tokens(self, parsed: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Build top-k user-position tokens.

        Current token fields:
            [ x_norm, y_norm ]

        Shape:
            (B, K, 2)
        """
        topk_pos = parsed["topk_pos"]  # (B,K,2)
        out = topk_pos / max(self.world_size, 1.0)
        return out.astype(np.float32)

    def _build_global_token(self, parsed: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Build one global token per observation.

        Current token fields:
            [ max_last_visit, mean_last_visit ]

        Shape:
            (B, 1, 2)
        """
        sensor_values = parsed["sensor_values"]  # (B,N)

        max_lv = sensor_values.max(axis=1, keepdims=True)
        mean_lv = sensor_values.mean(axis=1, keepdims=True)

        # Per-batch simple normalization
        denom = 1.0 + np.maximum(max_lv, 1.0)
        max_lv_n = max_lv / denom
        mean_lv_n = mean_lv / denom

        tok = np.concatenate([max_lv_n, mean_lv_n], axis=1)[:, None, :]
        return tok.astype(np.float32)

    # ============================================================
    # Utility / conversion
    # ============================================================
    def _to_numpy(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x, dtype=np.float32)

    def _to_numpy_2d(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        x_np = self._to_numpy(x).astype(np.float32)
        if x_np.ndim == 1:
            x_np = x_np[None, :]
        assert x_np.ndim == 2, f"Expected 2D array, got shape {x_np.shape}"
        return x_np

    def _to_torch(self, x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)