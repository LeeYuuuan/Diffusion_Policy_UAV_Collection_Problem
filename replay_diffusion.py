from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch

from config import ExpCfg, CFG


# ============================================================
# Batch dataclasses
# ============================================================
@dataclass
class StepBatch:
    """
    Batch for critic update.

    Shapes:
        obs:          (B, obs_dim)
        act:          (B, act_dim)
        rew:          (B, 1)
        next_obs:     (B, obs_dim)
        done:         (B, 1)

        obs_hist:     (B, Hs, obs_dim)
        next_obs_hist:(B, Hs, obs_dim)

    Notes:
    - obs_hist / next_obs_hist are provided so critic-side adapter can build
      K-state critic inputs without needing to reconstruct them elsewhere.
    - Hs = critic_state_horizon by default.
    """
    obs: torch.Tensor
    act: torch.Tensor
    rew: torch.Tensor
    next_obs: torch.Tensor
    done: torch.Tensor
    obs_hist: torch.Tensor
    next_obs_hist: torch.Tensor


@dataclass
class ChunkBatch:
    """
    Batch for diffusion actor update.

    Shapes:
        obs0:          (B, obs_dim)
        obs_hist:      (B, Ho, obs_dim)
        act_chunk:     (B, Hc, act_dim)
        rew_chunk:     (B, Hc, 1)
        done_chunk:    (B, Hc, 1)
        next_obs_chunk:(B, Hc, obs_dim)

    Meanings:
    - obs0:
        starting observation at chunk start t
    - obs_hist:
        actor-side observation history ending at t
    - act_chunk:
        [a_t, a_{t+1}, ..., a_{t+Hc-1}]
    """
    obs0: torch.Tensor
    obs_hist: torch.Tensor
    act_chunk: torch.Tensor
    rew_chunk: torch.Tensor
    done_chunk: torch.Tensor
    next_obs_chunk: torch.Tensor


# ============================================================
# Replay buffer
# ============================================================
class ReplayBufferDiffusion:
    """
    Unified replay buffer for diffusion RL.

    Core idea:
    - Store step transitions only.
    - Sample step batches for critic.
    - Sample consecutive chunks for diffusion actor.

    This keeps the design flexible and avoids maintaining two separate stores.

    Stored fields per step:
        obs_t
        act_t
        rew_t
        next_obs_t
        done_t
        episode_id
        step_in_episode

    Important:
    - If allow_cross_episode_chunk=False, chunk sampling will never cross
      episode boundaries.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        capacity: int,
        chunk_len: int,
        critic_state_horizon: int = 1,
        actor_obs_horizon: int = 1,
        device: str = "cpu",
        allow_cross_episode_chunk: bool = False,
    ):
        assert obs_dim > 0
        assert act_dim > 0
        assert capacity > 0
        assert chunk_len >= 1
        assert critic_state_horizon >= 1
        assert actor_obs_horizon >= 1

        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.capacity = int(capacity)
        self.chunk_len = int(chunk_len)
        self.critic_state_horizon = int(critic_state_horizon)
        self.actor_obs_horizon = int(actor_obs_horizon)
        self.allow_cross_episode_chunk = bool(allow_cross_episode_chunk)

        self.device = torch.device(device)

        # step-level storage
        self.obs_buf = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((self.capacity, self.act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_obs_buf = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.done_buf = np.zeros((self.capacity, 1), dtype=np.float32)

        # episode bookkeeping
        self.episode_id_buf = np.full((self.capacity,), fill_value=-1, dtype=np.int64)
        self.step_in_ep_buf = np.full((self.capacity,), fill_value=-1, dtype=np.int64)

        self.ptr = 0
        self.size = 0

        # runtime episode counter, advanced by external caller or auto helper
        self.current_episode_id = 0
        self.current_step_in_episode = 0

    # ============================================================
    # Basic API
    # ============================================================
    def __len__(self) -> int:
        return self.size

    def to(self, device: str) -> "ReplayBufferDiffusion":
        self.device = torch.device(device)
        return self

    def start_new_episode(self) -> None:
        """
        Call this at the beginning of each episode before storing its first step.
        """
        if self.size == 0 and self.current_step_in_episode == 0 and self.current_episode_id == 0:
            # very first episode: keep episode_id = 0
            return
        self.current_episode_id += 1
        self.current_step_in_episode = 0

    def store_step(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
        episode_id: Optional[int] = None,
        step_in_episode: Optional[int] = None,
    ) -> None:
        """
        Store one environment transition.

        If episode_id / step_in_episode are not provided, internal counters are used.
        """
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
        act = np.asarray(act, dtype=np.float32).reshape(-1)

        assert obs.shape[0] == self.obs_dim, f"obs_dim mismatch: {obs.shape[0]} vs {self.obs_dim}"
        assert next_obs.shape[0] == self.obs_dim, f"next_obs_dim mismatch: {next_obs.shape[0]} vs {self.obs_dim}"
        assert act.shape[0] == self.act_dim, f"act_dim mismatch: {act.shape[0]} vs {self.act_dim}"

        ep_id = self.current_episode_id if episode_id is None else int(episode_id)
        ep_step = self.current_step_in_episode if step_in_episode is None else int(step_in_episode)

        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr, 0] = float(rew)
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr, 0] = 1.0 if bool(done) else 0.0

        self.episode_id_buf[self.ptr] = ep_id
        self.step_in_ep_buf[self.ptr] = ep_step

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        if step_in_episode is None:
            self.current_step_in_episode += 1

    # ============================================================
    # Sampling: step batch for critic
    # ============================================================
    def sample_step_batch(self, batch_size: int) -> StepBatch:
        assert self.size > 0, "ReplayBufferDiffusion is empty."
        assert batch_size > 0

        valid_idx = self._valid_step_indices()
        assert len(valid_idx) > 0, "No valid step indices available for sampling."

        idx = np.random.choice(valid_idx, size=batch_size, replace=True)

        obs = self.obs_buf[idx]
        act = self.act_buf[idx]
        rew = self.rew_buf[idx]
        next_obs = self.next_obs_buf[idx]
        done = self.done_buf[idx]

        obs_hist = self._gather_obs_history(idx, hist_len=self.critic_state_horizon, use_next_obs=False)
        next_obs_hist = self._gather_next_obs_history(idx, hist_len=self.critic_state_horizon)

        return StepBatch(
            obs=self._to_torch(obs),
            act=self._to_torch(act),
            rew=self._to_torch(rew),
            next_obs=self._to_torch(next_obs),
            done=self._to_torch(done),
            obs_hist=self._to_torch(obs_hist),
            next_obs_hist=self._to_torch(next_obs_hist),
        )

    # ============================================================
    # Sampling: chunk batch for diffusion actor
    # ============================================================
    def sample_chunk_batch(self, batch_size: int, chunk_len: Optional[int] = None) -> ChunkBatch:
        assert self.size > 0, "ReplayBufferDiffusion is empty."
        assert batch_size > 0

        h = int(self.chunk_len if chunk_len is None else chunk_len)
        assert h >= 1

        valid_start_idx = self._valid_chunk_start_indices(chunk_len=h)
        assert len(valid_start_idx) > 0, "No valid chunk start indices available."

        starts = np.random.choice(valid_start_idx, size=batch_size, replace=True)

        obs0 = self.obs_buf[starts]
        obs_hist = self._gather_obs_history(starts, hist_len=self.actor_obs_horizon, use_next_obs=False)

        act_chunk = np.zeros((batch_size, h, self.act_dim), dtype=np.float32)
        rew_chunk = np.zeros((batch_size, h, 1), dtype=np.float32)
        done_chunk = np.zeros((batch_size, h, 1), dtype=np.float32)
        next_obs_chunk = np.zeros((batch_size, h, self.obs_dim), dtype=np.float32)

        for b, s in enumerate(starts):
            ids = np.arange(s, s + h, dtype=np.int64)
            act_chunk[b] = self.act_buf[ids]
            rew_chunk[b] = self.rew_buf[ids]
            done_chunk[b] = self.done_buf[ids]
            next_obs_chunk[b] = self.next_obs_buf[ids]

        return ChunkBatch(
            obs0=self._to_torch(obs0),
            obs_hist=self._to_torch(obs_hist),
            act_chunk=self._to_torch(act_chunk),
            rew_chunk=self._to_torch(rew_chunk),
            done_chunk=self._to_torch(done_chunk),
            next_obs_chunk=self._to_torch(next_obs_chunk),
        )

    # ============================================================
    # Validity helpers
    # ============================================================
    def _valid_step_indices(self) -> np.ndarray:
        """
        Return valid indices for single-step critic sampling.

        Current first version:
        - any filled slot is valid
        """
        if self.size < 1:
            return np.zeros((0,), dtype=np.int64)
        return np.arange(self.size, dtype=np.int64)

    def _valid_chunk_start_indices(self, chunk_len: int) -> np.ndarray:
        """
        Return valid start indices for chunk sampling.

        Conditions:
        - start + chunk_len - 1 must be within currently filled region [0, size-1]
        - if allow_cross_episode_chunk=False, all steps in the chunk must share
          the same episode_id
        """
        if self.size < chunk_len:
            return np.zeros((0,), dtype=np.int64)

        max_start = self.size - chunk_len
        valid: List[int] = []

        for s in range(max_start + 1):
            e = s + chunk_len - 1

            if not self.allow_cross_episode_chunk:
                ep0 = self.episode_id_buf[s]
                ep1 = self.episode_id_buf[e]
                if ep0 < 0 or ep1 < 0 or ep0 != ep1:
                    continue

                # stronger check: all in same episode
                seg = self.episode_id_buf[s:e + 1]
                if not np.all(seg == ep0):
                    continue

            valid.append(s)

        return np.asarray(valid, dtype=np.int64)

    # ============================================================
    # History gathering
    # ============================================================
    def _gather_obs_history(
        self,
        idx: np.ndarray,
        hist_len: int,
        use_next_obs: bool = False,
    ) -> np.ndarray:
        """
        Gather observation history ending at the current step index.

        For each sampled step i:
            history = [obs_{t-h+1}, ..., obs_t]
        padded on the left by repeating the earliest available obs within episode.

        If use_next_obs=True:
            replace the final element by next_obs_t is NOT done here.
            This function is only for obs-history ending at current t.
        """
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        out = np.zeros((len(idx), hist_len, self.obs_dim), dtype=np.float32)

        for b, i in enumerate(idx):
            ep = self.episode_id_buf[i]
            step_in_ep = self.step_in_ep_buf[i]

            hist_indices: List[int] = []
            for delta in range(hist_len - 1, -1, -1):
                target_step = step_in_ep - delta
                if target_step < 0:
                    target_step = 0

                j = i - (step_in_ep - target_step)
                if j < 0 or self.episode_id_buf[j] != ep:
                    j = i - step_in_ep  # first step of this episode

                hist_indices.append(j)

            hist_arr = self.obs_buf[np.asarray(hist_indices, dtype=np.int64)]
            out[b] = hist_arr

        return out

    def _gather_next_obs_history(
        self,
        idx: np.ndarray,
        hist_len: int,
    ) -> np.ndarray:
        """
        Gather next-observation history for target critic input.

        For each sampled step i:
            next_history corresponds conceptually to history ending at s_{t+1}.

        A simple and consistent first-version construction:
            [obs_{t-h+2}, ..., obs_t, next_obs_t]

        with left padding when needed.
        """
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        out = np.zeros((len(idx), hist_len, self.obs_dim), dtype=np.float32)

        for b, i in enumerate(idx):
            cur_hist = self._gather_obs_history(np.asarray([i]), hist_len=max(hist_len - 1, 1))[0]

            if hist_len == 1:
                out[b, 0] = self.next_obs_buf[i]
            else:
                out[b, :-1] = cur_hist[-(hist_len - 1):]
                out[b, -1] = self.next_obs_buf[i]

        return out

    # ============================================================
    # Utilities
    # ============================================================
    def _to_torch(self, x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def state_dict_meta(self) -> Dict[str, int | bool]:
        return {
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "capacity": self.capacity,
            "chunk_len": self.chunk_len,
            "critic_state_horizon": self.critic_state_horizon,
            "actor_obs_horizon": self.actor_obs_horizon,
            "allow_cross_episode_chunk": self.allow_cross_episode_chunk,
            "size": self.size,
            "ptr": self.ptr,
            "current_episode_id": self.current_episode_id,
            "current_step_in_episode": self.current_step_in_episode,
        }