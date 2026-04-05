from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from uavdiff.config import ExpCfg, CFG
from uavdiff.condition import ConditionBatch
from uavdiff.diffusion_core import (
    sinusoidal_time_embedding,
    DiffusionSchedule,
    p_sample_loop,
    p_sample_loop_with_guidance,
    ActionRange,
    unit_to_env,
    clip_unit_action,
)


# ============================================================
# Small utility modules
# ============================================================
class MLP(nn.Module):
    """
    Simple 2-layer MLP.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout_p: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TokenProjection(nn.Module):
    """
    Project token features into common model dimension.
    """

    def __init__(self, in_dim: int, model_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ============================================================
# Transformer block with cross-attention
# ============================================================
class DiffusionTransformerBlock(nn.Module):
    """
    One block:
        1) self-attention on action tokens
        2) cross-attention from action tokens to condition tokens
        3) feed-forward

    Notes
    -----
    - action tokens are the "query sequence"
    - condition tokens provide context
    """

    def __init__(
        self,
        model_dim: int,
        n_heads: int,
        dropout_p: float = 0.0,
        ff_mult: int = 4,
    ):
        super().__init__()
        assert model_dim > 0
        assert n_heads > 0

        self.norm1 = nn.LayerNorm(model_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=n_heads,
            dropout=dropout_p,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(model_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=n_heads,
            dropout=dropout_p,
            batch_first=True,
        )

        self.norm3 = nn.LayerNorm(model_dim)
        self.ff = nn.Sequential(
            nn.Linear(model_dim, ff_mult * model_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(ff_mult * model_dim, model_dim),
            nn.Dropout(dropout_p),
        )

    def forward(
        self,
        x: torch.Tensor,
        cond_tokens: torch.Tensor,
    ) -> torch.Tensor:
        # self-attention on action tokens
        x1 = self.norm1(x)
        sa_out, _ = self.self_attn(x1, x1, x1, need_weights=False)
        x = x + sa_out

        # cross-attention: action queries attend to condition tokens
        x2 = self.norm2(x)
        ca_out, _ = self.cross_attn(x2, cond_tokens, cond_tokens, need_weights=False)
        x = x + ca_out

        # feed-forward
        x3 = self.norm3(x)
        x = x + self.ff(x3)
        return x


# ============================================================
# Condition encoder
# ============================================================
class ConditionEncoder(nn.Module):
    """
    Encode ConditionBatch into a unified condition-token sequence.

    Inputs may include:
        - sensor_tokens
        - uav_tokens
        - topk_tokens
        - global_token

    Each token type may have different feature dimension, so each gets its own
    input projection into the common model dimension.
    """

    def __init__(
        self,
        sensor_in_dim: Optional[int],
        uav_in_dim: Optional[int],
        topk_in_dim: Optional[int],
        global_in_dim: Optional[int],
        model_dim: int,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.model_dim = int(model_dim)

        self.sensor_proj = TokenProjection(sensor_in_dim, model_dim) if sensor_in_dim is not None else None
        self.uav_proj = TokenProjection(uav_in_dim, model_dim) if uav_in_dim is not None else None
        self.topk_proj = TokenProjection(topk_in_dim, model_dim) if topk_in_dim is not None else None
        self.global_proj = TokenProjection(global_in_dim, model_dim) if global_in_dim is not None else None

        self.dropout = nn.Dropout(dropout_p)
        self.out_norm = nn.LayerNorm(model_dim)

        # type embeddings
        self.sensor_type = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02) if sensor_in_dim is not None else None
        self.uav_type = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02) if uav_in_dim is not None else None
        self.topk_type = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02) if topk_in_dim is not None else None
        self.global_type = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02) if global_in_dim is not None else None

    def forward(self, cond: ConditionBatch) -> torch.Tensor:
        parts: List[torch.Tensor] = []

        if self.sensor_proj is not None and cond.sensor_tokens is not None:
            x = self.sensor_proj(cond.sensor_tokens)
            x = x + self.sensor_type
            parts.append(x)

        if self.uav_proj is not None and cond.uav_tokens is not None:
            x = self.uav_proj(cond.uav_tokens)
            x = x + self.uav_type
            parts.append(x)

        if self.topk_proj is not None and cond.topk_tokens is not None:
            x = self.topk_proj(cond.topk_tokens)
            x = x + self.topk_type
            parts.append(x)

        if self.global_proj is not None and cond.global_token is not None:
            x = self.global_proj(cond.global_token)
            x = x + self.global_type
            parts.append(x)

        if len(parts) == 0:
            raise ValueError("ConditionEncoder received no valid condition tokens.")

        out = torch.cat(parts, dim=1)
        out = self.dropout(out)
        out = self.out_norm(out)
        return out


# ============================================================
# Diffusion actor
# ============================================================
class DiffusionActor(nn.Module):
    """
    Conditional diffusion actor.

    Input
    -----
    x_t:
        noisy normalized action chunk, shape (B, H, act_dim)

    t:
        diffusion timestep, shape (B,)

    cond:
        ConditionBatch

    Output
    ------
    pred:
        shape (B, H, act_dim)
        interpreted as:
            - epsilon prediction if predict_type == "eps"
            - x0 prediction if predict_type == "x0"

    Design notes
    ------------
    - action tokens are the primary sequence
    - condition enters through cross-attention
    - time embedding is injected into all action tokens
    """

    def __init__(
        self,
        act_dim: int,
        chunk_len: int,
        model_dim: int = 256,
        time_embed_dim: int = 128,
        cond_dim: int = 256,
        n_blocks: int = 4,
        n_heads: int = 4,
        dropout_p: float = 0.0,
        sensor_in_dim: Optional[int] = 1,
        uav_in_dim: Optional[int] = 2,
        topk_in_dim: Optional[int] = 2,
        global_in_dim: Optional[int] = 2,
    ):
        super().__init__()
        assert act_dim > 0
        assert chunk_len > 0
        assert model_dim > 0
        assert time_embed_dim > 0
        assert n_blocks >= 1
        assert n_heads >= 1

        self.act_dim = int(act_dim)
        self.chunk_len = int(chunk_len)
        self.model_dim = int(model_dim)
        self.time_embed_dim = int(time_embed_dim)

        # action token input projection
        self.action_in = nn.Linear(self.act_dim, self.model_dim)

        # learnable horizon positional embedding
        self.pos_emb = nn.Parameter(torch.randn(1, self.chunk_len, self.model_dim) * 0.02)

        # diffusion timestep embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.model_dim),
        )

        # condition encoder
        self.cond_encoder = ConditionEncoder(
            sensor_in_dim=sensor_in_dim,
            uav_in_dim=uav_in_dim,
            topk_in_dim=topk_in_dim,
            global_in_dim=global_in_dim,
            model_dim=self.model_dim,
            dropout_p=dropout_p,
        )

        # transformer blocks
        self.blocks = nn.ModuleList([
            DiffusionTransformerBlock(
                model_dim=self.model_dim,
                n_heads=n_heads,
                dropout_p=dropout_p,
            )
            for _ in range(n_blocks)
        ])

        self.out_norm = nn.LayerNorm(self.model_dim)
        self.out_head = nn.Linear(self.model_dim, self.act_dim)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: ConditionBatch,
    ) -> torch.Tensor:
        """
        x_t: (B, H, act_dim)
        t:   (B,)
        """
        assert x_t.ndim == 3, f"x_t must be 3D, got {x_t.shape}"
        assert x_t.shape[-1] == self.act_dim, (
            f"act_dim mismatch: got {x_t.shape[-1]}, expected {self.act_dim}"
        )
        assert x_t.shape[1] == self.chunk_len, (
            f"chunk_len mismatch: got {x_t.shape[1]}, expected {self.chunk_len}"
        )
        assert t.ndim == 1 and t.shape[0] == x_t.shape[0], (
            f"t shape mismatch: got {t.shape}, batch size {x_t.shape[0]}"
        )

        # encode noisy action chunk as action tokens
        x = self.action_in(x_t)
        x = x + self.pos_emb

        # inject diffusion timestep embedding into all action tokens
        t_emb = sinusoidal_time_embedding(t, self.time_embed_dim)   # (B, time_embed_dim)
        t_emb = self.time_mlp(t_emb)                                # (B, model_dim)
        x = x + t_emb[:, None, :]

        # encode condition tokens
        cond_tokens = self.cond_encoder(cond)                       # (B, Nc, model_dim)

        # transformer stack
        for blk in self.blocks:
            x = blk(x, cond_tokens)

        x = self.out_norm(x)
        pred = self.out_head(x)                                     # (B, H, act_dim)
        return pred


# ============================================================
# Full policy wrapper
# ============================================================
class DiffusionPolicy(nn.Module):
    """
    High-level diffusion policy wrapper.

    Responsibilities
    ----------------
    - hold the denoising model
    - hold diffusion schedule
    - sample normalized chunk and convert to env action chunk

    This wrapper does NOT build condition itself.
    ConditionBatch should be prepared by condition.py.
    """

    def __init__(
        self,
        cfg: Optional[ExpCfg],
        act_dim: int,
        chunk_len: int,
        sensor_in_dim: Optional[int] = 1,
        uav_in_dim: Optional[int] = 2,
        topk_in_dim: Optional[int] = 2,
        global_in_dim: Optional[int] = 2,
    ):
        super().__init__()
        self.cfg = cfg or CFG
        self.cfg.validate()

        self.act_dim = int(act_dim)
        self.chunk_len = int(chunk_len)

        self.action_range = ActionRange(
            low=float(self.cfg.diffusion.action_low),
            high=float(self.cfg.diffusion.action_high),
        )

        self.schedule = DiffusionSchedule(
            n_steps=int(self.cfg.diffusion.train_diffusion_steps),
            schedule=self.cfg.diffusion.beta_schedule,
        )

        self.model = DiffusionActor(
            act_dim=self.act_dim,
            chunk_len=self.chunk_len,
            model_dim=int(self.cfg.diffusion.hidden_dim),
            time_embed_dim=int(self.cfg.diffusion.time_embed_dim),
            cond_dim=int(self.cfg.diffusion.cond_dim),
            n_blocks=int(self.cfg.diffusion.n_blocks),
            n_heads=int(self.cfg.diffusion.n_heads),
            dropout_p=float(self.cfg.diffusion.dropout_p),
            sensor_in_dim=sensor_in_dim,
            uav_in_dim=uav_in_dim,
            topk_in_dim=topk_in_dim,
            global_in_dim=global_in_dim,
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: ConditionBatch) -> torch.Tensor:
        return self.model(x_t, t, cond)

    @torch.no_grad()
    def sample_chunk_unit(
        self,
        cond: ConditionBatch,
        deterministic: bool = False,
        return_all: bool = False,
    ):
        """
        Sample a normalized action chunk in [-1,1].

        deterministic:
            Currently only controls initial noise:
            - False: random init noise
            - True: zero init noise

        return_all:
            If True, also return diffusion trajectory
        """
        device = cond.flat_obs.device
        bsz = cond.flat_obs.shape[0]

        init_noise = None
        if deterministic:
            init_noise = torch.zeros(
                (bsz, self.chunk_len, self.act_dim),
                device=device,
                dtype=torch.float32,
            )

        out = p_sample_loop(
            model=self.model,
            sched=self.schedule,
            shape=torch.Size([bsz, self.chunk_len, self.act_dim]),
            cond=cond,
            device=device,
            predict_type=self.cfg.diffusion.predict_type,
            clip_x0=True,
            init_noise=init_noise,
            return_all=return_all,
        )
        return out

    @torch.no_grad()
    def sample_chunk_env(
        self,
        cond: ConditionBatch,
        deterministic: bool = False,
        return_unit: bool = False,
    ):
        """
        Sample env-range action chunk.

        Returns
        -------
        a_env:
            (B, H, act_dim) in env action range
        optionally:
            a_unit:
                normalized chunk in [-1,1]
        """
        a_unit = self.sample_chunk_unit(cond=cond, deterministic=deterministic, return_all=False)
        a_unit = clip_unit_action(a_unit)
        a_env = unit_to_env(a_unit, self.action_range)

        if return_unit:
            return a_env, a_unit
        return a_env

    @torch.no_grad()
    def sample_chunk_env_with_guidance(
        self,
        cond: ConditionBatch,
        guide_fn=None,
        guide_scale: float = 0.0,
        deterministic: bool = False,
        return_unit: bool = False,
    ):
        """
        Optional guided sampling.
        First version usually does not need this.
        """
        device = cond.flat_obs.device
        bsz = cond.flat_obs.shape[0]

        init_noise = None
        if deterministic:
            init_noise = torch.zeros(
                (bsz, self.chunk_len, self.act_dim),
                device=device,
                dtype=torch.float32,
            )

        a_unit = p_sample_loop_with_guidance(
            model=self.model,
            sched=self.schedule,
            shape=torch.Size([bsz, self.chunk_len, self.act_dim]),
            cond=cond,
            device=device,
            predict_type=self.cfg.diffusion.predict_type,
            clip_x0=True,
            init_noise=init_noise,
            guide_fn=guide_fn,
            guide_scale=guide_scale,
        )

        a_unit = clip_unit_action(a_unit)
        a_env = unit_to_env(a_unit, self.action_range)

        if return_unit:
            return a_env, a_unit
        return a_env