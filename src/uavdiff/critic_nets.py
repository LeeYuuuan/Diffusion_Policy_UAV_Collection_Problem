from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from uavdiff.config import ExpCfg, CFG


@dataclass
class CriticInputSpec:
    """
    Critic input specification.

    state_dim:
        Dimension of the critic state input after history stacking.

    act_dim:
        Dimension of the action vector used by critic.

    Notes
    -----
    In the first diffusion version, critic evaluates:
        Q(s_hist, a_first)

    where:
        s_hist is the stacked observation history
        a_first is the first action of the generated chunk
    """
    state_dim: int
    act_dim: int


class MLPBlock(nn.Module):
    """
    Simple MLP block:
        Linear -> ReLU -> Dropout
    """

    def __init__(self, in_dim: int, out_dim: int, dropout_p: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CriticHead(nn.Module):
    """
    Single Q-network:
        Q(state, action) -> scalar
    """

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        assert state_dim > 0
        assert act_dim > 0
        assert hidden_dim > 0

        in_dim = state_dim + act_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a], dim=-1)
        return self.net(x)


class CriticTwinStep(nn.Module):
    """
    Twin Q-networks for the first diffusion version.

    Input
    -----
    s_hist_flat:
        shape (B, state_dim)
        Usually this is the flattened stacked observation history:
            [obs_{t-K+1}, ..., obs_t]

    a_first:
        shape (B, act_dim)
        The first action of the action chunk.

    Output
    ------
    q1, q2:
        shape (B, 1), (B, 1)

    Why this design
    ---------------
    This is the safest first version:
    - keep critic simple
    - still allow multi-state input via state stacking
    - compatible with Q-guided diffusion actor
    """

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        assert state_dim > 0
        assert act_dim > 0

        self.state_dim = int(state_dim)
        self.act_dim = int(act_dim)

        self.q1 = CriticHead(
            state_dim=self.state_dim,
            act_dim=self.act_dim,
            hidden_dim=hidden_dim,
            dropout_p=dropout_p,
        )
        self.q2 = CriticHead(
            state_dim=self.state_dim,
            act_dim=self.act_dim,
            hidden_dim=hidden_dim,
            dropout_p=dropout_p,
        )

    def forward(self, s_hist_flat: torch.Tensor, a_first: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert s_hist_flat.ndim == 2, f"s_hist_flat must be 2D, got {s_hist_flat.shape}"
        assert a_first.ndim == 2, f"a_first must be 2D, got {a_first.shape}"
        assert s_hist_flat.shape[-1] == self.state_dim, (
            f"critic state dim mismatch: got {s_hist_flat.shape[-1]}, expected {self.state_dim}"
        )
        assert a_first.shape[-1] == self.act_dim, (
            f"critic action dim mismatch: got {a_first.shape[-1]}, expected {self.act_dim}"
        )

        q1 = self.q1(s_hist_flat, a_first)
        q2 = self.q2(s_hist_flat, a_first)
        return q1, q2

    def q_min(self, s_hist_flat: torch.Tensor, a_first: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(s_hist_flat, a_first)
        return torch.min(q1, q2)


class CriticFactory:
    """
    Helper factory for critic construction.

    Current first version only supports:
        mode = "state_action"

    but we keep a factory so later we can add:
        - sequence critic
        - chunk critic
        - transformer critic
    """

    @staticmethod
    def build(cfg: Optional[ExpCfg], obs_dim: int, act_dim: int) -> CriticTwinStep:
        cfg = cfg or CFG
        cfg.validate()

        state_horizon = int(cfg.critic.state_horizon)
        state_dim = state_horizon * int(obs_dim)

        return CriticTwinStep(
            state_dim=state_dim,
            act_dim=int(act_dim),
            hidden_dim=int(cfg.critic.hidden_dim),
            dropout_p=float(cfg.critic.dropout_p),
        )


# ============================================================
# Utility functions
# ============================================================
def flatten_state_history(obs_hist: torch.Tensor) -> torch.Tensor:
    """
    Flatten critic observation history.

    Input
    -----
    obs_hist:
        shape (B, K, obs_dim)

    Output
    ------
    shape (B, K * obs_dim)

    Notes
    -----
    This matches the first-version critic design:
        critic consumes concatenated observation history.
    """
    assert obs_hist.ndim == 3, f"obs_hist must be 3D, got {obs_hist.shape}"
    bsz = obs_hist.shape[0]
    return obs_hist.reshape(bsz, -1)


def extract_first_action(act_chunk: torch.Tensor) -> torch.Tensor:
    """
    Extract the first action from an action chunk.

    Input
    -----
    act_chunk:
        shape (B, H, act_dim)

    Output
    ------
    a_first:
        shape (B, act_dim)
    """
    assert act_chunk.ndim == 3, f"act_chunk must be 3D, got {act_chunk.shape}"
    return act_chunk[:, 0, :]


@torch.no_grad()
def soft_update(src: nn.Module, dst: nn.Module, tau: float) -> None:
    """
    Soft update:
        dst <- (1 - tau) * dst + tau * src
    """
    assert 0.0 < tau <= 1.0
    for p_t, p in zip(dst.parameters(), src.parameters()):
        p_t.data.mul_(1.0 - tau).add_(tau * p.data)