from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Tuple

import math
import torch
import torch.nn as nn


PredictType = Literal["eps", "x0"]
BetaScheduleType = Literal["linear", "cosine"]


# ============================================================
# Action range helpers
# ============================================================
@dataclass
class ActionRange:
    """
    Action range helper.

    We keep diffusion internally in normalized action space:
        x in [-1, 1]

    while the environment action range is:
        a_env in [low, high]
    """
    low: float = -50.0
    high: float = 50.0

    @property
    def scale(self) -> float:
        return (self.high - self.low) / 2.0

    @property
    def bias(self) -> float:
        return (self.high + self.low) / 2.0


def env_to_unit(a_env: torch.Tensor, spec: ActionRange) -> torch.Tensor:
    """
    Map env-range action to normalized diffusion space [-1, 1].
    """
    a_unit = (a_env - spec.bias) / (spec.scale + 1e-8)
    return torch.clamp(a_unit, -1.0, 1.0)


def unit_to_env(a_unit: torch.Tensor, spec: ActionRange) -> torch.Tensor:
    """
    Map normalized action [-1,1] back to env range.
    """
    a_unit = torch.clamp(a_unit, -1.0, 1.0)
    return a_unit * spec.scale + spec.bias


def clip_unit_action(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, -1.0, 1.0)


# ============================================================
# Timestep embedding
# ============================================================
def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Standard sinusoidal timestep embedding.

    Input
    -----
    t:
        shape (B,) integer or float tensor

    Output
    ------
    emb:
        shape (B, dim)
    """
    assert dim > 0
    device = t.device
    half = dim // 2

    t = t.float()
    freq = torch.exp(
        -math.log(10_000.0) * torch.arange(0, half, device=device).float() / max(half - 1, 1)
    )
    args = t[:, None] * freq[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=device)], dim=-1)

    return emb


# ============================================================
# Beta schedules
# ============================================================
def make_beta_schedule(
    n_steps: int,
    schedule: BetaScheduleType = "cosine",
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    cosine_s: float = 0.008,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create beta schedule of length n_steps.

    Returns
    -------
    betas:
        shape (T,)
    """
    assert n_steps >= 2

    if schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, n_steps, device=device, dtype=torch.float32)
        return torch.clamp(betas, 1e-8, 0.999)

    if schedule == "cosine":
        steps = n_steps + 1
        x = torch.linspace(0, n_steps, steps, device=device, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / n_steps) + cosine_s) / (1 + cosine_s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 1e-8, 0.999)

    raise ValueError(f"Unknown beta schedule: {schedule}")


# ============================================================
# Diffusion schedule container
# ============================================================
class DiffusionSchedule(nn.Module):
    """
    Stores all precomputed diffusion coefficients.

    Convention
    ----------
    - timesteps are indexed as:
        t in {0, 1, ..., T-1}
    - x_0 is clean normalized action chunk
    - x_t is noisy chunk at timestep t
    """

    def __init__(
        self,
        n_steps: int,
        schedule: BetaScheduleType = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        super().__init__()
        assert n_steps >= 2
        self.n_steps = int(n_steps)
        self.schedule_type = schedule

        betas = make_beta_schedule(
            n_steps=n_steps,
            schedule=schedule,
            beta_start=beta_start,
            beta_end=beta_end,
        )

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=torch.float32), alphas_cumprod[:-1]], dim=0
        )

        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1.0)

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod + 1e-8)
        posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))

        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod + 1e-8)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod + 1e-8)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)
        self.register_buffer("sqrt_recip_alphas_cumprod", sqrt_recip_alphas_cumprod)
        self.register_buffer("sqrt_recipm1_alphas_cumprod", sqrt_recipm1_alphas_cumprod)

        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", posterior_log_variance_clipped)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    def extract(self, arr: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """
        Extract per-batch coefficients from shape (T,) buffer and reshape for broadcasting.

        Parameters
        ----------
        arr:
            shape (T,)
        t:
            shape (B,)
        x_shape:
            target tensor shape, usually (B, ...)

        Returns
        -------
        out:
            shape (B, 1, ..., 1)
        """
        bsz = t.shape[0]
        out = arr.gather(0, t.long())
        return out.reshape(bsz, *([1] * (len(x_shape) - 1)))


# ============================================================
# Forward diffusion: q(x_t | x_0)
# ============================================================
def q_sample(
    sched: DiffusionSchedule,
    x0: torch.Tensor,
    t: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Sample x_t from q(x_t | x_0).

    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
    """
    if noise is None:
        noise = torch.randn_like(x0)

    sqrt_ab = sched.extract(sched.sqrt_alphas_cumprod, t, x0.shape)
    sqrt_omb = sched.extract(sched.sqrt_one_minus_alphas_cumprod, t, x0.shape)
    return sqrt_ab * x0 + sqrt_omb * noise


def predict_x0_from_eps(
    sched: DiffusionSchedule,
    x_t: torch.Tensor,
    t: torch.Tensor,
    eps_pred: torch.Tensor,
) -> torch.Tensor:
    """
    Recover x_0 estimate from epsilon prediction.
    """
    sqrt_recip_ab = sched.extract(sched.sqrt_recip_alphas_cumprod, t, x_t.shape)
    sqrt_recipm1_ab = sched.extract(sched.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
    x0_pred = sqrt_recip_ab * x_t - sqrt_recipm1_ab * eps_pred
    return x0_pred


def predict_eps_from_x0(
    sched: DiffusionSchedule,
    x_t: torch.Tensor,
    t: torch.Tensor,
    x0_pred: torch.Tensor,
) -> torch.Tensor:
    """
    Recover epsilon estimate from x_0 prediction.
    """
    sqrt_ab = sched.extract(sched.sqrt_alphas_cumprod, t, x_t.shape)
    sqrt_omb = sched.extract(sched.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
    return (x_t - sqrt_ab * x0_pred) / (sqrt_omb + 1e-8)


# ============================================================
# Training target helpers
# ============================================================
@dataclass
class DiffusionTrainingTarget:
    """
    Container for one training target construction.
    """
    x0: torch.Tensor
    x_t: torch.Tensor
    t: torch.Tensor
    noise: torch.Tensor
    target: torch.Tensor


def make_training_target(
    sched: DiffusionSchedule,
    x0: torch.Tensor,
    predict_type: PredictType = "eps",
    t: Optional[torch.Tensor] = None,
    noise: Optional[torch.Tensor] = None,
) -> DiffusionTrainingTarget:
    """
    Build training tuple for diffusion loss.

    If predict_type == "eps":
        target = noise
    If predict_type == "x0":
        target = x0
    """
    bsz = x0.shape[0]
    device = x0.device

    if t is None:
        t = torch.randint(0, sched.n_steps, (bsz,), device=device, dtype=torch.long)

    if noise is None:
        noise = torch.randn_like(x0)

    x_t = q_sample(sched=sched, x0=x0, t=t, noise=noise)

    if predict_type == "eps":
        target = noise
    elif predict_type == "x0":
        target = x0
    else:
        raise ValueError(f"Unknown predict_type: {predict_type}")

    return DiffusionTrainingTarget(
        x0=x0,
        x_t=x_t,
        t=t,
        noise=noise,
        target=target,
    )


# ============================================================
# Reverse diffusion: p(x_{t-1} | x_t)
# ============================================================
def q_posterior_mean_variance(
    sched: DiffusionSchedule,
    x0_pred: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute posterior q(x_{t-1} | x_t, x_0_pred).

    Returns
    -------
    posterior_mean
    posterior_variance
    posterior_log_variance_clipped
    """
    coef1 = sched.extract(sched.posterior_mean_coef1, t, x_t.shape)
    coef2 = sched.extract(sched.posterior_mean_coef2, t, x_t.shape)

    posterior_mean = coef1 * x0_pred + coef2 * x_t
    posterior_variance = sched.extract(sched.posterior_variance, t, x_t.shape)
    posterior_log_variance_clipped = sched.extract(
        sched.posterior_log_variance_clipped, t, x_t.shape
    )
    return posterior_mean, posterior_variance, posterior_log_variance_clipped


@torch.no_grad()
def p_mean_variance(
    model: nn.Module,
    sched: DiffusionSchedule,
    x_t: torch.Tensor,
    t: torch.Tensor,
    cond,
    predict_type: PredictType = "eps",
    clip_x0: bool = True,
):
    """
    Compute reverse-process mean/variance from model prediction.

    Model contract
    --------------
    model(x_t, t, cond) -> pred

    where pred is:
    - epsilon if predict_type == "eps"
    - x0 if predict_type == "x0"
    """
    pred = model(x_t, t, cond)

    if predict_type == "eps":
        x0_pred = predict_x0_from_eps(sched, x_t, t, pred)
    elif predict_type == "x0":
        x0_pred = pred
    else:
        raise ValueError(f"Unknown predict_type: {predict_type}")

    if clip_x0:
        x0_pred = clip_unit_action(x0_pred)

    model_mean, posterior_variance, posterior_log_variance = q_posterior_mean_variance(
        sched=sched,
        x0_pred=x0_pred,
        x_t=x_t,
        t=t,
    )

    return model_mean, posterior_variance, posterior_log_variance, x0_pred


@torch.no_grad()
def p_sample(
    model: nn.Module,
    sched: DiffusionSchedule,
    x_t: torch.Tensor,
    t: torch.Tensor,
    cond,
    predict_type: PredictType = "eps",
    clip_x0: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample one reverse step:
        x_t -> x_{t-1}

    Returns
    -------
    x_prev:
        sampled x_{t-1}
    x0_pred:
        current x0 estimate
    """
    model_mean, _, model_log_variance, x0_pred = p_mean_variance(
        model=model,
        sched=sched,
        x_t=x_t,
        t=t,
        cond=cond,
        predict_type=predict_type,
        clip_x0=clip_x0,
    )

    noise = torch.randn_like(x_t)

    # no noise when t == 0
    nonzero_mask = (t != 0).float().reshape(x_t.shape[0], *([1] * (x_t.ndim - 1)))
    x_prev = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
    return x_prev, x0_pred


@torch.no_grad()
def p_sample_loop(
    model: nn.Module,
    sched: DiffusionSchedule,
    shape: torch.Size,
    cond,
    device: torch.device,
    predict_type: PredictType = "eps",
    clip_x0: bool = True,
    init_noise: Optional[torch.Tensor] = None,
    return_all: bool = False,
):
    """
    Full reverse sampling loop.

    Parameters
    ----------
    shape:
        Usually (B, chunk_len, act_dim)
    return_all:
        If True, also return the full trajectory [x_T, ..., x_0_est]

    Returns
    -------
    x_final:
        final denoised normalized chunk in [-1,1] after clipping
    traj (optional):
        list of intermediate tensors
    """
    if init_noise is None:
        x = torch.randn(shape, device=device)
    else:
        x = init_noise.to(device)

    traj = [x.clone()] if return_all else None

    for step in reversed(range(sched.n_steps)):
        t = torch.full((shape[0],), step, device=device, dtype=torch.long)
        x, x0_pred = p_sample(
            model=model,
            sched=sched,
            x_t=x,
            t=t,
            cond=cond,
            predict_type=predict_type,
            clip_x0=clip_x0,
        )
        if return_all:
            traj.append(x.clone())

    x = clip_unit_action(x)

    if return_all:
        return x, traj
    return x


# ============================================================
# Optional guidance hook
# ============================================================
@torch.no_grad()
def p_sample_loop_with_guidance(
    model: nn.Module,
    sched: DiffusionSchedule,
    shape: torch.Size,
    cond,
    device: torch.device,
    predict_type: PredictType = "eps",
    clip_x0: bool = True,
    init_noise: Optional[torch.Tensor] = None,
    guide_fn=None,
    guide_scale: float = 0.0,
):
    """
    Reverse sampling loop with a simple x-space guidance hook.

    guide_fn signature
    ------------------
    guide_fn(x_t, t, cond, x0_pred) -> grad_like_tensor

    Notes
    -----
    This is intentionally lightweight for the first version.
    For your first implementation, you may not even use this during sampling;
    Q-guidance can first be applied in actor training loss instead.
    """
    if init_noise is None:
        x = torch.randn(shape, device=device)
    else:
        x = init_noise.to(device)

    for step in reversed(range(sched.n_steps)):
        t = torch.full((shape[0],), step, device=device, dtype=torch.long)

        model_mean, _, model_log_variance, x0_pred = p_mean_variance(
            model=model,
            sched=sched,
            x_t=x,
            t=t,
            cond=cond,
            predict_type=predict_type,
            clip_x0=clip_x0,
        )

        if guide_fn is not None and guide_scale > 0.0:
            g = guide_fn(x, t, cond, x0_pred)
            if g is not None:
                model_mean = model_mean + guide_scale * g

        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().reshape(x.shape[0], *([1] * (x.ndim - 1)))
        x = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    return clip_unit_action(x)