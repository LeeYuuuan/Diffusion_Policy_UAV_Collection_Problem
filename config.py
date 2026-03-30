from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional


# =========================
# Seeds / Reproducibility
# =========================
@dataclass
class SeedCfg:
    """
    map_seed:
        Controls ONLY user distribution (fixed map).
        Change it only when you want a different user map.

    exp_seed:
        Controls experiment randomness (env dynamics + RL training).
        Same exp_seed => fully reproducible run, while episodes can still differ
        naturally as long as RNG is not reset every episode.
    """
    map_seed: int = 42
    exp_seed: int = 43

    # RNG stream offsets derived from exp_seed
    env_dyn_offset: int = 0       # Poisson arrivals
    agent_offset: int = 1         # network init / action sampling / training randomness
    eval_offset: int = 2          # evaluation randomness


def derive_seed(base: int, offset: int) -> int:
    """
    Deterministically derive different seeds from the same base seed.

    This is only for organizing reproducible experiment streams.
    """
    return int(base) + int(offset) * 1_000_003


# =========================
# Environment config
# =========================
@dataclass
class EnvCfg:
    # World
    world_size: float = 400.0

    # Time model
    step_move_sec: float = 10.0
    step_collect_sec: float = 50.0

    # UAV / coverage
    n_uavs: int = 3
    cov_radius: float = 86.6

    # Traffic
    lam: float = 0.05

    # Optional cold start at episode reset
    cold_start_dt: float = 60.0

    # Penalties
    oob_penalty: float = 5.0
    oob_penalty_scale: float = 5.0

    # Reward
    team_max_sum_weight: float = 0.3
    max_before_weight: float = 1.0
    ratio_reward: bool = False
    ratio_weight: float = 10.0

    def validate(self) -> None:
        assert self.world_size > 0
        assert self.step_move_sec >= 0
        assert self.step_collect_sec >= 0
        assert self.n_uavs > 0
        assert self.cov_radius > 0
        assert self.lam >= 0
        assert self.cold_start_dt >= 0
        assert self.oob_penalty >= 0
        assert self.oob_penalty_scale >= 0
        assert self.team_max_sum_weight >= 0
        assert self.max_before_weight >= 0
        if self.ratio_reward:
            assert self.ratio_weight >= 0


# =========================
# Fixed user map generation
# =========================
@dataclass
class UserGenCfg:
    """
    Controls ONLY the user/sensor spatial distribution.

    This is used once to build a fixed map.
    Users' positions do not change across episodes unless the map is rebuilt.
    """
    n_users: int = 50

    # Mixture: clustered + uniform
    cluster_ratio: float = 0.7
    n_clusters: int = 4
    cluster_std: float = 20.0

    def validate(self) -> None:
        assert self.n_users >= 0
        assert 0.0 <= self.cluster_ratio <= 1.0
        assert self.n_clusters > 0
        assert self.cluster_std >= 0


# =========================
# Observation config
# =========================
ObsMode = Literal["lv+uav", "lv+uav+topk"]


@dataclass
class ObsCfg:
    """
    Current supported flat observation modes:

    1) "lv+uav":
        [ last_visit(all users), uav_pos(flat) ]

    2) "lv+uav+topk":
        [ last_visit(all users), uav_pos(flat), topK_oldest_user_pos(flat) ]

    Notes:
    - The environment still outputs a flat vector.
    - Higher-level model-side parsing should be handled by an adapter layer.
    """
    mode: ObsMode = "lv+uav+topk"
    topk_oldest: int = 1

    def validate(self) -> None:
        if self.mode == "lv+uav":
            assert self.topk_oldest == 0
        elif self.mode == "lv+uav+topk":
            assert self.topk_oldest >= 1
        else:
            raise ValueError(f"Unknown obs mode: {self.mode}")


# =========================
# Action / chunk execution config
# =========================
ExecuteMode = Literal["first_k"]


@dataclass
class ChunkCfg:
    """
    Diffusion policy outputs an action chunk:
        [a_t, a_{t+1}, ..., a_{t+H-1}]

    But the runner does not have to execute all generated actions.

    chunk_len:
        Number of actions generated in one shot.

    exec_len:
        Number of actions actually executed before replanning.

    Example:
        chunk_len = 10, exec_len = 2
        -> generate 10 actions, execute only first 2, then observe again.

    execute_mode:
        "first_k" means execute the first exec_len actions.
    """
    chunk_len: int = 10
    exec_len: int = 2
    execute_mode: ExecuteMode = "first_k"

    # Optional future extension:
    # if True, the runner may cache the unused tail for debugging or reuse logic
    keep_unused_tail: bool = False

    def validate(self) -> None:
        assert self.chunk_len >= 1
        assert self.exec_len >= 1
        assert self.exec_len <= self.chunk_len
        if self.execute_mode not in ("first_k",):
            raise ValueError(f"Unknown execute_mode: {self.execute_mode}")


# =========================
# Diffusion model config
# =========================
PredictType = Literal["eps", "x0"]
BetaScheduleType = Literal["linear", "cosine"]


@dataclass
class DiffusionCfg:
    """
    Configuration for the diffusion actor.

    denoise_steps:
        Number of reverse denoising steps used during action sampling.

    train_diffusion_steps:
        Number of diffusion timesteps used in training schedule.

    predict_type:
        "eps" -> predict noise
        "x0"  -> predict clean action chunk directly

    action_low / action_high:
        Environment action range for each action dimension.
        Keep consistent with env semantics.
    """
    denoise_steps: int = 8
    train_diffusion_steps: int = 32
    beta_schedule: BetaScheduleType = "cosine"
    predict_type: PredictType = "eps"

    action_low: float = -50.0
    action_high: float = 50.0

    hidden_dim: int = 256
    cond_dim: int = 256
    time_embed_dim: int = 128
    n_blocks: int = 4
    n_heads: int = 4
    dropout_p: float = 0.0

    # Actor update
    q_guidance_weight: float = 0.1
    bc_weight: float = 1.0

    # Optional stabilization
    clip_grad_norm: float = 10.0

    def validate(self) -> None:
        assert self.denoise_steps >= 1
        assert self.train_diffusion_steps >= 2
        assert self.beta_schedule in ("linear", "cosine")
        assert self.predict_type in ("eps", "x0")
        assert self.action_high > self.action_low
        assert self.hidden_dim > 0
        assert self.cond_dim > 0
        assert self.time_embed_dim > 0
        assert self.n_blocks >= 1
        assert self.n_heads >= 1
        assert self.dropout_p >= 0.0
        assert self.q_guidance_weight >= 0.0
        assert self.bc_weight >= 0.0
        assert self.clip_grad_norm > 0.0


# =========================
# Critic config
# =========================
CriticMode = Literal["state_action"]


@dataclass
class CriticCfg:
    """
    Critic configuration.

    state_horizon:
        Number of states concatenated (or otherwise adapted) as critic input.

    First version recommendation:
        state_horizon = 1

    Later:
        Increase to K > 1 if you want the critic to see recent state history.
    """
    mode: CriticMode = "state_action"

    state_horizon: int = 1
    hidden_dim: int = 256
    dropout_p: float = 0.0

    twin_q: bool = True
    target_tau: float = 0.005
    gamma: float = 0.99

    clip_grad_norm: float = 10.0

    def validate(self) -> None:
        assert self.mode in ("state_action",)
        assert self.state_horizon >= 1
        assert self.hidden_dim > 0
        assert self.dropout_p >= 0.0
        assert self.target_tau > 0.0
        assert 0.0 < self.gamma <= 1.0
        assert self.clip_grad_norm > 0.0


# =========================
# Observation adapter / condition builder config
# =========================
AdapterType = Literal["flat", "structured"]


@dataclass
class AdapterCfg:
    """
    Controls how the flat env observation is interpreted by the model side.

    adapter_type:
        "flat":
            Keep observation as a flat vector. Simplest baseline.

        "structured":
            Parse observation into structured pieces such as:
            - sensor-related tokens
            - UAV tokens
            - optional top-k user position tokens
            - global summary token

    actor_condition_horizon:
        Number of recent observations used to build actor condition.

    critic_condition_horizon:
        Number of recent observations used to build critic input.

    These horizons are adapter-level concepts and do not force env changes.
    """
    adapter_type: AdapterType = "structured"

    actor_condition_horizon: int = 1
    critic_condition_horizon: int = 1

    # Structured-token options
    include_uav_tokens: bool = True
    include_sensor_tokens: bool = True
    include_topk_tokens: bool = True
    include_global_token: bool = True

    # Future-facing option:
    # if later you add UAV trajectory history into obs, this can be increased
    uav_history_len: int = 1

    def validate(self) -> None:
        assert self.adapter_type in ("flat", "structured")
        assert self.actor_condition_horizon >= 1
        assert self.critic_condition_horizon >= 1
        assert self.uav_history_len >= 1


# =========================
# Replay / sampling config
# =========================
@dataclass
class ReplayCfg:
    """
    Replay buffer config for diffusion training.

    The plan is to store step transitions, then sample:
    - step batches for critic
    - chunk sequences for actor
    """
    capacity: int = 100_000
    batch_size_critic: int = 128
    batch_size_actor: int = 128

    # Number of consecutive actions sampled as one actor target chunk
    actor_chunk_len: int = 10

    # Whether chunk sampling is allowed to cross episode boundaries
    allow_cross_episode_chunk: bool = False

    def validate(self) -> None:
        assert self.capacity > 0
        assert self.batch_size_critic > 0
        assert self.batch_size_actor > 0
        assert self.actor_chunk_len >= 1


# =========================
# Training config for diffusion agent
# =========================
@dataclass
class TrainCfg:
    """
    High-level training loop config for diffusion policy training.
    """
    num_episodes: int = 200
    num_timestep: int = 100

    warmup_steps: int = 1000
    updates_per_env_step: int = 1

    lr_actor: float = 3e-4
    lr_critic: float = 3e-4

    eval_every: int = 5
    return_every: int = 10

    # Logging / debug
    print_every: int = 1

    def validate(self) -> None:
        assert self.num_episodes >= 1
        assert self.num_timestep >= 1
        assert self.warmup_steps >= 0
        assert self.updates_per_env_step >= 1
        assert self.lr_actor > 0
        assert self.lr_critic > 0
        assert self.eval_every >= 0
        assert self.return_every >= 1
        assert self.print_every >= 1


# =========================
# Unified experiment config
# =========================
@dataclass
class ExpCfg:
    seeds: SeedCfg = field(default_factory=SeedCfg)
    env: EnvCfg = field(default_factory=EnvCfg)
    user_gen: UserGenCfg = field(default_factory=UserGenCfg)
    obs: ObsCfg = field(default_factory=ObsCfg)

    # New blocks for diffusion version
    chunk: ChunkCfg = field(default_factory=ChunkCfg)
    diffusion: DiffusionCfg = field(default_factory=DiffusionCfg)
    critic: CriticCfg = field(default_factory=CriticCfg)
    adapter: AdapterCfg = field(default_factory=AdapterCfg)
    replay: ReplayCfg = field(default_factory=ReplayCfg)
    train: TrainCfg = field(default_factory=TrainCfg)

    def validate(self) -> None:
        self.seeds  # kept for completeness
        self.env.validate()
        self.user_gen.validate()
        self.obs.validate()

        self.chunk.validate()
        self.diffusion.validate()
        self.critic.validate()
        self.adapter.validate()
        self.replay.validate()
        self.train.validate()

        # Cross-config consistency checks
        assert self.replay.actor_chunk_len == self.chunk.chunk_len, (
            "replay.actor_chunk_len should match chunk.chunk_len "
            "for the first diffusion version."
        )
        assert self.critic.state_horizon >= 1
        assert self.adapter.critic_condition_horizon >= 1
        assert self.adapter.actor_condition_horizon >= 1


# Default global config
CFG = ExpCfg()