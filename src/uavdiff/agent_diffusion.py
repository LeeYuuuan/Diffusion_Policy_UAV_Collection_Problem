from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from uavdiff.config import ExpCfg, CFG, derive_seed
from uavdiff.env import CollectDataEnv
from uavdiff.condition import ConditionBuilder
from uavdiff.replay_diffusion import ReplayBufferDiffusion, StepBatch, ChunkBatch
from uavdiff.critic_nets import CriticFactory, flatten_state_history, extract_first_action, soft_update
from uavdiff.diffusion_core import env_to_unit, ActionRange, make_training_target
from uavdiff.diffusion_nets import DiffusionPolicy
from uavdiff.runner_diffusion import DiffusionRunner


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class UpdateStats:
    critic_loss: Optional[float]
    actor_loss: Optional[float]
    bc_loss: Optional[float]
    q_loss: Optional[float]


class AgentDiffusion:
    """
    First-version diffusion RL agent.

    Main design
    -----------
    1) Critic:
        step-based twin Q with configurable state-history stacking

    2) Actor:
        conditional diffusion chunk policy

    3) Actor loss:
        actor_loss = bc_weight * diffusion_loss + q_guidance_weight * q_loss
        where:
            q_loss = - mean( min(Q1, Q2)(s_hist, a_first_pred) )

    Notes
    -----
    - This first version does NOT use guided sampling during rollout.
    - Q-guidance is applied during training loss instead.
    - The policy generates a chunk, but critic only evaluates the first action.
    """

    def __init__(
        self,
        env_cfg: Optional[ExpCfg] = None,
        device: Optional[str] = None,
    ):
        self.cfg = env_cfg or CFG
        self.cfg.validate()

        self.device = torch.device(device) if device is not None else get_device()

        agent_seed = derive_seed(self.cfg.seeds.exp_seed, self.cfg.seeds.agent_offset)
        seed_everything(int(agent_seed))

        # --------------------------------------------------------
        # env
        # --------------------------------------------------------
        self.env = CollectDataEnv(self.cfg)
        obs0 = self.env.reset()

        self.obs_dim = int(self.env.obs_dim())
        self.act_dim = int(self.env.act_dim())
        self.n_uavs = int(self.cfg.env.n_uavs)
        self.chunk_len = int(self.cfg.chunk.chunk_len)

        # --------------------------------------------------------
        # adapters / replay / policy / critic / runner
        # --------------------------------------------------------
        self.cond_builder = ConditionBuilder(self.cfg, device=str(self.device))

        self.replay = ReplayBufferDiffusion(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            capacity=int(self.cfg.replay.capacity),
            chunk_len=int(self.cfg.chunk.chunk_len),
            critic_state_horizon=int(self.cfg.critic.state_horizon),
            actor_obs_horizon=int(self.cfg.adapter.actor_condition_horizon),
            device=str(self.device),
            allow_cross_episode_chunk=bool(self.cfg.replay.allow_cross_episode_chunk),
        )

        self.policy = DiffusionPolicy(
            cfg=self.cfg,
            act_dim=self.act_dim,
            chunk_len=self.chunk_len,
            sensor_in_dim=1,
            uav_in_dim=2,
            topk_in_dim=2 if self.cfg.obs.mode == "lv+uav+topk" else None,
            global_in_dim=2,
        ).to(self.device)

        self.critic = CriticFactory.build(
            cfg=self.cfg,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
        ).to(self.device)

        self.critic_target = CriticFactory.build(
            cfg=self.cfg,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.runner = DiffusionRunner(
            env=self.env,
            policy=self.policy,
            condition_builder=self.cond_builder,
            replay=self.replay,
            cfg=self.cfg,
            device=str(self.device),
        )

        # --------------------------------------------------------
        # optimizers
        # --------------------------------------------------------
        self.opt_actor = torch.optim.Adam(
            self.policy.parameters(),
            lr=float(self.cfg.train.lr_actor),
        )
        self.opt_critic = torch.optim.Adam(
            self.critic.parameters(),
            lr=float(self.cfg.train.lr_critic),
        )

        self.action_range = ActionRange(
            low=float(self.cfg.diffusion.action_low),
            high=float(self.cfg.diffusion.action_high),
        )

        # --------------------------------------------------------
        # logs
        # --------------------------------------------------------
        self.train_rewards: List[float] = []
        self.eval_rewards: List[float] = []
        self.train_buf_metric: List[float] = []
        self.eval_buf_metric: List[float] = []

        self.critic_loss_log: List[float] = []
        self.actor_loss_log: List[float] = []
        self.bc_loss_log: List[float] = []
        self.q_loss_log: List[float] = []

    # ============================================================
    # Interaction helpers
    # ============================================================
    def reset_episode(self) -> np.ndarray:
        return self.runner.reset_episode()

    @torch.no_grad()
    def select_chunk(
        self,
        obs: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> np.ndarray:
        """
        Return one env-range action chunk of shape (H, act_dim).
        """
        plan = self.runner.plan_chunk(obs=obs, deterministic=deterministic, return_unit=False)
        return plan.action_chunk_env.copy()

    @torch.no_grad()
    def select_action(
        self,
        obs: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> np.ndarray:
        """
        Return first action of one generated chunk, shape (act_dim,).
        """
        chunk = self.select_chunk(obs=obs, deterministic=deterministic)
        return chunk[0].copy()

    def _random_action_flat(self) -> np.ndarray:
        low = float(self.cfg.diffusion.action_low)
        high = float(self.cfg.diffusion.action_high)
        return np.random.uniform(low, high, size=(self.act_dim,)).astype(np.float32)

    def _flat_to_env_action(self, act_flat: np.ndarray) -> np.ndarray:
        act_flat = np.asarray(act_flat, dtype=np.float32).reshape(-1)
        assert act_flat.shape[0] == self.act_dim
        return act_flat.reshape(self.n_uavs, 2).astype(np.float32)

    # ============================================================
    # Warmup
    # ============================================================
    def warmup(self, warmup_steps: Optional[int] = None) -> None:
        n = int(self.cfg.train.warmup_steps if warmup_steps is None else warmup_steps)
        if n <= 0:
            return

        self.reset_episode()

        for _ in range(n):
            act_flat = self._random_action_flat()
            _ = self.runner.execute_one_action(
                act_flat=act_flat,
                store_replay=True,
            )

    # ============================================================
    # Critic update
    # ============================================================
    def update_critic(self, batch: StepBatch) -> torch.Tensor:
        """
        Step-wise TD update.

        target action:
            sample next chunk from diffusion policy using next-state history,
            then use its first action.
        """
        s_hist = flatten_state_history(batch.obs_hist)              # (B, K*D)
        s2_hist = flatten_state_history(batch.next_obs_hist)        # (B, K*D)

        q1, q2 = self.critic(s_hist, batch.act)

        with torch.no_grad():
            cond_next = self.cond_builder.build_condition(
                obs_batch=batch.next_obs,
                critic_obs_history=batch.next_obs_hist,
            )
            a2_chunk_env = self.policy.sample_chunk_env(
                cond=cond_next,
                deterministic=False,
                return_unit=False,
            )                                                       # (B,H,A)
            a2_first = extract_first_action(a2_chunk_env)           # (B,A)

            q1_t, q2_t = self.critic_target(s2_hist, a2_first)
            q_t = torch.min(q1_t, q2_t)
            target_q = batch.rew + float(self.cfg.critic.gamma) * (1.0 - batch.done) * q_t

        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.opt_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), float(self.cfg.critic.clip_grad_norm))
        self.opt_critic.step()

        return critic_loss.detach()

    # ============================================================
    # Actor update
    # ============================================================
    def update_actor(self, batch: ChunkBatch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Actor loss:
            bc_weight * diffusion_denoise_loss + q_guidance_weight * q_loss

        where:
            q_loss = -mean Q(s_hist, a_first_pred_env)
        """
        # ---------------------------------------------
        # build condition
        # ---------------------------------------------
        cond = self.cond_builder.build_condition(
            obs_batch=batch.obs0,
            critic_obs_history=batch.obs_hist,
        )

        # ---------------------------------------------
        # diffusion reconstruction target
        # ---------------------------------------------
        x0_env = batch.act_chunk                               # (B,H,A), env range
        x0_unit = env_to_unit(x0_env, self.action_range)       # normalize to [-1,1]

        train_target = make_training_target(
            sched=self.policy.schedule,
            x0=x0_unit,
            predict_type=self.cfg.diffusion.predict_type,
        )

        pred = self.policy(
            x_t=train_target.x_t,
            t=train_target.t,
            cond=cond,
        )

        bc_loss = F.mse_loss(pred, train_target.target)

        # ---------------------------------------------
        # Q-guidance term
        # ---------------------------------------------
        # derive predicted x0 in normalized space
        if self.cfg.diffusion.predict_type == "eps":
            from src.uavdiff.diffusion_core import predict_x0_from_eps, unit_to_env
            x0_pred_unit = predict_x0_from_eps(
                sched=self.policy.schedule,
                x_t=train_target.x_t,
                t=train_target.t,
                eps_pred=pred,
            )
        elif self.cfg.diffusion.predict_type == "x0":
            from src.uavdiff.diffusion_core import unit_to_env
            x0_pred_unit = pred
        else:
            raise ValueError(f"Unknown predict_type: {self.cfg.diffusion.predict_type}")

        x0_pred_unit = torch.clamp(x0_pred_unit, -1.0, 1.0)
        x0_pred_env = unit_to_env(x0_pred_unit, self.action_range)
        a_first_pred = extract_first_action(x0_pred_env)        # (B,A)

        s_hist = flatten_state_history(batch.obs_hist)
        q1_pi, q2_pi = self.critic(s_hist, a_first_pred)
        q_pi = torch.min(q1_pi, q2_pi)

        q_loss = -q_pi.mean()

        actor_loss = (
            float(self.cfg.diffusion.bc_weight) * bc_loss
            + float(self.cfg.diffusion.q_guidance_weight) * q_loss
        )

        self.opt_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), float(self.cfg.diffusion.clip_grad_norm))
        self.opt_actor.step()

        return actor_loss.detach(), bc_loss.detach(), q_loss.detach()

    # ============================================================
    # Full update
    # ============================================================
    def update(self) -> UpdateStats:
        if len(self.replay) < max(
            int(self.cfg.replay.batch_size_critic),
            int(self.cfg.replay.batch_size_actor),
            int(self.cfg.chunk.chunk_len),
        ):
            return UpdateStats(None, None, None, None)

        step_batch = self.replay.sample_step_batch(int(self.cfg.replay.batch_size_critic))
        critic_loss = self.update_critic(step_batch)

        chunk_batch = self.replay.sample_chunk_batch(
            batch_size=int(self.cfg.replay.batch_size_actor),
            chunk_len=int(self.cfg.chunk.chunk_len),
        )
        actor_loss, bc_loss, q_loss = self.update_actor(chunk_batch)

        soft_update(
            src=self.critic,
            dst=self.critic_target,
            tau=float(self.cfg.critic.target_tau),
        )

        return UpdateStats(
            critic_loss=float(critic_loss.item()),
            actor_loss=float(actor_loss.item()),
            bc_loss=float(bc_loss.item()),
            q_loss=float(q_loss.item()),
        )

    # ============================================================
    # Training
    # ============================================================
    def train(
        self,
        num_episodes: Optional[int] = None,
        num_timestep: Optional[int] = None,
        eval_every: Optional[int] = None,
        render_eval: bool = False,
    ) -> None:
        n_ep = int(self.cfg.train.num_episodes if num_episodes is None else num_episodes)
        n_ts = int(self.cfg.train.num_timestep if num_timestep is None else num_timestep)
        ev_every = int(self.cfg.train.eval_every if eval_every is None else eval_every)

        self.warmup()

        for ep in range(n_ep):
            obs = self.reset_episode()
            ep_ret = 0.0
            ep_buf_blocks: List[float] = []
            last_stats: Optional[UpdateStats] = None

            executed = 0
            while executed < n_ts:
                result = self.runner.rollout_one_chunk(
                    deterministic=False,
                    store_replay=True,
                    return_unit=False,
                )

                ep_ret += float(result.total_reward)
                executed += len(result.steps)

                for rec in result.steps:
                    ep_buf_blocks.append(float(rec.info.get("max_before", 0.0)))

                for _ in range(int(self.cfg.train.updates_per_env_step) * max(1, len(result.steps))):
                    last_stats = self.update()

            self.train_rewards.append(float(ep_ret))
            self.train_buf_metric.append(
                float(np.mean(ep_buf_blocks)) if len(ep_buf_blocks) > 0 else 0.0
            )

            if last_stats is not None and last_stats.critic_loss is not None:
                self.critic_loss_log.append(last_stats.critic_loss)
                self.actor_loss_log.append(last_stats.actor_loss)
                self.bc_loss_log.append(last_stats.bc_loss)
                self.q_loss_log.append(last_stats.q_loss)

            if ev_every > 0 and (ep % ev_every == 0):
                eval_ret, eval_buf = self.evaluate(
                    num_timestep=n_ts,
                    deterministic=True,
                    render=render_eval,
                )
                self.eval_rewards.append(eval_ret)
                self.eval_buf_metric.append(eval_buf)

                if last_stats is not None and last_stats.critic_loss is not None:
                    print(
                        f"[EP {ep:04d}] "
                        f"TrainR={ep_ret:.2f}  EvalR={eval_ret:.2f}  "
                        f"TrainBuf={self.train_buf_metric[-1]:.2f}  EvalBuf={eval_buf:.2f}  "
                        f"Critic={last_stats.critic_loss:.4f}  "
                        f"Actor={last_stats.actor_loss:.4f}  "
                        f"BC={last_stats.bc_loss:.4f}  "
                        f"Q={last_stats.q_loss:.4f}"
                    )
                else:
                    print(
                        f"[EP {ep:04d}] "
                        f"TrainR={ep_ret:.2f}  EvalR={eval_ret:.2f}  "
                        f"TrainBuf={self.train_buf_metric[-1]:.2f}  EvalBuf={eval_buf:.2f}"
                    )

    # ============================================================
    # Evaluation
    # ============================================================
    @torch.no_grad()
    def evaluate(
        self,
        num_timestep: int = 100,
        deterministic: bool = True,
        render: bool = False,
    ) -> Tuple[float, float]:
        obs = self.reset_episode()
        ret = 0.0
        buf_blocks: List[float] = []

        executed = 0
        while executed < int(num_timestep):
            result = self.runner.rollout_one_chunk(
                deterministic=deterministic,
                store_replay=False,
                return_unit=False,
            )
            ret += float(result.total_reward)
            executed += len(result.steps)

            for rec in result.steps:
                buf_blocks.append(float(rec.info.get("max_before", 0.0)))

            if len(result.steps) == 0:
                break
            if result.steps[-1].done:
                break

        avg_buf = float(np.mean(buf_blocks)) if len(buf_blocks) > 0 else 0.0

        if render:
            self.env.render(show=True)

        return float(ret), float(avg_buf)

    @torch.no_grad()
    def inference(
        self,
        horizon: int = 100,
        render: bool = True,
    ) -> Tuple[float, float]:
        ret, avg_buf = self.evaluate(
            num_timestep=horizon,
            deterministic=True,
            render=render,
        )
        print(f"Inference Return: {ret:.2f}, AvgMaxBuffer={avg_buf:.2f}")
        return ret, avg_buf

    # ============================================================
    # Save / load
    # ============================================================
    def save_training_logs(self, path: str) -> None:
        np.savez(
            path,
            train_rewards=np.asarray(self.train_rewards, dtype=np.float32),
            train_buf_metric=np.asarray(self.train_buf_metric, dtype=np.float32),
            eval_rewards=np.asarray(self.eval_rewards, dtype=np.float32),
            eval_buf_metric=np.asarray(self.eval_buf_metric, dtype=np.float32),
            critic_loss=np.asarray(self.critic_loss_log, dtype=np.float32),
            actor_loss=np.asarray(self.actor_loss_log, dtype=np.float32),
            bc_loss=np.asarray(self.bc_loss_log, dtype=np.float32),
            q_loss=np.asarray(self.q_loss_log, dtype=np.float32),
        )

    def save(self, path: str) -> None:
        payload = {
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "cfg": self.cfg,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
        }
        torch.save(payload, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(payload["policy"])
        self.critic.load_state_dict(payload["critic"])
        self.critic_target.load_state_dict(payload["critic_target"])