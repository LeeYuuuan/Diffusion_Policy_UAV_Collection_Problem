from __future__ import annotations

from pathlib import Path
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any
import json
import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Basic serialization helpers
# ============================================================
def _to_serializable(obj: Any):
    """
    Recursively convert dataclass / Path / tuple objects into JSON-serializable format.
    """
    if is_dataclass(obj):
        return {k: _to_serializable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


# ============================================================
# Experiment directory / metadata
# ============================================================
def create_experiment_dir(
    root: str = "logs",
    exp_name: str = "default_exp",
    exp_info: str = "",
    allow_overwrite: bool = False,
) -> Path:
    """
    Create one experiment folder under root.

    Example:
        logs/uav3_1_1/
    """
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    exp_dir = root_path / exp_name

    if exp_dir.exists() and not allow_overwrite:
        raise FileExistsError(
            f"Experiment folder already exists: {exp_dir}. "
            f"Please change exp_name or set allow_overwrite=True."
        )

    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment description
    info_path = exp_dir / "exp_info.txt"
    with open(info_path, "w", encoding="utf-8") as f:
        f.write(exp_info.strip() + "\n")

    # Save simple metadata
    meta = {
        "exp_name": exp_name,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cwd": os.getcwd(),
    }
    with open(exp_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return exp_dir


def save_config_json(cfg: Any, save_path: str | Path) -> None:
    """
    Save config object as JSON.
    """
    save_path = Path(save_path)
    data = _to_serializable(cfg)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_note(exp_dir: str | Path, text: str, filename: str = "notes.txt") -> None:
    """
    Append extra notes into a text file inside the experiment folder.
    """
    exp_dir = Path(exp_dir)
    with open(exp_dir / filename, "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


# ============================================================
# Experiment info builder
# ============================================================
def build_experiment_info(cfg, exp_name: str, user_notes: str = "") -> str:
    """
    Build a readable experiment description string from cfg.

    This keeps the notebook clean:
    - key settings are read automatically from cfg
    - user only needs to write a short note for the experiment purpose
    """
    lines = [
        f"Experiment name: {exp_name}",
        "",
        "Config summary:",
        f"- n_uavs = {cfg.env.n_uavs}",
        f"- world_size = {cfg.env.world_size}",
        f"- cov_radius = {cfg.env.cov_radius}",
        f"- lam = {cfg.env.lam}",
        f"- obs_mode = {cfg.obs.mode}",
        f"- topk_oldest = {cfg.obs.topk_oldest}",
        f"- chunk_len = {cfg.chunk.chunk_len}",
        f"- exec_len = {cfg.chunk.exec_len}",
        f"- predict_type = {cfg.diffusion.predict_type}",
        f"- train_diffusion_steps = {cfg.diffusion.train_diffusion_steps}",
        f"- hidden_dim = {cfg.diffusion.hidden_dim}",
        f"- n_blocks = {cfg.diffusion.n_blocks}",
        f"- n_heads = {cfg.diffusion.n_heads}",
        f"- bc_weight = {cfg.diffusion.bc_weight}",
        f"- q_guidance_weight = {cfg.diffusion.q_guidance_weight}",
        f"- critic_state_horizon = {cfg.critic.state_horizon}",
        f"- gamma = {cfg.critic.gamma}",
        f"- target_tau = {cfg.critic.target_tau}",
        f"- batch_size_actor = {cfg.replay.batch_size_actor}",
        f"- batch_size_critic = {cfg.replay.batch_size_critic}",
        f"- replay_capacity = {cfg.replay.capacity}",
        f"- num_episodes = {cfg.train.num_episodes}",
        f"- num_timestep = {cfg.train.num_timestep}",
        f"- warmup_steps = {cfg.train.warmup_steps}",
        f"- lr_actor = {cfg.train.lr_actor}",
        f"- lr_critic = {cfg.train.lr_critic}",
        f"- eval_every = {cfg.train.eval_every}",
    ]

    if hasattr(cfg.env, "return_to_center_every"):
        lines.append(f"- return_to_center_every = {cfg.env.return_to_center_every}")

    if user_notes.strip():
        lines.extend([
            "",
            "User notes:",
            textwrap.dedent(user_notes).strip(),
        ])

    return "\n".join(lines)


# ============================================================
# Plot helpers
# ============================================================
def _save_single_curve(
    y,
    save_path: str | Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    y = np.asarray(y, dtype=np.float32).reshape(-1)

    plt.figure(figsize=(8, 5))
    plt.plot(y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_reward_and_buffer_plots(agent, cfg, exp_dir: str | Path) -> None:
    """
    Save reward and buffer curves separately.

    Saved files:
    - Episode_Returns.png
    - Episode_Buffer_Metric.png
    """
    exp_dir = Path(exp_dir)

    # ----------------------------
    # Reward figure
    # ----------------------------
    plt.figure(figsize=(8, 5))

    if len(agent.train_rewards) > 0:
        plt.plot(agent.train_rewards, label="Train Reward")

    if len(agent.eval_rewards) > 0:
        eval_x = np.arange(len(agent.eval_rewards)) * int(cfg.train.eval_every)
        plt.plot(eval_x, agent.eval_rewards, marker="o", label="Eval Reward")

    plt.title("Episode Returns")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(exp_dir / "Episode_Returns.png", dpi=200)
    plt.close()

    # ----------------------------
    # Buffer figure
    # ----------------------------
    plt.figure(figsize=(8, 5))

    if len(agent.train_buf_metric) > 0:
        plt.plot(agent.train_buf_metric, label="Train Buffer Metric")

    if len(agent.eval_buf_metric) > 0:
        eval_x = np.arange(len(agent.eval_buf_metric)) * int(cfg.train.eval_every)
        plt.plot(eval_x, agent.eval_buf_metric, marker="o", label="Eval Buffer Metric")

    plt.title("Episode Buffer Metric")
    plt.xlabel("Episode")
    plt.ylabel("Buffer Metric")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(exp_dir / "Episode_Buffer_Metric.png", dpi=200)
    plt.close()


def save_loss_plots(agent, exp_dir: str | Path) -> None:
    """
    Save loss curves separately because their scales can be very different.

    Saved files:
    - Loss_Critic.png
    - Loss_Actor.png
    - Loss_BC.png
    - Loss_Q.png
    - Loss_Overview.png
    """
    exp_dir = Path(exp_dir)

    # Individual figures
    _save_single_curve(
        agent.critic_loss_log,
        exp_dir / "Loss_Critic.png",
        title="Critic Loss",
        xlabel="Logged Update Index",
        ylabel="Critic Loss",
    )

    _save_single_curve(
        agent.actor_loss_log,
        exp_dir / "Loss_Actor.png",
        title="Actor Loss",
        xlabel="Logged Update Index",
        ylabel="Actor Loss",
    )

    _save_single_curve(
        agent.bc_loss_log,
        exp_dir / "Loss_BC.png",
        title="BC Loss",
        xlabel="Logged Update Index",
        ylabel="BC Loss",
    )

    _save_single_curve(
        agent.q_loss_log,
        exp_dir / "Loss_Q.png",
        title="Q Loss",
        xlabel="Logged Update Index",
        ylabel="Q Loss",
    )

    # Overview figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    logs_and_titles = [
        (agent.critic_loss_log, "Critic Loss"),
        (agent.actor_loss_log, "Actor Loss"),
        (agent.bc_loss_log, "BC Loss"),
        (agent.q_loss_log, "Q Loss"),
    ]

    for ax, (series, title) in zip(axes.flatten(), logs_and_titles):
        series = np.asarray(series, dtype=np.float32).reshape(-1)
        if len(series) > 0:
            ax.plot(series)
        ax.set_title(title)
        ax.set_xlabel("Logged Update Index")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(exp_dir / "Loss_Overview.png", dpi=200)
    plt.close()


# ============================================================
# Optional convenience wrapper
# ============================================================
def save_basic_experiment_outputs(agent, cfg, exp_dir: str | Path) -> None:
    """
    Convenience wrapper to save the most common outputs together.
    """
    exp_dir = Path(exp_dir)

    agent.save_training_logs(str(exp_dir / "train_logs.npz"))
    agent.save(str(exp_dir / "model.pt"))

    save_reward_and_buffer_plots(agent, cfg, exp_dir)
    save_loss_plots(agent, exp_dir)