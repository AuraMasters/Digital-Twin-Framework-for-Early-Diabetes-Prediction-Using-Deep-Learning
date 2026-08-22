from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np


def plot_training_loss(loss_history: List[float], save_path: Optional[str] = None):
    """Plots and optionally saves the training loss trajectory."""
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, marker="o", color="#1f77b4", linewidth=2, label="MSE Loss")
    plt.title("Digital Twin State-Transition Training Loss", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Mean Squared Error (MSE)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_rollout_comparison(
    ground_truth: np.ndarray,
    predicted: np.ndarray,
    participant_id: str,
    horizon: int,
    save_path: Optional[str] = None,
):
    """Plots ground truth vs multi-step digital twin rollout state trajectory."""
    plt.figure(figsize=(12, 6))
    time_steps = np.arange(len(ground_truth))
    plt.plot(time_steps, ground_truth, label="Observed Patient State Norm", color="#2ca02c", linewidth=2)
    plt.plot(time_steps, predicted, label=f"Digital Twin Forecast (H={horizon})", color="#d62728", linestyle="--", linewidth=2)
    plt.title(f"Patient {participant_id}: Digital Twin {horizon}-Step Rollout Trajectory", fontsize=14, fontweight="bold")
    plt.xlabel("Discrete Time Step (t)", fontsize=12)
    plt.ylabel("Patient State L2 Magnitude", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_counterfactual_simulation(
    baseline_traj: np.ndarray,
    scenario_traj: np.ndarray,
    participant_id: str,
    save_path: Optional[str] = None,
):
    """Plots baseline vs counterfactual what-if scenario trajectory divergence."""
    plt.figure(figsize=(12, 6))
    steps = np.arange(len(baseline_traj))
    plt.plot(steps, baseline_traj, label="Baseline Twin Trajectory", color="#1f77b4", linewidth=2.5)
    plt.plot(steps, scenario_traj, label="Counterfactual Scenario (Perturbed)", color="#ff7f0e", linestyle="--", linewidth=2.5)
    plt.fill_between(steps, baseline_traj, scenario_traj, color="#ff7f0e", alpha=0.15, label="Trajectory Divergence Gap")
    plt.title(f"Counterfactual 'What-If' Simulation for {participant_id}", fontsize=14, fontweight="bold")
    plt.xlabel("Rollout Horizon Steps forward", fontsize=12)
    plt.ylabel("Physiological State Embedding L2 Norm", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
