import argparse
import numpy as np
import torch
import yaml

from src.data.dataset import (
    build_vocabularies,
    load_participant_modalities,
    generate_patient_states,
)
from src.models.modality_encoder import StateEncoder
from src.models.digital_twin import DigitalTwin
from src.utils.visualization import plot_counterfactual_simulation


def main():
    parser = argparse.ArgumentParser(description="Digital Twin Counterfactual What-If Simulation")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--participant", type=str, default="UoM2401", help="Participant ID for simulation")
    parser.add_argument("--horizon", type=int, default=60, help="Simulation steps forward")
    parser.add_argument("--perturbation", type=float, default=0.10, help="Scenario perturbation magnitude (e.g. 0.10 for 10%)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_data_dir = config["data"]["raw_data_dir"]
    train_participants = config["data"]["train_participants"]

    vocab_meal_type, vocab_meal_tag, vocab_act_type, vocab_intensity = build_vocabularies(
        raw_data_dir, train_participants
    )

    hidden_dim = config["model"]["hidden_dim"]
    state_dim = config["model"]["state_dim"]

    encoder = StateEncoder(
        hidden_dim=hidden_dim,
        state_dim=state_dim,
        vocab_meal_type_len=len(vocab_meal_type),
        vocab_meal_tag_len=len(vocab_meal_tag),
        vocab_act_type_len=len(vocab_act_type),
        vocab_intensity_len=len(vocab_intensity),
    ).to(device)

    twin = DigitalTwin(state_dim=state_dim, hidden_dim=hidden_dim).to(device)
    twin.eval()

    d = load_participant_modalities(
        args.participant, raw_data_dir, vocab_meal_type, vocab_meal_tag, vocab_act_type, vocab_intensity
    )
    st_test, _ = generate_patient_states(
        encoder, d, device,
        max_history=config["data"]["max_history"],
        stride=1,
        max_points=1500,
    )

    start_idx = len(st_test) // 2
    initial_state = torch.tensor(st_test[start_idx : start_idx + 1], dtype=torch.float32, device=device)

    # 1. Baseline Twin Initialization
    twin_baseline = twin.initialize(initial_state)

    # 2. Counterfactual Scenario Perturbation (S~_t = S_t + delta)
    delta = args.perturbation * initial_state
    twin_scenario = twin.scenario(twin_baseline, delta)

    # 3. Simulate Rollouts forward
    with torch.no_grad():
        baseline_sim = twin.rollout(twin_baseline, steps=args.horizon)[0].cpu().numpy()
        scenario_sim = twin.rollout(twin_scenario, steps=args.horizon)[0].cpu().numpy()

    baseline_norms = np.linalg.norm(baseline_sim, axis=-1)
    scenario_norms = np.linalg.norm(scenario_sim, axis=-1)
    divergence = np.linalg.norm(scenario_sim[-1] - baseline_sim[-1])

    print("=" * 85)
    print(f"DIGITAL TWIN WHAT-IF SIMULATION: {args.participant}")
    print("=" * 85)
    print(f"  Simulation Horizon           : {args.horizon} steps")
    print(f"  Perturbation Factor          : {args.perturbation * 100:.1f}%")
    print(f"  Initial State L2 Norm        : {float(torch.norm(initial_state).item()):.4f}")
    print(f"  Final Baseline State L2 Norm : {float(np.linalg.norm(baseline_sim[-1])):.4f}")
    print(f"  Final Scenario State L2 Norm : {float(np.linalg.norm(scenario_sim[-1])):.4f}")
    print(f"  Trajectory Divergence Gap    : {float(divergence):.4f}")
    print("=" * 85)


if __name__ == "__main__":
    main()
