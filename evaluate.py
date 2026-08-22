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
from src.utils.metrics import calc_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Digital Twin Recursive Rollouts on Held-out Cohort")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional model checkpoint path")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_data_dir = config["data"]["raw_data_dir"]
    train_participants = config["data"]["train_participants"]
    test_participants = config["data"]["test_participants"]

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

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        encoder.load_state_dict(ckpt["encoder"])
        twin.load_state_dict(ckpt["twin"])
        print(f"Loaded weights from {args.checkpoint}")

    twin.eval()
    horizons = config["evaluation"].get("rollout_horizons", [1, 5, 10, 30, 60])

    print("=" * 85)
    print("HELD-OUT DIGITAL TWIN MULTI-STEP EVALUATION")
    print("=" * 85)

    for pid in test_participants:
        d = load_participant_modalities(
            pid, raw_data_dir, vocab_meal_type, vocab_meal_tag, vocab_act_type, vocab_intensity
        )
        st_test, _ = generate_patient_states(
            encoder, d, device,
            max_history=config["data"]["max_history"],
            stride=config["evaluation"].get("stride", 1),
            max_points=config["evaluation"].get("max_points", 1500),
        )
        if len(st_test) < max(horizons) + 1:
            print(f"Skipping {pid}: insufficient trajectory length ({len(st_test)})")
            continue

        st_t = torch.tensor(st_test, dtype=torch.float32, device=device)
        curr = st_t[:-1]
        target = st_t[1:]

        with torch.no_grad():
            pred_1step = twin.evolve(curr)
        m1 = calc_metrics(pred_1step, target)

        print(f"\n--- Participant: {pid} ({len(st_test):,} states) ---")
        print("1-Step ML Prediction Performance:")
        print(f"  MSE:  {m1['MSE']:.6e} | RMSE: {m1['RMSE']:.6e} | MAE: {m1['MAE']:.6e} | R2: {m1['R2']:.4f}")

        print("\nDigital Twin Multi-Step Rollout Performance:")
        for H in horizons:
            starts = st_t[:-H]
            targets_H = st_t[H:]
            with torch.no_grad():
                rollouts = twin.rollout(starts, steps=H)
                pred_H = rollouts[:, -1, :]
            mH = calc_metrics(pred_H, targets_H)
            print(f"  Horizon H={H:02d} | MSE: {mH['MSE']:.6e} | RMSE: {mH['RMSE']:.6e} | MAE: {mH['MAE']:.6e} | R2: {mH['R2']:.4f}")


if __name__ == "__main__":
    main()
