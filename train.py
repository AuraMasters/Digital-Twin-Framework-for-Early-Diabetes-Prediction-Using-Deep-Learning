import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from src.data.dataset import (
    build_vocabularies,
    load_participant_modalities,
    generate_patient_states,
    TransitionDataset,
)
from src.models.modality_encoder import StateEncoder
from src.models.digital_twin import DigitalTwin


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train Digital Twin State Dynamics Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--save_path", type=str, default="checkpoints/digital_twin.pt", help="Path to save model weights")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    seed_everything(config["training"].get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using computing device: {device}")

    raw_data_dir = config["data"]["raw_data_dir"]
    train_participants = config["data"]["train_participants"]

    # 1. Build Vocabularies
    vocab_meal_type, vocab_meal_tag, vocab_act_type, vocab_intensity = build_vocabularies(
        raw_data_dir, train_participants
    )
    print(
        f"Vocabularies: meal_type={len(vocab_meal_type)}, meal_tag={len(vocab_meal_tag)}, "
        f"act_type={len(vocab_act_type)}, intensity={len(vocab_intensity)}"
    )

    # 2. Instantiate Multimodal State Encoder
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

    # 3. Ingest Data and Generate State Transitions
    train_states, train_targets = [], []
    print("Extracting Unified Patient State trajectories for training cohort...")
    for pid in train_participants:
        d = load_participant_modalities(
            pid, raw_data_dir, vocab_meal_type, vocab_meal_tag, vocab_act_type, vocab_intensity
        )
        st, _ = generate_patient_states(
            encoder, d, device,
            max_history=config["data"]["max_history"],
            stride=config["data"]["stride"],
            max_points=config["data"]["max_points"],
        )
        if len(st) >= 2:
            train_states.append(st[:-1])
            train_targets.append(st[1:])
            print(f"  {pid}: {len(st)-1} transition pairs generated")

    X_train = np.concatenate(train_states, axis=0)
    Y_train = np.concatenate(train_targets, axis=0)
    print(f"\nTotal Training Transitions: {len(X_train):,} pairs")

    train_dataset = TransitionDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)

    # 4. Instantiate Digital Twin Model
    twin = DigitalTwin(state_dim=state_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(
        twin.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    criterion = nn.MSELoss()

    epochs = config["training"]["epochs"]
    print(f"Training Digital Twin state-transition dynamics for {epochs} epochs...")
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        twin.train()
        total_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = twin.evolve(x_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x_b)

        avg_loss = total_loss / len(train_dataset)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] - MSE Loss: {avg_loss:.6e}")

    print(f"Training completed in {time.time() - t_start:.2f}s")

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    torch.save({"encoder": encoder.state_dict(), "twin": twin.state_dict()}, args.save_path)
    print(f"Model saved successfully to {args.save_path}")


if __name__ == "__main__":
    main()
