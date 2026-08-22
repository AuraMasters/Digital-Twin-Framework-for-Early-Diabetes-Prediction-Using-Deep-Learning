from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


class ModalityGRU(nn.Module):
    """Recurrent Gated Unit (GRU) encoder for a single physiological or lifestyle modality."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        out, hidden = self.gru(x)
        if lengths is None:
            return hidden[-1]
        batch_idx = torch.arange(x.shape[0], device=x.device)
        return out[batch_idx, lengths - 1]


class FiveGRU(nn.Module):
    """
    Five-branch parallel GRU network processing:
    1. Glucose (dim 1)
    2. Insulin (dim 2)
    3. Nutrition (dim 4 numeric + 2 categorical embeddings = 24)
    4. Activity (dim 10 numeric + 2 categorical embeddings = 17)
    5. Sleep (dim 6 numeric)
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        vocab_meal_type_len: int = 20,
        vocab_meal_tag_len: int = 1200,
        vocab_act_type_len: int = 15,
        vocab_intensity_len: int = 10,
    ):
        super().__init__()
        self.glucose_gru = ModalityGRU(1, hidden_dim)
        self.insulin_gru = ModalityGRU(2, hidden_dim)
        self.nutrition_gru = ModalityGRU(24, hidden_dim)
        self.activity_gru = ModalityGRU(17, hidden_dim)
        self.sleep_gru = ModalityGRU(6, hidden_dim)

        # Categorical embedding layers
        self.nut_meal_type_emb = nn.Embedding(vocab_meal_type_len + 5, 10)
        self.nut_meal_tag_emb = nn.Embedding(vocab_meal_tag_len + 5, 10)
        self.act_type_emb = nn.Embedding(vocab_act_type_len + 5, 4)
        self.act_intensity_emb = nn.Embedding(vocab_intensity_len + 5, 3)

    def forward(
        self,
        g: torch.Tensor,
        ins: torch.Tensor,
        nut_num: torch.Tensor,
        nut_type: torch.Tensor,
        nut_tag: torch.Tensor,
        act_num: torch.Tensor,
        act_type: torch.Tensor,
        act_int: torch.Tensor,
        slp: torch.Tensor,
        lengths: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Nutrition representation: 4 numeric + 10 + 10 = 24
        nut_emb1 = self.nut_meal_type_emb(nut_type)
        nut_emb2 = self.nut_meal_tag_emb(nut_tag)
        nut = torch.cat([nut_num, nut_emb1, nut_emb2], dim=-1)

        # Activity representation: 10 numeric + 4 + 3 = 17
        act_emb1 = self.act_type_emb(act_type)
        act_emb2 = self.act_intensity_emb(act_int)
        act = torch.cat([act_num, act_emb1, act_emb2], dim=-1)

        l = lengths or {}
        zg = self.glucose_gru(g, l.get("glucose"))
        zi = self.insulin_gru(ins, l.get("insulin"))
        zn = self.nutrition_gru(nut, l.get("nutrition"))
        za = self.activity_gru(act, l.get("activity"))
        zs = self.sleep_gru(slp, l.get("sleep"))
        return zg, zi, zn, za, zs


class MLPFusion(nn.Module):
    """Nonlinear multi-layer perceptron fusion module combining 5 modality embeddings into patient state S_t."""

    def __init__(self, hidden_dim: int = 64, state_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5 * hidden_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim),
        )

    def forward(
        self,
        zg: torch.Tensor,
        zi: torch.Tensor,
        zn: torch.Tensor,
        za: torch.Tensor,
        zs: torch.Tensor,
    ) -> torch.Tensor:
        fused = torch.cat([zg, zi, zn, za, zs], dim=-1)
        return self.net(fused)


class StateEncoder(nn.Module):
    """Unified Multimodal Patient State Encoder (FiveGRU + MLPFusion)."""

    def __init__(
        self,
        hidden_dim: int = 64,
        state_dim: int = 64,
        vocab_meal_type_len: int = 20,
        vocab_meal_tag_len: int = 1200,
        vocab_act_type_len: int = 15,
        vocab_intensity_len: int = 10,
    ):
        super().__init__()
        self.five_gru = FiveGRU(
            hidden_dim=hidden_dim,
            vocab_meal_type_len=vocab_meal_type_len,
            vocab_meal_tag_len=vocab_meal_tag_len,
            vocab_act_type_len=vocab_act_type_len,
            vocab_intensity_len=vocab_intensity_len,
        )
        self.fusion = MLPFusion(hidden_dim=hidden_dim, state_dim=state_dim)

    def forward(
        self,
        g: torch.Tensor,
        ins: torch.Tensor,
        nut_num: torch.Tensor,
        nut_type: torch.Tensor,
        nut_tag: torch.Tensor,
        act_num: torch.Tensor,
        act_type: torch.Tensor,
        act_int: torch.Tensor,
        slp: torch.Tensor,
        lengths: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        zg, zi, zn, za, zs = self.five_gru(
            g, ins, nut_num, nut_type, nut_tag, act_num, act_type, act_int, slp, lengths
        )
        return self.fusion(zg, zi, zn, za, zs)
