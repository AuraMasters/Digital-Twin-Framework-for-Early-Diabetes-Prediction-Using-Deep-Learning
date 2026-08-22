import os
import glob
from bisect import bisect_right
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def parse_ts(series: pd.Series) -> pd.Series:
    """Parses arbitrary timestamp string series to pandas datetime."""
    return pd.to_datetime(series.astype(str).str.strip(), dayfirst=True, errors="coerce")


def make_vocab(tokens: set) -> Dict[str, int]:
    """Constructs categorical integer token vocabulary with PAD, UNK, and MISSING tokens."""
    v = {"<PAD>": 0, "<UNK>": 1, "<MISSING>": 2}
    for t in sorted(tokens):
        if t and t not in v:
            v[t] = len(v)
    return v


def build_vocabularies(raw_data_dir: str, train_participants: List[str]) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int]]:
    """Builds vocabularies for meal types, meal tags, activity types, and intensity levels from training cohort."""
    meal_types, meal_tags, act_types, intensities = set(), set(), set(), set()
    for pid in train_participants:
        num = pid[3:]
        np_path = os.path.join(raw_data_dir, "Nutrition Data", f"UoMNutrition{num}.csv")
        if os.path.exists(np_path):
            df = pd.read_csv(np_path)
            if "meal_type" in df:
                meal_types.update(df["meal_type"].dropna().astype(str).unique())
            if "meal_tag" in df:
                meal_tags.update(df["meal_tag"].dropna().astype(str).unique())
        ap_path = os.path.join(raw_data_dir, "Activity Data", f"UoMActivity{num}.csv")
        if os.path.exists(ap_path):
            df = pd.read_csv(ap_path)
            if "activity_type" in df:
                act_types.update(df["activity_type"].dropna().astype(str).unique())
            if "intensity" in df:
                intensities.update(df["intensity"].dropna().astype(str).unique())

    vocab_meal_type = make_vocab(meal_types)
    vocab_meal_tag = make_vocab(meal_tags)
    vocab_act_type = make_vocab(act_types)
    vocab_intensity = make_vocab(intensities)

    return vocab_meal_type, vocab_meal_tag, vocab_act_type, vocab_intensity


def load_participant_modalities(
    pid: str,
    raw_data_dir: str,
    vocab_meal_type: Dict[str, int],
    vocab_meal_tag: Dict[str, int],
    vocab_act_type: Dict[str, int],
    vocab_intensity: Dict[str, int]
) -> Dict[str, Any]:
    """
    Ingests and parses all 5 physiological and lifestyle modalities for a given participant.
    Modalities: Glucose, Insulin (Basal & Bolus), Nutrition, Activity, and Sleep.
    """
    num = pid[3:]

    # 1. Glucose (dim 1)
    g_path = os.path.join(raw_data_dir, "Glucose Data", f"UoMGlucose{num}.csv")
    g_df = pd.read_csv(g_path)
    g_df["ts"] = parse_ts(g_df["bg_ts"])
    g_df = g_df.dropna(subset=["ts", "value"]).sort_values("ts")
    g_ts = g_df["ts"].to_numpy()
    g_vals = g_df[["value"]].to_numpy(dtype=np.float32)

    # 2. Insulin (dim 2: dose, event_type: 0 for Basal, 1 for Bolus)
    ins_frames = []
    b_path = os.path.join(raw_data_dir, "Insulin Data", "Basal Data", f"UoMBasal{num}.csv")
    if os.path.exists(b_path):
        b_df = pd.read_csv(b_path)
        b_df["ts"] = parse_ts(b_df["basal_ts"])
        b_df = b_df.dropna(subset=["ts", "basal_dose"])
        b_df["dose"] = pd.to_numeric(b_df["basal_dose"], errors="coerce")
        b_df["event_type"] = 0.0
        ins_frames.append(b_df[["ts", "dose", "event_type"]].dropna())

    bol_path = os.path.join(raw_data_dir, "Insulin Data", "Bolus Data", f"UoMBolus{num}.csv")
    if os.path.exists(bol_path):
        bol_df = pd.read_csv(bol_path)
        bol_df["ts"] = parse_ts(bol_df["bolus_ts"])
        bol_df = bol_df.dropna(subset=["ts", "bolus_dose"])
        bol_df["dose"] = pd.to_numeric(bol_df["bolus_dose"], errors="coerce")
        bol_df["event_type"] = 1.0
        ins_frames.append(bol_df[["ts", "dose", "event_type"]].dropna())

    if ins_frames:
        ins_df = pd.concat(ins_frames, ignore_index=True).sort_values("ts")
        ins_ts = ins_df["ts"].to_numpy()
        ins_vals = ins_df[["dose", "event_type"]].to_numpy(dtype=np.float32)
    else:
        ins_ts = np.array([])
        ins_vals = np.zeros((0, 2), dtype=np.float32)

    # 3. Nutrition (4 numeric + 2 categorical embeddings = 24 features)
    n_path = os.path.join(raw_data_dir, "Nutrition Data", f"UoMNutrition{num}.csv")
    if os.path.exists(n_path):
        n_df = pd.read_csv(n_path)
        n_df["ts"] = parse_ts(n_df["meal_ts"])
        cols = ["carbs_g", "prot_g", "fat_g", "fibre_g"]
        for c in cols:
            n_df[c] = pd.to_numeric(n_df[c], errors="coerce")
        n_df = n_df.dropna(subset=["ts"] + cols).sort_values("ts")
        nut_ts = n_df["ts"].to_numpy()
        nut_num = n_df[cols].to_numpy(dtype=np.float32)
        m_types = np.array([vocab_meal_type.get(str(x), 1) for x in n_df["meal_type"].fillna("")], dtype=np.int64)
        m_tags = np.array([vocab_meal_tag.get(str(x), 1) for x in n_df["meal_tag"].fillna("")], dtype=np.int64)
    else:
        nut_ts = np.array([])
        nut_num = np.zeros((0, 4), dtype=np.float32)
        m_types = np.zeros(0, dtype=np.int64)
        m_tags = np.zeros(0, dtype=np.int64)

    # 4. Activity (10 numeric + 2 categorical embeddings = 17 features)
    a_path = os.path.join(raw_data_dir, "Activity Data", f"UoMActivity{num}.csv")
    if os.path.exists(a_path):
        a_df = pd.read_csv(a_path)
        a_df["ts"] = parse_ts(a_df["activity_ts"])
        cols = [
            "active_Kcal", "step_count", "distance_m", "duration_s", "active_time_s",
            "start_time_s", "start_time_offset_s", "met", "motion_intensity_mean", "motion_intensity_max"
        ]
        for c in cols:
            a_df[c] = pd.to_numeric(a_df[c], errors="coerce")
        a_df = a_df.dropna(subset=["ts"] + cols).sort_values("ts")
        act_ts = a_df["ts"].to_numpy()
        act_num = a_df[cols].to_numpy(dtype=np.float32)
        a_types = np.array([vocab_act_type.get(str(x), 1) for x in a_df["activity_type"].fillna("")], dtype=np.int64)
        a_ints = np.array([vocab_intensity.get(str(x), 1) for x in a_df["intensity"].fillna("")], dtype=np.int64)
    else:
        act_ts = np.array([])
        act_num = np.zeros((0, 10), dtype=np.float32)
        a_types = np.zeros(0, dtype=np.int64)
        a_ints = np.zeros(0, dtype=np.int64)

    # 5. Sleep (6 numeric features)
    s_path = os.path.join(raw_data_dir, "Sleep Data", f"UoMsleep{num}.csv")
    if os.path.exists(s_path):
        s_df = pd.read_csv(s_path)
        s_df["ts"] = parse_ts(s_df["sleep_ts"])
        cols = ["step_count", "heart_rate", "current_activity_type_intensity", "stress_level_value", "sleep_level", "resting_heart_rate"]
        for c in cols:
            s_df[c] = pd.to_numeric(s_df[c], errors="coerce")
        s_df = s_df.dropna(subset=["ts"] + cols).sort_values("ts")
        slp_ts = s_df["ts"].to_numpy()
        slp_num = s_df[cols].to_numpy(dtype=np.float32)
    else:
        slp_ts = np.array([])
        slp_num = np.zeros((0, 6), dtype=np.float32)

    return {
        "glucose": (g_ts, g_vals),
        "insulin": (ins_ts, ins_vals),
        "nutrition": (nut_ts, nut_num, m_types, m_tags),
        "activity": (act_ts, act_num, a_types, a_ints),
        "sleep": (slp_ts, slp_num),
    }


def generate_patient_states(
    model: torch.nn.Module,
    data: Dict[str, Any],
    device: torch.device,
    max_history: int = 32,
    stride: int = 2,
    max_points: int = 1000
) -> Tuple[np.ndarray, List[Any]]:
    """
    Extracts unified causal patient state representations from 5-modality aligned historical windows.
    """
    model.eval()
    g_ts, g_vals = data["glucose"]
    ins_ts, ins_vals = data["insulin"]
    nut_ts, nut_num, nut_type, nut_tag = data["nutrition"]
    act_ts, act_num, act_type, act_int = data["activity"]
    slp_ts, slp_num = data["sleep"]

    valid_indices = []
    for i in range(0, len(g_ts), stride):
        t = g_ts[i]
        if (
            bisect_right(ins_ts, t) > 0
            and bisect_right(nut_ts, t) > 0
            and bisect_right(act_ts, t) > 0
            and bisect_right(slp_ts, t) > 0
        ):
            valid_indices.append(i)
        if len(valid_indices) >= max_points:
            break

    if len(valid_indices) < 2:
        return np.zeros((0, 64), dtype=np.float32), []

    states_list = []
    ts_list = []
    batch_size = 256

    for b_start in range(0, len(valid_indices), batch_size):
        b_idx = valid_indices[b_start : b_start + batch_size]
        B = len(b_idx)

        g_batch = np.zeros((B, max_history, 1), dtype=np.float32)
        ins_batch = np.zeros((B, max_history, 2), dtype=np.float32)
        nut_num_b = np.zeros((B, max_history, 4), dtype=np.float32)
        nut_type_b = np.zeros((B, max_history), dtype=np.int64)
        nut_tag_b = np.zeros((B, max_history), dtype=np.int64)
        act_num_b = np.zeros((B, max_history, 10), dtype=np.float32)
        act_type_b = np.zeros((B, max_history), dtype=np.int64)
        act_int_b = np.zeros((B, max_history), dtype=np.int64)
        slp_batch = np.zeros((B, max_history, 6), dtype=np.float32)

        g_len = np.zeros(B, dtype=np.int64)
        ins_len = np.zeros(B, dtype=np.int64)
        nut_len = np.zeros(B, dtype=np.int64)
        act_len = np.zeros(B, dtype=np.int64)
        slp_len = np.zeros(B, dtype=np.int64)

        for k, idx in enumerate(b_idx):
            t = g_ts[idx]
            ts_list.append(t)

            # Glucose causal window
            end_g = idx + 1
            start_g = max(0, end_g - max_history)
            hist_g = g_vals[start_g:end_g]
            Lg = len(hist_g)
            g_batch[k, max_history - Lg :] = hist_g
            g_len[k] = Lg

            # Insulin causal window
            end_ins = bisect_right(ins_ts, t)
            start_ins = max(0, end_ins - max_history)
            hist_ins = ins_vals[start_ins:end_ins]
            Lins = len(hist_ins)
            ins_batch[k, max_history - Lins :] = hist_ins
            ins_len[k] = Lins

            # Nutrition causal window
            end_n = bisect_right(nut_ts, t)
            start_n = max(0, end_n - max_history)
            hist_n_num = nut_num[start_n:end_n]
            hist_n_type = nut_type[start_n:end_n]
            hist_n_tag = nut_tag[start_n:end_n]
            Ln = len(hist_n_num)
            nut_num_b[k, max_history - Ln :] = hist_n_num
            nut_type_b[k, max_history - Ln :] = hist_n_type
            nut_tag_b[k, max_history - Ln :] = hist_n_tag
            nut_len[k] = Ln

            # Activity causal window
            end_a = bisect_right(act_ts, t)
            start_a = max(0, end_a - max_history)
            hist_a_num = act_num[start_a:end_a]
            hist_a_type = act_type[start_a:end_a]
            hist_a_int = act_int[start_a:end_a]
            La = len(hist_a_num)
            act_num_b[k, max_history - La :] = hist_a_num
            act_type_b[k, max_history - La :] = hist_a_type
            act_int_b[k, max_history - La :] = hist_a_int
            act_len[k] = La

            # Sleep causal window
            end_s = bisect_right(slp_ts, t)
            start_s = max(0, end_s - max_history)
            hist_s = slp_num[start_s:end_s]
            Ls = len(hist_s)
            slp_batch[k, max_history - Ls :] = hist_s
            slp_len[k] = Ls

        with torch.no_grad():
            lengths = {
                "glucose": torch.tensor(g_len, device=device),
                "insulin": torch.tensor(ins_len, device=device),
                "nutrition": torch.tensor(nut_len, device=device),
                "activity": torch.tensor(act_len, device=device),
                "sleep": torch.tensor(slp_len, device=device),
            }
            s = model(
                torch.tensor(g_batch, device=device),
                torch.tensor(ins_batch, device=device),
                torch.tensor(nut_num_b, device=device),
                torch.tensor(nut_type_b, device=device),
                torch.tensor(nut_tag_b, device=device),
                torch.tensor(act_num_b, device=device),
                torch.tensor(act_type_b, device=device),
                torch.tensor(act_int_b, device=device),
                torch.tensor(slp_batch, device=device),
                lengths=lengths,
            )
            states_list.append(s.cpu().numpy())

    return np.concatenate(states_list, axis=0), ts_list


class TransitionDataset(Dataset):
    """PyTorch Dataset for discrete state transitions (S_t -> S_{t+1})."""

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]
