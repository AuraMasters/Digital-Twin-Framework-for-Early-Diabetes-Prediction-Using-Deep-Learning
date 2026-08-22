from typing import Dict, Union
import numpy as np
import torch


def calc_metrics(pred: Union[torch.Tensor, np.ndarray], target: Union[torch.Tensor, np.ndarray]) -> Dict[str, float]:
    """
    Computes regression and trajectory tracking evaluation metrics:
    - Mean Squared Error (MSE)
    - Root Mean Squared Error (RMSE)
    - Mean Absolute Error (MAE)
    - Coefficient of Determination (R²)
    """
    p = pred.detach().cpu().numpy() if isinstance(pred, torch.Tensor) else pred
    t = target.detach().cpu().numpy() if isinstance(target, torch.Tensor) else target

    mse = np.mean((p - t) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(p - t))
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    r2 = 1.0 - np.sum((t - p) ** 2) / ss_tot if ss_tot > 0 else float("nan")

    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
    }
