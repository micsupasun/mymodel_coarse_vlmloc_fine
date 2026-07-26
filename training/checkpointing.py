import os
import os.path as osp
from typing import Any, Dict, Optional

import torch


def _atomic_save(payload: Dict[str, Any], path: str) -> None:
    os.makedirs(osp.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def build_training_state(
    *,
    model_state: Dict[str, Any],
    epoch: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    best_metric: Optional[float] = None,
    best_model_path: str = "",
    extra_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    state = {
        "checkpoint_type": "training_state",
        "epoch": epoch,
        "model_state": model_state,
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_metric": best_metric,
        "best_model_path": best_model_path,
        "extra_state": extra_state or {},
    }
    return state


def save_training_state(path: str, payload: Dict[str, Any]) -> None:
    _atomic_save(payload, path)


def load_training_state(path: str, map_location: Any) -> Dict[str, Any]:
    payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(f"{path} is not a resumable training-state checkpoint.")
    return payload

