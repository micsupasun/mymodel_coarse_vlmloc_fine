"""Checkpoint loading that refuses silent architecture incompatibilities."""

from __future__ import annotations

import inspect
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


def _load_torch_checkpoint(path: Path) -> Any:
    kwargs: dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = True
    return torch.load(path, **kwargs)


def extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    """Extract a tensor-only state dict without guessing arbitrary nesting."""

    if isinstance(checkpoint, Mapping) and checkpoint and all(
        isinstance(key, str) and torch.is_tensor(value)
        for key, value in checkpoint.items()
    ):
        return checkpoint

    if isinstance(checkpoint, Mapping):
        for container_key in ("state_dict", "model_state_dict"):
            candidate = checkpoint.get(container_key)
            if isinstance(candidate, Mapping) and candidate and all(
                isinstance(key, str) and torch.is_tensor(value)
                for key, value in candidate.items()
            ):
                return candidate

    raise TypeError(
        "Checkpoint is not a tensor-only state dict and has no recognized "
        "'state_dict' or 'model_state_dict' container."
    )


def load_checkpoint_state(path: str | Path) -> Mapping[str, torch.Tensor]:
    """Load and extract a checkpoint state dict using the safest supported mode."""

    return extract_state_dict(_load_torch_checkpoint(Path(path).resolve()))


def _prefix_counts(keys: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(key.split(".", 1)[0] for key in keys).items()))


def _matches_prefix(key: str, prefixes: tuple[str, ...]) -> bool:
    return any(key.startswith(prefix) for prefix in prefixes)


def audit_and_load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    report_path: str | Path,
    *,
    backend: str,
    allowed_missing_prefixes: tuple[str, ...] = (),
    allowed_unexpected_prefixes: tuple[str, ...] = (),
    architecture: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit keys and shapes, write a report, then load only if compatible.

    Released checkpoints may omit frozen language-model weights. A pinned
    public evaluation release may also deliberately ignore a fully enumerated
    set of checkpoint-only modules. Both cases require explicit prefix
    allowlists, and every affected key is still recorded and asserted after
    loading. Shape and dtype mismatches are never allowed.
    """

    checkpoint_path = Path(checkpoint_path).resolve()
    report_path = Path(report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    raw_checkpoint = _load_torch_checkpoint(checkpoint_path)
    checkpoint_state = extract_state_dict(raw_checkpoint)
    model_state = model.state_dict()

    checkpoint_keys = set(checkpoint_state)
    model_keys = set(model_state)
    missing_keys = sorted(model_keys - checkpoint_keys)
    unexpected_keys = sorted(checkpoint_keys - model_keys)
    shape_mismatches = []
    dtype_mismatches = []
    for key in sorted(model_keys & checkpoint_keys):
        expected = model_state[key]
        found = checkpoint_state[key]
        if tuple(expected.shape) != tuple(found.shape):
            shape_mismatches.append(
                {
                    "key": key,
                    "model_shape": list(expected.shape),
                    "checkpoint_shape": list(found.shape),
                }
            )
        elif expected.dtype != found.dtype:
            dtype_mismatches.append(
                {
                    "key": key,
                    "model_dtype": str(expected.dtype),
                    "checkpoint_dtype": str(found.dtype),
                }
            )

    allowed_missing_keys = sorted(
        key for key in missing_keys if _matches_prefix(key, allowed_missing_prefixes)
    )
    forbidden_missing_keys = sorted(set(missing_keys) - set(allowed_missing_keys))
    allowed_unexpected_keys = sorted(
        key
        for key in unexpected_keys
        if _matches_prefix(key, allowed_unexpected_prefixes)
    )
    forbidden_unexpected_keys = sorted(
        set(unexpected_keys) - set(allowed_unexpected_keys)
    )
    compatible = not (
        forbidden_missing_keys
        or forbidden_unexpected_keys
        or shape_mismatches
        or dtype_mismatches
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "backend": backend,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_key_count": len(checkpoint_keys),
        "model_key_count": len(model_keys),
        "checkpoint_prefix_counts": _prefix_counts(checkpoint_keys),
        "model_prefix_counts": _prefix_counts(model_keys),
        "allowed_missing_prefixes": list(allowed_missing_prefixes),
        "allowed_unexpected_prefixes": list(allowed_unexpected_prefixes),
        "missing_keys": missing_keys,
        "allowed_missing_keys": allowed_missing_keys,
        "forbidden_missing_keys": forbidden_missing_keys,
        "unexpected_keys": unexpected_keys,
        "allowed_unexpected_keys": allowed_unexpected_keys,
        "forbidden_unexpected_keys": forbidden_unexpected_keys,
        "shape_mismatches": shape_mismatches,
        "dtype_mismatches": dtype_mismatches,
        "architecture": dict(architecture or {}),
        "compatible": compatible,
        "load_attempted": False,
        "load_succeeded": False,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )

    if not compatible:
        raise RuntimeError(
            f"{backend} checkpoint audit failed; see {report_path}. "
            f"forbidden missing={len(forbidden_missing_keys)}, "
            f"forbidden unexpected={len(forbidden_unexpected_keys)}, "
            f"shape mismatches={len(shape_mismatches)}, "
            f"dtype mismatches={len(dtype_mismatches)}"
        )

    # strict=False is used only after both mismatch sets have been fully
    # enumerated and allowlisted above. The return value is asserted against
    # those exact lists, so it cannot hide a new incompatibility.
    incompatible = model.load_state_dict(checkpoint_state, strict=False)
    loaded_missing = sorted(incompatible.missing_keys)
    loaded_unexpected = sorted(incompatible.unexpected_keys)
    if (
        loaded_missing != allowed_missing_keys
        or loaded_unexpected != allowed_unexpected_keys
    ):
        raise RuntimeError(
            f"{backend} load result changed after audit: missing={loaded_missing}, "
            f"unexpected={loaded_unexpected}"
        )

    report["load_attempted"] = True
    report["load_succeeded"] = True
    report["load_result_missing_keys"] = loaded_missing
    report["load_result_unexpected_keys"] = loaded_unexpected
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report
