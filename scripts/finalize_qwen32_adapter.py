"""Finalize a retrained KITTI360Pose Qwen3-VL-32B LoRA with provenance.

This command copies only the inference adapter artifacts from a selected
ms-swift checkpoint into the experiment's canonical checkpoint directory.
It never writes under the KITTI360Pose dataset root.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.vlmloc_kitti360pose import RENDERER_NAME
from evaluation.vlmloc_release import (
    EXPECTED_DATASET_TOKEN,
    EXPECTED_TEST_SCENES,
    OFFICIAL_SOURCE_COMMIT,
    QWEN3_VL_32B_MODEL_ID,
    TABLE8_PROVENANCE_NAME,
    inspect_adapter,
    inspect_base_model,
    inspect_official_source,
    sha256_file,
)


REQUIRED_ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "args.json",
)
OPTIONAL_ADAPTER_FILES = (
    "README.md",
    "additional_config.json",
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def _resolved_existing_paths(value: Any) -> set[Path]:
    if not isinstance(value, list):
        return set()
    return {
        Path(item).expanduser().resolve()
        for item in value
        if isinstance(item, str) and Path(item).expanduser().is_file()
    }


def _validate_training(
    *,
    adapter: dict[str, Any],
    training_path: Path,
    validation_path: Path,
    training_world_size: int,
) -> dict[str, Any]:
    mismatches: dict[str, Any] = {}
    expected = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "rank": 8,
        "lora_alpha": 16,
        "training_model_type": "qwen3_vl",
        "training_template": "qwen3_vl",
        "training_type": "lora",
        "torch_dtype": "bfloat16",
        "training_target_modules": ["all-linear"],
        "training_num_epochs": 2.0,
        "training_learning_rate": 1e-4,
        "training_attention_implementation": "flash_attn",
        "training_per_device_batch_size": 1,
        "training_seed": 42,
        "data_seed": 42,
    }
    for key, expected_value in expected.items():
        if adapter.get(key) != expected_value:
            mismatches[key] = {
                "expected": expected_value,
                "actual": adapter.get(key),
            }

    model_tokens = " ".join(
        str(adapter.get(key) or "")
        for key in ("base_model_name_or_path", "training_model")
    ).lower()
    if "qwen3-vl-32b-instruct" not in model_tokens:
        mismatches["base_model"] = {
            "expected": "Qwen3-VL-32B-Instruct",
            "actual": model_tokens,
        }

    training_paths = _resolved_existing_paths(
        adapter.get("training_datasets")
    )
    validation_paths = _resolved_existing_paths(
        adapter.get("validation_datasets")
    )
    if training_path.resolve() not in training_paths:
        mismatches["training_dataset"] = {
            "expected": str(training_path.resolve()),
            "actual": sorted(str(path) for path in training_paths),
        }
    if validation_path.resolve() not in validation_paths:
        mismatches["validation_dataset"] = {
            "expected": str(validation_path.resolve()),
            "actual": sorted(str(path) for path in validation_paths),
        }

    accumulation = adapter.get("training_gradient_accumulation_steps")
    per_device = adapter.get("training_per_device_batch_size")
    if (
        not isinstance(accumulation, int)
        or accumulation < 1
        or not isinstance(per_device, int)
        or per_device < 1
    ):
        mismatches["gradient_accumulation_steps"] = {
            "expected": (
                "positive integer accumulation and per-device batch sizes"
            ),
            "actual": {
                "gradient_accumulation_steps": accumulation,
                "per_device_train_batch_size": per_device,
            },
        }
        effective_global_batch = None
    else:
        effective_global_batch = training_world_size * per_device * accumulation
        if effective_global_batch != 4:
            mismatches["effective_global_batch_size"] = {
                "expected": 4,
                "actual": effective_global_batch,
                "world_size": training_world_size,
                "per_device_batch_size": per_device,
                "gradient_accumulation_steps": accumulation,
            }

    if adapter.get("internal_shape_mismatches"):
        mismatches["internal_shape_mismatches"] = adapter[
            "internal_shape_mismatches"
        ]
    if adapter.get("unpaired_lora_keys"):
        mismatches["unpaired_lora_keys"] = adapter["unpaired_lora_keys"]
    return {
        "compatible": not mismatches,
        "mismatches": mismatches,
        "training_world_size": training_world_size,
        "effective_global_batch_size": effective_global_batch,
    }


def finalize(
    *,
    checkpoint_root: Path,
    adapter_dir: Path,
    vlmloc_data_dir: Path,
    training_world_size: int,
    destination: Path | None,
) -> Path:
    checkpoint_root = Path(checkpoint_root).resolve()
    adapter_dir = Path(adapter_dir).resolve()
    data_dir = Path(vlmloc_data_dir).resolve()
    destination = (
        Path(destination).resolve()
        if destination is not None
        else checkpoint_root
        / "VLM-Loc"
        / "table8_kitti360pose_30m"
        / "qwen3_vl_32b"
    )
    preparation_path = data_dir / "vlmloc_data_preparation_audit.json"
    training_path = data_dir / "vlmloc_training_data.json"
    validation_path = data_dir / "vlmloc_validation_data.json"
    required_inputs = (
        preparation_path,
        training_path,
        validation_path,
        *(adapter_dir / name for name in REQUIRED_ADAPTER_FILES),
    )
    missing = [str(path) for path in required_inputs if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing finalization input(s): {missing}")
    if destination.exists():
        raise RuntimeError(
            "Refusing to overwrite an existing finalized adapter: "
            f"{destination}"
        )

    preparation = _load_json(preparation_path)
    immutability = preparation.get("source_dataset_immutability") or {}
    preparation_checks = {
        "renderer_uses_original_processed_cells": (
            preparation.get("renderer") == RENDERER_NAME
            and preparation.get("uses_dense_raw_kitti360_points") is False
        ),
        "source_dataset_unchanged": (
            immutability.get("source_dataset_modified") is False
            and immutability.get("new_kitti360pose_samples_added") == 0
            and immutability.get("snapshot_before")
            == immutability.get("snapshot_after")
        ),
        "geometry_is_30m_224px": (
            preparation.get("bev_range_m") == 30.0
            and preparation.get("image_size_px") == 224
        ),
    }
    if not all(preparation_checks.values()):
        raise RuntimeError(
            "Prepared VLM data violates the immutable KITTI360Pose protocol: "
            f"{preparation_checks}"
        )

    adapter = inspect_adapter(adapter_dir)
    training_audit = _validate_training(
        adapter=adapter,
        training_path=training_path,
        validation_path=validation_path,
        training_world_size=training_world_size,
    )
    if not training_audit["compatible"]:
        raise RuntimeError(
            "Selected adapter does not match the Qwen3-VL-32B experiment: "
            f"{training_audit['mismatches']}"
        )

    vlmloc_root = checkpoint_root / "VLM-Loc"
    base_model_dir = vlmloc_root / "base_models" / "Qwen3-VL-32B-Instruct"
    source_dir = (
        vlmloc_root
        / "official_source"
        / f"nku-3d-vision-{OFFICIAL_SOURCE_COMMIT[:12]}"
    )
    base = inspect_base_model(base_model_dir)
    source = inspect_official_source(source_dir)
    if not (
        base.get("architecture_matches_qwen3_vl")
        and not base.get("missing_base_shards", ["not-audited"])
        and base.get("revision_marker_exists")
    ):
        raise RuntimeError(
            f"Qwen3-VL-32B base-model audit failed: {base}"
        )
    if not (
        source.get("source_commit_matches")
        and not source.get("missing_required_source_files", ["missing"])
    ):
        raise RuntimeError(f"Pinned VLM-Loc source audit failed: {source}")

    source_root = Path(source["vlmloc_source_root"])
    prompt_path = source_root / "system_prompt.txt"
    signature = preparation["testing"]["ordered_dataset_signature"]
    provenance = {
        "schema_version": 1,
        "backend": "vlmloc",
        "experiment": "CMMLoc Top-1 coarse -> VLM-Loc Qwen3-VL-32B fine",
        "comparison_scope": "table8_like_local_full_test_no_filtering",
        "dataset_token": EXPECTED_DATASET_TOKEN,
        "split": "test",
        "query_count": preparation["testing"]["sample_count"],
        "test_scenes": list(EXPECTED_TEST_SCENES),
        "cell_size_m": 30.0,
        "bev_range_m": 30.0,
        "image_size_px": 224,
        "renderer": RENDERER_NAME,
        "uses_original_processed_kitti360pose_cells_only": True,
        "base_model_id": QWEN3_VL_32B_MODEL_ID,
        "base_model_revision": base.get("resolved_model_revision"),
        "source_commit": OFFICIAL_SOURCE_COMMIT,
        "query_order_sha256": signature["ordered_query_sha256"],
        "cell_order_sha256": signature["ordered_cell_sha256"],
        "adapter_config_sha256": adapter["adapter_config_sha256"],
        "adapter_weights_sha256": adapter["adapter_weights_sha256"],
        "args_sha256": adapter["args_sha256"],
        "base_model_config_sha256": base["config_sha256"],
        "system_prompt_sha256": sha256_file(prompt_path),
        "training_dataset_path": str(training_path),
        "training_dataset_sha256": sha256_file(training_path),
        "validation_dataset_path": str(validation_path),
        "validation_dataset_sha256": sha256_file(validation_path),
        "preparation_audit_path": str(preparation_path),
        "preparation_audit_sha256": sha256_file(preparation_path),
        "cmmloc_manifest_path": preparation["cmmloc_manifest_path"],
        "cmmloc_manifest_sha256": preparation["cmmloc_manifest_sha256"],
        "training_world_size": training_world_size,
        "per_device_train_batch_size": adapter[
            "training_per_device_batch_size"
        ],
        "gradient_accumulation_steps": adapter[
            "training_gradient_accumulation_steps"
        ],
        "effective_global_batch_size": training_audit[
            "effective_global_batch_size"
        ],
        "paper_table8_reference": {
            "query_count": 11_404,
            "r_at_5_10_15_m": [0.4036, 0.5169, 0.5474],
            "used_for_sample_filtering": False,
        },
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (
        f".{destination.name}.finalize-{uuid.uuid4().hex}"
    )
    temporary.mkdir()
    try:
        for name in REQUIRED_ADAPTER_FILES:
            shutil.copy2(adapter_dir / name, temporary / name)
        for name in OPTIONAL_ADAPTER_FILES:
            source_file = adapter_dir / name
            if source_file.is_file():
                shutil.copy2(source_file, temporary / name)
        (temporary / TABLE8_PROVENANCE_NAME).write_text(
            json.dumps(provenance, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination / TABLE8_PROVENANCE_NAME


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--vlmloc-data-dir", required=True)
    parser.add_argument(
        "--training-world-size",
        type=int,
        choices=(1, 2, 4),
        required=True,
        help=(
            "Number of data-parallel training processes. Combined with the "
            "adapter args, this must prove global batch size 4."
        ),
    )
    parser.add_argument(
        "--destination",
        help=(
            "Optional finalized adapter directory. Defaults to "
            "VLM-Loc/table8_kitti360pose_30m/qwen3_vl_32b."
        ),
    )
    return parser


def main() -> int:
    cli = build_parser().parse_args()
    path = finalize(
        checkpoint_root=Path(cli.checkpoint_root),
        adapter_dir=Path(cli.adapter_dir),
        vlmloc_data_dir=Path(cli.vlmloc_data_dir),
        training_world_size=cli.training_world_size,
        destination=Path(cli.destination) if cli.destination else None,
    )
    print(f"Finalized Qwen3-VL-32B adapter provenance: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
