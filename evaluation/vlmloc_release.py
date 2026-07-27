"""Static, dependency-free audits for VLM-Loc release artifacts.

The public VLM-Loc release contains CityLoc LoRA adapters.  The supplementary
KITTI360Pose/Table-8 experiment uses a separately retrained adapter.  These
helpers keep those two artifact families separate and deliberately avoid
loading a PEFT checkpoint into an unrelated PyTorch architecture.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import struct
from pathlib import Path
from typing import Any, Iterable


OFFICIAL_SOURCE_REPOSITORY = "https://github.com/MCG-NKU/nku-3d-vision.git"
OFFICIAL_SOURCE_COMMIT = "494a8b4e3fe9226849697e11d85e70a98e071283"
OFFICIAL_HF_DATASET = "kang233/VLM-Loc"
QWEN3_VL_8B_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
QWEN3_VL_8B_ARCHITECTURE = "Qwen3VLForConditionalGeneration"

PUBLIC_QWEN8_ADAPTER_RELATIVE = Path(
    "output/v0-20251101-125202-qwen3_8b/checkpoint-3600"
)
TABLE8_ADAPTER_RELATIVE = Path(
    "table8_kitti360pose_30m/qwen3_vl_8b"
)
TABLE8_PROVENANCE_NAME = "vlmloc_table8_provenance.json"
DEFAULT_BASE_MODEL_RELATIVE = Path("base_models/Qwen3-VL-8B-Instruct")
DEFAULT_OFFICIAL_SOURCE_RELATIVE = Path(
    f"official_source/nku-3d-vision-{OFFICIAL_SOURCE_COMMIT[:12]}"
)
BASE_MODEL_REVISION_MARKER = ".huggingface_model_revision"

EXPECTED_DATASET_TOKEN = "k360_30-10_scG_pd10_pc4_spY_all"
EXPECTED_TEST_SCENES = (
    "2013_05_28_drive_0003_sync",
    "2013_05_28_drive_0005_sync",
    "2013_05_28_drive_0009_sync",
)

PUBLIC_RELEASE_FILES = {
    "checkpoints.tar.gz": 901_513_025,
    "CityLoc-C.tar.gz": 16_092_589_771,
    "CityLoc-K.tar.gz": 13_213_169_068,
    "dataset_items.tar.gz": 2_494_464,
}


def inspect_python_environment() -> dict[str, Any]:
    distributions = (
        "torch",
        "transformers",
        "peft",
        "ms-swift",
        "qwen-vl-utils",
        "decord",
    )
    versions: dict[str, str | None] = {}
    for name in distributions:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "package_versions": versions,
        "checkpoint_recorded_peft_version": "0.11.1",
        "checkpoint_model_meta_requirement": (
            "transformers>=4.57.0.dev, qwen_vl_utils>=0.0.14, decord"
        ),
        "exact_ms_swift_version_recorded": False,
        "automatic_environment_install_allowed": False,
        "reason": (
            "The released adapter records PEFT 0.11.1 but does not record an "
            "exact ms-swift/transformers/torch environment. Versions are "
            "reported rather than silently replaced."
        ),
    }


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _flatten_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _flatten_strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _flatten_strings(child)


def dataset_tokens(value: Any) -> list[str]:
    """Return all k360_* path components recorded in nested metadata."""

    tokens: set[str] = set()
    for text in _flatten_strings(value):
        normalized = text.replace("\\", "/")
        for match in re.findall(r"(?:^|/)(k360_[^/]+)", normalized):
            tokens.add(match)
    return sorted(tokens)


def read_safetensors_header(path: Path) -> dict[str, Any]:
    """Read safetensors tensor metadata without importing torch/safetensors."""

    path = Path(path)
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"{path} is too short to be a safetensors file")
        (header_length,) = struct.unpack("<Q", raw_length)
        if header_length <= 0 or header_length > 64 * 1024 * 1024:
            raise ValueError(
                f"{path} has an invalid safetensors header length {header_length}"
            )
        header_bytes = handle.read(header_length)
        if len(header_bytes) != header_length:
            raise ValueError(f"{path} has a truncated safetensors header")

    header = json.loads(header_bytes.decode("utf-8").rstrip(" "))
    if not isinstance(header, dict):
        raise ValueError(f"{path} safetensors header is not an object")
    return header


def inspect_adapter(adapter_dir: Path) -> dict[str, Any]:
    adapter_dir = Path(adapter_dir).resolve()
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    args_path = adapter_dir / "args.json"
    record: dict[str, Any] = {
        "adapter_dir": str(adapter_dir),
        "adapter_config_path": str(config_path),
        "adapter_weights_path": str(weights_path),
        "args_path": str(args_path),
        "adapter_config_exists": config_path.is_file(),
        "adapter_weights_exists": weights_path.is_file(),
        "args_exists": args_path.is_file(),
    }
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        record.update(
            {
                "base_model_name_or_path": config.get("base_model_name_or_path"),
                "peft_type": config.get("peft_type"),
                "task_type": config.get("task_type"),
                "rank": config.get("r"),
                "lora_alpha": config.get("lora_alpha"),
                "target_modules": config.get("target_modules"),
                "adapter_config_sha256": sha256_file(config_path),
            }
        )
    if args_path.is_file():
        args = json.loads(args_path.read_text(encoding="utf-8"))
        record.update(
            {
                "training_model": args.get("model"),
                "training_model_type": args.get("model_type"),
                "training_template": args.get("template"),
                "training_type": (
                    args.get("train_type") or args.get("tuner_type")
                ),
                "training_datasets": args.get("dataset"),
                "validation_datasets": args.get("val_dataset"),
                "training_dataset_tokens": dataset_tokens(
                    {
                        "dataset": args.get("dataset"),
                        "val_dataset": args.get("val_dataset"),
                    }
                ),
                "training_seed": args.get("seed"),
                "data_seed": args.get("data_seed"),
                "torch_dtype": args.get("torch_dtype"),
                "training_target_modules": args.get("target_modules"),
                "training_freeze_vit": args.get("freeze_vit"),
                "training_freeze_aligner": args.get("freeze_aligner"),
                "training_num_epochs": args.get("num_train_epochs"),
                "training_learning_rate": args.get("learning_rate"),
                "training_gradient_accumulation_steps": args.get(
                    "gradient_accumulation_steps"
                ),
                "args_sha256": sha256_file(args_path),
            }
        )
    if weights_path.is_file():
        header = read_safetensors_header(weights_path)
        tensors = {
            key: value
            for key, value in header.items()
            if key != "__metadata__"
        }
        dtype_counts: dict[str, int] = {}
        shape_examples: list[dict[str, Any]] = []
        invalid_records: list[str] = []
        tensor_shapes: dict[str, Any] = {}
        for key, value in tensors.items():
            if not isinstance(value, dict):
                invalid_records.append(key)
                continue
            dtype = str(value.get("dtype"))
            tensor_shapes[key] = value.get("shape")
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
            if len(shape_examples) < 20:
                shape_examples.append(
                    {
                        "key": key,
                        "shape": value.get("shape"),
                        "dtype": dtype,
                        "data_offsets": value.get("data_offsets"),
                    }
                )
        lora_pair_count = 0
        unpaired_lora_keys: list[str] = []
        internal_shape_mismatches: list[dict[str, Any]] = []
        for key, shape_a in tensor_shapes.items():
            marker = ".lora_A.weight"
            if not key.endswith(marker):
                continue
            key_b = key[: -len(marker)] + ".lora_B.weight"
            shape_b = tensor_shapes.get(key_b)
            if shape_b is None:
                unpaired_lora_keys.append(key)
                continue
            lora_pair_count += 1
            valid_2d = (
                isinstance(shape_a, list)
                and len(shape_a) == 2
                and isinstance(shape_b, list)
                and len(shape_b) == 2
            )
            configured_rank = record.get("rank")
            if (
                not valid_2d
                or shape_a[0] != shape_b[1]
                or (
                    isinstance(configured_rank, int)
                    and (
                        shape_a[0] != configured_rank
                        or shape_b[1] != configured_rank
                    )
                )
            ):
                internal_shape_mismatches.append(
                    {
                        "lora_A_key": key,
                        "lora_A_shape": shape_a,
                        "lora_B_key": key_b,
                        "lora_B_shape": shape_b,
                        "configured_rank": configured_rank,
                    }
                )
        for key in tensor_shapes:
            marker = ".lora_B.weight"
            if key.endswith(marker):
                key_a = key[: -len(marker)] + ".lora_A.weight"
                if key_a not in tensor_shapes:
                    unpaired_lora_keys.append(key)
        record.update(
            {
                "adapter_weights_bytes": weights_path.stat().st_size,
                "adapter_weights_sha256": sha256_file(weights_path),
                "tensor_key_count": len(tensors),
                "tensor_dtype_counts": dict(sorted(dtype_counts.items())),
                "tensor_shape_examples": shape_examples,
                "invalid_tensor_metadata_keys": invalid_records,
                "lora_pair_count": lora_pair_count,
                "unpaired_lora_keys": sorted(unpaired_lora_keys),
                "internal_shape_mismatches": internal_shape_mismatches,
            }
        )
    return record


def inspect_base_model(base_model_dir: Path) -> dict[str, Any]:
    base_model_dir = Path(base_model_dir).resolve()
    config_path = base_model_dir / "config.json"
    index_path = base_model_dir / "model.safetensors.index.json"
    revision_path = base_model_dir / BASE_MODEL_REVISION_MARKER
    record: dict[str, Any] = {
        "base_model_dir": str(base_model_dir),
        "config_exists": config_path.is_file(),
        "safetensors_index_exists": index_path.is_file(),
        "revision_marker_path": str(revision_path),
        "revision_marker_exists": revision_path.is_file(),
        "resolved_model_revision": (
            revision_path.read_text(encoding="utf-8").strip()
            if revision_path.is_file()
            else None
        ),
    }
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        architectures = config.get("architectures") or []
        record.update(
            {
                "model_type": config.get("model_type"),
                "architectures": architectures,
                "architecture_matches_qwen3_vl_8b": (
                    QWEN3_VL_8B_ARCHITECTURE in architectures
                ),
                "config_sha256": sha256_file(config_path),
            }
        )
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map") or {}
        shard_names = sorted(set(weight_map.values()))
        missing_shards = [
            name for name in shard_names if not (base_model_dir / name).is_file()
        ]
        record.update(
            {
                "base_tensor_key_count": len(weight_map),
                "base_shard_count": len(shard_names),
                "missing_base_shards": missing_shards,
                "index_sha256": sha256_file(index_path),
            }
        )
    return record


def inspect_official_source(source_root: Path) -> dict[str, Any]:
    source_root = Path(source_root).resolve()
    marker_path = source_root / ".vlmloc_source_commit"
    if (source_root / "vlm-loc").is_dir():
        vlmloc_source = source_root / "vlm-loc"
    else:
        vlmloc_source = source_root
    required = (
        "README.md",
        "test.sh",
        "recall.py",
        "system_prompt.txt",
        "datapreparation/kitti360pose/prepare_cityloc-k.py",
        "data/dataset_generation_semantics_cityloc-k.py",
    )
    missing = [name for name in required if not (vlmloc_source / name).is_file()]
    marker = (
        marker_path.read_text(encoding="utf-8").strip()
        if marker_path.is_file()
        else None
    )
    return {
        "source_root": str(source_root),
        "vlmloc_source_root": str(vlmloc_source),
        "source_commit_marker": marker,
        "expected_source_commit": OFFICIAL_SOURCE_COMMIT,
        "source_commit_matches": marker == OFFICIAL_SOURCE_COMMIT,
        "missing_required_source_files": missing,
    }


def validate_table8_provenance(
    provenance_path: Path,
    *,
    current_query_count: int,
    current_query_order_sha256: str | None,
    current_cell_order_sha256: str | None,
) -> dict[str, Any]:
    provenance_path = Path(provenance_path).resolve()
    if not provenance_path.is_file():
        return {
            "provenance_path": str(provenance_path),
            "exists": False,
            "compatible": False,
            "mismatches": {
                "provenance": {
                    "expected": "a signed/recorded KITTI360Pose 30 m training record",
                    "actual": "missing",
                }
            },
        }

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected: dict[str, Any] = {
        "schema_version": 1,
        "backend": "vlmloc",
        "dataset_token": EXPECTED_DATASET_TOKEN,
        "split": "test",
        "query_count": current_query_count,
        "test_scenes": list(EXPECTED_TEST_SCENES),
        "cell_size_m": 30.0,
        "bev_range_m": 30.0,
        "image_size_px": 224,
        "base_model_id": QWEN3_VL_8B_MODEL_ID,
        "source_commit": OFFICIAL_SOURCE_COMMIT,
    }
    if current_query_order_sha256 is not None:
        expected["query_order_sha256"] = current_query_order_sha256
    if current_cell_order_sha256 is not None:
        expected["cell_order_sha256"] = current_cell_order_sha256
    mismatches = {
        key: {"expected": value, "actual": provenance.get(key)}
        for key, value in expected.items()
        if provenance.get(key) != value
    }
    required_hashes = (
        "adapter_config_sha256",
        "adapter_weights_sha256",
        "args_sha256",
        "base_model_config_sha256",
        "system_prompt_sha256",
        "training_dataset_sha256",
        "validation_dataset_sha256",
    )
    for key in required_hashes:
        value = provenance.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
            mismatches[key] = {
                "expected": "64-character SHA-256",
                "actual": value,
            }
    return {
        "provenance_path": str(provenance_path),
        "exists": True,
        "compatible": not mismatches,
        "mismatches": mismatches,
        "record": provenance,
        "sha256": sha256_file(provenance_path),
    }


def validate_table8_artifact_hashes(
    provenance_audit: dict[str, Any],
    *,
    adapter_audit: dict[str, Any],
    base_model_audit: dict[str, Any],
    official_source_audit: dict[str, Any],
) -> dict[str, Any]:
    """Cross-check provenance hashes against every locally available artifact."""

    record = provenance_audit.get("record")
    mismatches: dict[str, dict[str, Any]] = {}
    if not isinstance(record, dict):
        return {
            "compatible": False,
            "mismatches": {
                "provenance_record": {
                    "expected": "loaded JSON provenance record",
                    "actual": None,
                }
            },
        }

    direct_hashes = {
        "adapter_config_sha256": adapter_audit.get("adapter_config_sha256"),
        "adapter_weights_sha256": adapter_audit.get("adapter_weights_sha256"),
        "args_sha256": adapter_audit.get("args_sha256"),
        "base_model_config_sha256": base_model_audit.get("config_sha256"),
    }
    for key, actual in direct_hashes.items():
        expected = record.get(key)
        if actual != expected:
            mismatches[key] = {"expected": expected, "actual": actual}

    source_root_value = official_source_audit.get("vlmloc_source_root")
    prompt_path = (
        Path(source_root_value) / "system_prompt.txt"
        if isinstance(source_root_value, str)
        else None
    )
    actual_prompt_hash = (
        sha256_file(prompt_path)
        if prompt_path is not None and prompt_path.is_file()
        else None
    )
    if actual_prompt_hash != record.get("system_prompt_sha256"):
        mismatches["system_prompt_sha256"] = {
            "expected": record.get("system_prompt_sha256"),
            "actual": actual_prompt_hash,
        }

    provenance_parent = Path(
        provenance_audit.get("provenance_path", ".")
    ).resolve().parent
    for prefix in ("training", "validation"):
        path_key = f"{prefix}_dataset_path"
        hash_key = f"{prefix}_dataset_sha256"
        raw_path = record.get(path_key)
        if not isinstance(raw_path, str) or not raw_path:
            mismatches[path_key] = {
                "expected": "existing absolute path or path relative to provenance",
                "actual": raw_path,
            }
            continue
        artifact_path = Path(raw_path)
        if not artifact_path.is_absolute():
            artifact_path = provenance_parent / artifact_path
        artifact_path = artifact_path.resolve()
        actual_hash = (
            sha256_file(artifact_path) if artifact_path.is_file() else None
        )
        if actual_hash != record.get(hash_key):
            mismatches[hash_key] = {
                "expected": record.get(hash_key),
                "actual": actual_hash,
                "resolved_path": str(artifact_path),
            }

    return {
        "compatible": not mismatches,
        "mismatches": mismatches,
        "adapter_semantics": {
            "peft_type": adapter_audit.get("peft_type"),
            "task_type": adapter_audit.get("task_type"),
            "rank": adapter_audit.get("rank"),
            "lora_alpha": adapter_audit.get("lora_alpha"),
            "training_template": adapter_audit.get("training_template"),
            "training_seed": adapter_audit.get("training_seed"),
            "data_seed": adapter_audit.get("data_seed"),
        },
    }


def default_vlmloc_paths(vlmloc_root: Path) -> dict[str, Path]:
    vlmloc_root = Path(vlmloc_root).resolve()
    return {
        "public_adapter": vlmloc_root / PUBLIC_QWEN8_ADAPTER_RELATIVE,
        "table8_adapter": vlmloc_root / TABLE8_ADAPTER_RELATIVE,
        "table8_provenance": (
            vlmloc_root / TABLE8_ADAPTER_RELATIVE / TABLE8_PROVENANCE_NAME
        ),
        "base_model": vlmloc_root / DEFAULT_BASE_MODEL_RELATIVE,
        "official_source": vlmloc_root / DEFAULT_OFFICIAL_SOURCE_RELATIVE,
    }
