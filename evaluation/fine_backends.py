"""Separate fine-stage backends and their fail-closed preflight checks."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.checkpoint_loading import (
    audit_and_load_checkpoint,
    load_checkpoint_state,
)
from evaluation.vlmloc_release import (
    EXPECTED_DATASET_TOKEN,
    OFFICIAL_HF_DATASET,
    OFFICIAL_SOURCE_COMMIT,
    PUBLIC_RELEASE_FILES,
    QWEN3_VL_8B_MODEL_ID,
    default_vlmloc_paths,
    inspect_adapter,
    inspect_base_model,
    inspect_official_source,
    inspect_python_environment,
    validate_table8_artifact_hashes,
    validate_table8_provenance,
)


FROZEN_TEXT_PREFIXES = ("language_encoder.llm_model.",)
T5_LARGE_CONFIG = {
    "model_type": "t5",
    "d_model": 1024,
    "d_ff": 4096,
    "num_layers": 24,
    "num_heads": 16,
    "vocab_size": 32128,
}


@dataclass
class BackendPreflight:
    name: str
    compatible: bool
    report_path: Path
    model: Any = None


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _prefix_counts(keys: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(key.split(".", 1)[0] for key in keys).items()))


def t5_config_record(config: Any) -> dict[str, Any]:
    return {key: getattr(config, key, None) for key in T5_LARGE_CONFIG}


def t5_config_mismatches(config: Any) -> dict[str, dict[str, Any]]:
    actual = t5_config_record(config)
    return {
        key: {"expected": expected, "actual": actual[key]}
        for key, expected in T5_LARGE_CONFIG.items()
        if actual[key] != expected
    }


def validate_sentence_preprocessing() -> dict[str, Any]:
    from nltk import tokenize as text_tokenize

    sample = (
        "The pose is north of a gray road. "
        "The pose is east of a green building."
    )
    sentences = text_tokenize.sent_tokenize(sample)
    if len(sentences) != 2:
        raise RuntimeError(
            f"NLTK sentence preprocessing returned {len(sentences)} sentences, expected 2"
        )
    return {
        "implementation": "nltk.tokenize.sent_tokenize",
        "probe_sentence_count": len(sentences),
    }


def preflight_cmmloc(
    *,
    args: Any,
    checkpoint_path: Path,
    report_dir: Path,
    known_classes: list[str],
    known_colors: list[str],
) -> BackendPreflight:
    """Instantiate only the CMMLoc architecture and audit its own checkpoint."""

    from models.fine.cross_matcher import CrossMatch

    checkpoint_path = checkpoint_path.resolve()
    report_path = (report_dir / "cmmloc_checkpoint_audit.json").resolve()
    model = CrossMatch(known_classes, known_colors, args)
    language = model.language_encoder
    llm_config = language.llm_model.config
    sentence_preprocessing = validate_sentence_preprocessing()
    architecture = {
        "backend_class": "models.fine.cross_matcher.CrossMatch",
        "fine_embed_dim": args.fine_embed_dim,
        "fine_num_decoder_heads": args.fine_num_decoder_heads,
        "fine_num_decoder_layers": args.fine_num_decoder_layers,
        "fine_intra_module_num_heads": args.fine_intra_module_num_heads,
        "fine_intra_module_num_layers": args.fine_intra_module_num_layers,
        "text_backbone_requested": args.hungging_model,
        "text_backbone_config": t5_config_record(llm_config),
        "text_backbone_expected_config": T5_LARGE_CONFIG,
        "tokenizer_class": type(language.tokenizer).__name__,
        "tokenizer_vocab_size": len(language.tokenizer),
        "text_max_length": args.text_max_length,
        "fixed_embedding": args.fixed_embedding,
        "sentence_preprocessing": sentence_preprocessing,
        "pointnet_layers": args.pointnet_layers,
        "pointnet_variation": args.pointnet_variation,
        "pointnet_numpoints": args.pointnet_numpoints,
        "pointnet_features": args.pointnet_features,
        "pointnet_external_prealign_loaded": bool(args.prealign_pointnet_path),
        "preprocessing": {
            "point_transform": "torch_geometric.transforms.FixedPoints",
            "normalize_scale": False,
            "pad_size": args.pad_size,
            "num_mentioned": args.num_mentioned,
        },
    }
    try:
        report = audit_and_load_checkpoint(
            model,
            checkpoint_path,
            report_path,
            backend="cmmloc",
            allowed_missing_prefixes=FROZEN_TEXT_PREFIXES,
            architecture=architecture,
        )
    except Exception:
        return BackendPreflight("cmmloc", False, report_path)

    config_mismatches = t5_config_mismatches(llm_config)
    if config_mismatches:
        report["compatible"] = False
        report["post_load_validation_succeeded"] = False
        report["text_backbone_config_mismatches"] = config_mismatches
        report["post_load_error"] = (
            f"CMMLoc fine requires canonical T5-large config; mismatches: "
            f"{config_mismatches}"
        )
        _write_report(report_path, report)
        return BackendPreflight("cmmloc", False, report_path)

    report["post_load_validation_succeeded"] = True
    report["text_backbone_config_mismatches"] = {}
    _write_report(report_path, report)
    return BackendPreflight("cmmloc", True, report_path, model)


def preflight_mncl(
    *, checkpoint_path: Path, report_dir: Path
) -> BackendPreflight:
    """Report the released MNCL source/checkpoint incompatibility.

    This deliberately does not instantiate CMMLoc's CrossMatch for MNCL.
    """

    checkpoint_path = checkpoint_path.resolve()
    report_path = (report_dir / "mncl_checkpoint_audit.json").resolve()
    state = load_checkpoint_state(checkpoint_path)
    keys = sorted(state)
    checkpoint_signature = {
        "has_flat_attention": any(
            key.startswith("language_encoder.attention.") for key in keys
        ),
        "has_flat_gpool": any(key.startswith("language_encoder.gpool.") for key in keys),
        "has_flat_toare": any(key.startswith("language_encoder.toare.") for key in keys),
        "has_source_msg_namespace": any(
            key.startswith("language_encoder.MSG.") for key in keys
        ),
    }
    source_signature = {
        "repository": "https://github.com/dqliua/MNCL",
        "released_source_language_namespace": "language_encoder.MSG.*",
        "released_source_language_class": "LanguageMsgEncoder",
    }
    report = {
        "schema_version": 1,
        "backend": "mncl",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_key_count": len(keys),
        "checkpoint_prefix_counts": _prefix_counts(keys),
        "checkpoint_signature": checkpoint_signature,
        "source_signature": source_signature,
        "compatible": False,
        "load_attempted": False,
        "load_succeeded": False,
        "missing_keys": [
            "An exact MNCL architecture matching the checkpoint's flat "
            "language_encoder.attention/gpool/toare namespaces is not present."
        ],
        "unexpected_keys": [
            "The public MNCL source expects language_encoder.MSG.*, which the "
            "provided checkpoint does not contain."
        ],
        "shape_mismatches": [],
        "reason": (
            "The supplied MNCL checkpoint cannot be truthfully loaded by the "
            "public released MNCL source, and must not be loaded into CMMLoc's "
            "CrossMatch. Obtain the exact training source revision/architecture "
            "that produced this checkpoint."
        ),
    }
    _write_report(report_path, report)
    return BackendPreflight("mncl", False, report_path)


def _adapter_records(vlmloc_root: Path) -> list[dict[str, Any]]:
    records = []
    for config_path in sorted(vlmloc_root.glob("output/**/adapter_config.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        weights = config_path.parent / "adapter_model.safetensors"
        records.append(
            {
                "adapter_config": str(config_path.resolve()),
                "adapter_weights": str(weights.resolve()),
                "adapter_weights_exists": weights.is_file(),
                "base_model_name_or_path": config.get("base_model_name_or_path"),
                "peft_type": config.get("peft_type"),
                "task_type": config.get("task_type"),
                "rank": config.get("r"),
                "target_modules": config.get("target_modules"),
            }
        )
    return records


def preflight_vlmloc(
    *,
    vlmloc_root: Path,
    report_dir: Path,
    current_dataset_signature: dict[str, Any],
) -> BackendPreflight:
    """Audit VLM-Loc artifacts without pretending they are CrossMatch weights.

    Public CityLoc-K assets and the separately retrained KITTI360Pose/Table-8
    adapter are intentionally recorded as different artifact families.
    """

    vlmloc_root = vlmloc_root.resolve()
    report_path = (report_dir / "vlmloc_checkpoint_audit.json").resolve()
    testing_path = vlmloc_root / "vlmloc_testing_data.json"
    testing_data = (
        json.loads(testing_path.read_text(encoding="utf-8"))
        if testing_path.is_file()
        else []
    )
    image_paths = [
        str(item.get("images", [""])[0])
        for item in testing_data
        if isinstance(item, dict) and item.get("images")
    ]
    dataset_tokens = sorted(
        {
            part
            for image_path in image_paths
            for part in Path(image_path.replace("\\", "/")).parts
            if part.startswith("k360_")
        }
    )
    adapter_records = _adapter_records(vlmloc_root)
    paths = default_vlmloc_paths(vlmloc_root)
    public_qwen8_adapter = inspect_adapter(paths["public_adapter"])
    table8_adapter = inspect_adapter(paths["table8_adapter"])
    base_model = inspect_base_model(paths["base_model"])
    official_source = inspect_official_source(paths["official_source"])
    provenance = validate_table8_provenance(
        paths["table8_provenance"],
        current_query_count=current_dataset_signature["query_count"],
        current_query_order_sha256=current_dataset_signature.get(
            "ordered_query_sha256"
        ),
        current_cell_order_sha256=current_dataset_signature.get(
            "ordered_cell_sha256"
        ),
    )
    artifact_hash_audit = validate_table8_artifact_hashes(
        provenance,
        adapter_audit=table8_adapter,
        base_model_audit=base_model,
        official_source_audit=official_source,
    )
    adapter_semantic_mismatches = {
        key: {"expected": expected, "actual": table8_adapter.get(key)}
        for key, expected in {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "rank": 8,
            "lora_alpha": 16,
            "training_template": "qwen3_vl",
            "training_seed": 42,
            "data_seed": 42,
        }.items()
        if table8_adapter.get(key) != expected
    }
    public_adapter_is_cityloc = bool(
        public_qwen8_adapter.get("adapter_weights_exists")
        and (
            public_qwen8_adapter.get("training_dataset_tokens")
            or len(testing_data) != current_dataset_signature["query_count"]
        )
    )
    exact_adapter_files_exist = bool(
        table8_adapter.get("adapter_config_exists")
        and table8_adapter.get("adapter_weights_exists")
        and table8_adapter.get("args_exists")
    )
    exact_assets_compatible = bool(
        exact_adapter_files_exist
        and provenance.get("compatible")
        and artifact_hash_audit.get("compatible")
        and not adapter_semantic_mismatches
        and base_model.get("architecture_matches_qwen3_vl_8b")
        and not base_model.get("missing_base_shards", ["not-audited"])
        and official_source.get("source_commit_matches")
        and not official_source.get("missing_required_source_files")
    )
    missing_keys = []
    if not exact_adapter_files_exist:
        missing_keys.append(
            "Exact KITTI360Pose 30 m Qwen3-VL-8B adapter_config.json, "
            "adapter_model.safetensors, and args.json are absent from "
            f"{paths['table8_adapter']}."
        )
    if not provenance.get("compatible"):
        missing_keys.append(
            "Exact Table-8 training/preprocessing provenance is absent or does "
            f"not match this ordered {current_dataset_signature['query_count']}-query "
            "test set."
        )
    if provenance.get("exists") and not artifact_hash_audit.get("compatible"):
        missing_keys.append(
            "One or more Table-8 adapter/base-model/prompt/training-data hashes "
            "do not match the provenance record."
        )
    if exact_adapter_files_exist and adapter_semantic_mismatches:
        missing_keys.append(
            "The exact-adapter PEFT/task/rank/alpha/template/seed metadata does "
            f"not match the required VLM-Loc configuration: "
            f"{adapter_semantic_mismatches}."
        )
    if not base_model.get("architecture_matches_qwen3_vl_8b"):
        missing_keys.append(
            f"The complete {QWEN3_VL_8B_MODEL_ID} base model is absent or its "
            "config architecture does not match Qwen3VLForConditionalGeneration."
        )
    if not official_source.get("source_commit_matches"):
        missing_keys.append(
            f"Pinned official VLM-Loc source commit {OFFICIAL_SOURCE_COMMIT} "
            "has not been downloaded and marked."
        )
    table8_like_runner_exists = (
        Path(__file__).with_name("vlmloc_kitti360pose.py").is_file()
    )
    if not table8_like_runner_exists:
        missing_keys.append(
            "The validated Qwen3-VL preparation/world-metric runner is absent."
        )
    unexpected_keys = []
    if public_adapter_is_cityloc:
        unexpected_keys.append(
            "The public checkpoint-3600 adapter records CityLoc-K/50 m training "
            "paths and is quarantined as a reference artifact, not selected as "
            "the KITTI360Pose 30 m fine backend."
        )
    if len(testing_data) != current_dataset_signature["query_count"]:
        unexpected_keys.append(
            f"Public VLM test data contains {len(testing_data)} items instead of "
            f"the current ordered test set's {current_dataset_signature['query_count']}."
        )
    if dataset_tokens:
        unexpected_keys.append(
            f"Public VLM images reference dataset token(s) {dataset_tokens}, "
            f"not {EXPECTED_DATASET_TOKEN}."
        )
    report = {
        "schema_version": 1,
        "backend": "vlmloc",
        "vlmloc_root": str(vlmloc_root),
        "testing_data_path": str(testing_path.resolve()),
        "testing_item_count": len(testing_data),
        "current_test_query_count": current_dataset_signature["query_count"],
        "testing_image_dataset_tokens": dataset_tokens,
        "expected_dataset_token": EXPECTED_DATASET_TOKEN,
        "adapters": adapter_records,
        "adapter_count": len(adapter_records),
        "selected_base_model_id": QWEN3_VL_8B_MODEL_ID,
        "public_release_repository": OFFICIAL_HF_DATASET,
        "public_release_files": PUBLIC_RELEASE_FILES,
        "public_qwen8_cityloc_adapter_audit": public_qwen8_adapter,
        "table8_adapter_audit": table8_adapter,
        "table8_provenance_audit": provenance,
        "table8_artifact_hash_audit": artifact_hash_audit,
        "table8_adapter_semantic_mismatches": adapter_semantic_mismatches,
        "base_model_audit": base_model,
        "official_source_audit": official_source,
        "python_environment_audit": inspect_python_environment(),
        "public_adapter_quarantined_as_cityloc": public_adapter_is_cityloc,
        "exact_assets_compatible_before_runner_check": exact_assets_compatible,
        "table8_like_preparation_and_world_metric_runner_exists": (
            table8_like_runner_exists
        ),
        "shape_comparison_to_base_attempted": bool(
            exact_adapter_files_exist
            and base_model.get("architecture_matches_qwen3_vl_8b")
        ),
        "compatible": False,
        "load_attempted": False,
        "load_succeeded": False,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "shape_mismatches": table8_adapter.get(
            "internal_shape_mismatches", []
        ),
        "public_reference_internal_shape_mismatches": (
            public_qwen8_adapter.get("internal_shape_mismatches", [])
        ),
        "reason": (
            "The downloadable files are PEFT/LoRA CAUSAL_LM adapters and "
            "CityLoc-style 50 m BEV data. The paper's Table-8 supplement says "
            "VLM-Loc was separately retrained on KITTI360Pose, but that exact "
            "30 m adapter/provenance is not in the public release. The public "
            "adapter is therefore not substituted and is never loaded into "
            "CrossMatch or any other architecture. The Table-8-like path "
            "requires local 30 m data generation, retraining, and a one-query "
            "runtime load/generation smoke before full inference."
        ),
        "source_reference": (
            "https://github.com/MCG-NKU/nku-3d-vision/tree/main/vlm-loc"
        ),
    }
    _write_report(report_path, report)
    return BackendPreflight("vlmloc", False, report_path)
