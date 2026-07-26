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


FROZEN_TEXT_PREFIXES = ("language_encoder.llm_model.",)
T5_LARGE_CONFIG = {
    "model_type": "t5",
    "d_model": 1024,
    "d_ff": 2816,
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


def is_t5_large(config: Any) -> bool:
    return t5_config_record(config) == T5_LARGE_CONFIG


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

    if not is_t5_large(llm_config):
        report["compatible"] = False
        report["load_succeeded"] = False
        report["post_load_error"] = (
            "CMMLoc fine source/checkpoint requires the T5-large 1024-dimensional "
            "text backbone."
        )
        _write_report(report_path, report)
        return BackendPreflight("cmmloc", False, report_path)

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
    """Inspect VLM-Loc adapters/data without pretending they are CrossMatch weights."""

    vlmloc_root = vlmloc_root.resolve()
    report_path = (report_dir / "vlmloc_checkpoint_audit.json").resolve()
    testing_path = vlmloc_root / "vlmloc_testing_data.json"
    testing_data = json.loads(testing_path.read_text(encoding="utf-8"))
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
    report = {
        "schema_version": 1,
        "backend": "vlmloc",
        "vlmloc_root": str(vlmloc_root),
        "testing_data_path": str(testing_path.resolve()),
        "testing_item_count": len(testing_data),
        "current_test_query_count": current_dataset_signature["query_count"],
        "testing_image_dataset_tokens": dataset_tokens,
        "expected_dataset_token": "k360_30-10_scG_pd10_pc4_spY_all",
        "adapters": adapter_records,
        "adapter_count": len(adapter_records),
        "compatible": False,
        "load_attempted": False,
        "load_succeeded": False,
        "missing_keys": [
            "No exact KITTI360Pose Table-8 VLM-Loc fine inference/BEV-rendering "
            "source is present in this repository.",
            "The adapter base models are external absolute paths and are not "
            "contained in the adapter checkpoints.",
        ],
        "unexpected_keys": [
            f"Provided VLM test data contains {len(testing_data)} items instead of "
            f"the current ordered test set's {current_dataset_signature['query_count']}.",
            f"Provided VLM images reference dataset token(s) {dataset_tokens}.",
        ],
        "shape_mismatches": [],
        "reason": (
            "The provided files are PEFT/LoRA CAUSAL_LM adapters and CityLoc-style "
            "50 m BEV data, not a fine CrossMatch checkpoint for the current 30 m "
            "KITTI360Pose cells. Loading them into another architecture would "
            "change model semantics."
        ),
        "source_reference": (
            "https://github.com/MCG-NKU/nku-3d-vision/tree/main/vlm-loc"
        ),
    }
    _write_report(report_path, report)
    return BackendPreflight("vlmloc", False, report_path)
