"""Fail-closed audit for the VLM-Loc Table-8 reproduction protocol.

The requested experiment is deliberately kept separate from the repository's
``my_model`` shared-retrieval experiment:

    official CMMLoc coarse Top-1 -> VLM-Loc fine -> KITTI360Pose test

This module performs static release/source/dataset checks.  It does not invent
an architecture for an incompatible checkpoint and it never substitutes the
public CityLoc VLM-Loc adapter for the separately retrained Table-8 adapter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.checkpoint_metadata import inspect_pytorch_zip_checkpoint
from evaluation.vlmloc_release import sha256_file


PROTOCOL_NAME = "vlm-loc-table8/cmmloc-top1/vlmloc-fine/kitti360pose-test"
PROTOCOL_VERSION = 1
EXPECTED_SPLIT = "test"
EXPECTED_SEED = 42
EXPECTED_TOP_K = (1,)
EXPECTED_THRESHOLDS_M = (5, 10, 15)
EXPECTED_QUERY_COUNT = 11_404
EXPECTED_LOCAL_FULL_QUERY_COUNT = 11_505
EXPECTED_CELL_SIZE_M = 30.0
EXPECTED_DATASET_TOKEN = "k360_30-10_scG_pd10_pc4_spY_all"
EXPECTED_TEST_SCENES = (
    "2013_05_28_drive_0003_sync",
    "2013_05_28_drive_0005_sync",
    "2013_05_28_drive_0009_sync",
)
TABLE8_VLMLOC_R_AT_METERS = {
    "5": 0.4036,
    "10": 0.5169,
    "15": 0.5474,
}

CMMLOC_SOURCE_REPOSITORY = "https://github.com/kevin301342/CMMLoc"
CMMLOC_SOURCE_COMMIT = "d49458963d4caeddf7e9169e0e6384cb8223e22c"
CMMLOC_COARSE_CHECKPOINT_SHA256 = (
    "5e14e158c3de1fc046d9b970ef1d06c6d4a98d55a1cfdd09f6d26dfc23076f85"
)
CMMLOC_COARSE_CHECKPOINT_BYTES = 118_308_613
CMMLOC_COARSE_KEY_COUNT = 446
CMMLOC_COARSE_PREFIX_COUNTS = {
    "cell_encoder1": 130,
    "cell_encoder2": 130,
    "cell_input_proj": 4,
    "language_encoder": 31,
    "modular_vector_mapping": 1,
    "obj_inter_module": 24,
    "object_encoder": 122,
    "object_pos_embed": 3,
    "weight_token": 1,
}

# These checkpoint modules are absent from the pinned public constructor and
# are consequently ignored by the public evaluation script's strict=False.
CMMLOC_CHECKPOINT_ONLY_PREFIX_COUNTS = {
    "cell_encoder2": 130,
    "modular_vector_mapping": 1,
    "obj_inter_module": 24,
}
CMMLOC_PUBLIC_COARSE_SOURCE_SHA256 = (
    "5853a9f1fe40bdae19da5d7ca0d94c9c464c967803773a3e731db1a7ab541558"
)
CMMLOC_PUBLIC_PIPELINE_SOURCE_SHA256 = (
    "bf982655d55a5dfe5fccb107c183d422f51e3d0aa853271611789be927b7a753"
)


def validate_table8_protocol(
    *,
    split: str,
    seed: int,
    top_k: Sequence[int],
    thresholds_m: Sequence[int],
) -> dict[str, Any]:
    actual = {
        "split": split,
        "seed": int(seed),
        "top_k": list(top_k),
        "thresholds_m": list(thresholds_m),
    }
    expected = {
        "split": EXPECTED_SPLIT,
        "seed": EXPECTED_SEED,
        "top_k": list(EXPECTED_TOP_K),
        "thresholds_m": list(EXPECTED_THRESHOLDS_M),
    }
    mismatches = {
        key: {"expected": expected[key], "actual": actual[key]}
        for key in expected
        if actual[key] != expected[key]
    }
    return {
        "protocol_name": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "coarse_backend": "cmmloc",
        "fine_backend": "vlmloc",
        "expected": expected,
        "actual": actual,
        "compatible": not mismatches,
        "mismatches": mismatches,
        "paper_reference_metrics": {
            "model": "VLM-Loc",
            "metric": "localization recall",
            "values": TABLE8_VLMLOC_R_AT_METERS,
        },
    }


def audit_table8_dataset(
    *,
    data_root: Path,
    dataset_signature: Mapping[str, Any],
    require_exact_paper_count: bool = True,
) -> dict[str, Any]:
    actual_scenes = list(dataset_signature.get("ordered_scenes", []))
    expected_query_count = (
        EXPECTED_QUERY_COUNT
        if require_exact_paper_count
        else EXPECTED_LOCAL_FULL_QUERY_COUNT
    )
    expected = {
        "dataset_token": EXPECTED_DATASET_TOKEN,
        "split": EXPECTED_SPLIT,
        "query_count": expected_query_count,
        "ordered_scenes": list(EXPECTED_TEST_SCENES),
        "cell_size_m": EXPECTED_CELL_SIZE_M,
    }
    actual = {
        "dataset_token": Path(data_root).resolve().name,
        "split": EXPECTED_SPLIT,
        "query_count": dataset_signature.get("query_count"),
        "ordered_scenes": actual_scenes,
        # The directory token is the only static 30 m declaration available;
        # the complete ordered cell fingerprint is retained below.
        "cell_size_m": (
            EXPECTED_CELL_SIZE_M
            if Path(data_root).resolve().name == EXPECTED_DATASET_TOKEN
            else None
        ),
    }
    mismatches = {
        key: {"expected": expected[key], "actual": actual[key]}
        for key in expected
        if actual[key] != expected[key]
    }
    warnings = []
    if (
        not require_exact_paper_count
        and actual["query_count"] == EXPECTED_LOCAL_FULL_QUERY_COUNT
    ):
        warnings.append(
            {
                "field": "query_count",
                "paper_value": EXPECTED_QUERY_COUNT,
                "local_full_test_value": EXPECTED_LOCAL_FULL_QUERY_COUNT,
                "delta": (
                    EXPECTED_LOCAL_FULL_QUERY_COUNT - EXPECTED_QUERY_COUNT
                ),
                "relative_delta_from_paper": (
                    EXPECTED_LOCAL_FULL_QUERY_COUNT - EXPECTED_QUERY_COUNT
                )
                / EXPECTED_QUERY_COUNT,
                "worst_case_recall_shift_if_only_101_samples_are_added": {
                    "fraction": (
                        EXPECTED_LOCAL_FULL_QUERY_COUNT - EXPECTED_QUERY_COUNT
                    )
                    / EXPECTED_LOCAL_FULL_QUERY_COUNT,
                    "percentage_points": 100
                    * (
                        EXPECTED_LOCAL_FULL_QUERY_COUNT
                        - EXPECTED_QUERY_COUNT
                    )
                    / EXPECTED_LOCAL_FULL_QUERY_COUNT,
                    "assumption": (
                        "The paper's 11,404 ordered samples are an unchanged "
                        "subset and only 101 samples are added."
                    ),
                },
                "effect": (
                    "Accepted for the requested full local test. Table-8 "
                    "metrics remain a sanity-check reference, not an exact "
                    "reproduction target."
                ),
            }
        )
    return {
        "data_root": str(Path(data_root).resolve()),
        "evaluation_scope": (
            "exact_table8_11404"
            if require_exact_paper_count
            else "table8_like_local_full_test_11505"
        ),
        "expected": expected,
        "actual": actual,
        "compatible": not mismatches,
        "mismatches": mismatches,
        "warnings": warnings,
        "paper_query_count_reference": EXPECTED_QUERY_COUNT,
        "ordered_query_sha256": dataset_signature.get("ordered_query_sha256"),
        "ordered_cell_sha256": dataset_signature.get("ordered_cell_sha256"),
        "scene_query_counts": dataset_signature.get("scene_query_counts"),
        "scene_cell_counts": dataset_signature.get("scene_cell_counts"),
        "reason": (
            "Exact reproduction requires the paper's 11,404 samples."
            if require_exact_paper_count
            else (
                "The requested evaluation uses every ordered local test query. "
                "The 101-sample difference from Table 8 is recorded as a "
                "comparison warning and does not fail this dataset audit."
            )
        ),
    }


def _source_paths(source_root: Path) -> tuple[Path, Path]:
    source_root = Path(source_root).resolve()
    return (
        source_root / "models" / "coarse" / "cell_retrieval.py",
        source_root / "evaluation" / "pipeline.py",
    )


def audit_cmmloc_public_source(source_root: Path) -> dict[str, Any]:
    coarse_source, pipeline_source = _source_paths(source_root)
    record: dict[str, Any] = {
        "source_root": str(Path(source_root).resolve()),
        "source_repository": CMMLOC_SOURCE_REPOSITORY,
        "expected_source_commit": CMMLOC_SOURCE_COMMIT,
        "coarse_source_path": str(coarse_source),
        "pipeline_source_path": str(pipeline_source),
        "coarse_source_exists": coarse_source.is_file(),
        "pipeline_source_exists": pipeline_source.is_file(),
    }
    if not coarse_source.is_file() or not pipeline_source.is_file():
        record.update(
            {
                "compatible": False,
                "missing_files": [
                    str(path)
                    for path in (coarse_source, pipeline_source)
                    if not path.is_file()
                ],
                "reason": "Pinned official CMMLoc source is incomplete or absent.",
            }
        )
        return record

    coarse_text = coarse_source.read_text(encoding="utf-8")
    pipeline_text = pipeline_source.read_text(encoding="utf-8")
    coarse_hash = sha256_file(coarse_source)
    pipeline_hash = sha256_file(pipeline_source)
    missing_checkpoint_modules = {
        prefix: prefix not in coarse_text
        for prefix in CMMLOC_CHECKPOINT_ONLY_PREFIX_COUNTS
    }
    uses_unchecked_strict_false = bool(
        "load_state_dict(model_coarse_dic, strict = False)" in pipeline_text
        or "load_state_dict(model_coarse_dic, strict=False)" in pipeline_text
    )
    source_files_match = bool(
        coarse_hash == CMMLOC_PUBLIC_COARSE_SOURCE_SHA256
        and pipeline_hash == CMMLOC_PUBLIC_PIPELINE_SOURCE_SHA256
    )
    record.update(
        {
            "coarse_source_sha256": coarse_hash,
            "expected_coarse_source_sha256": (
                CMMLOC_PUBLIC_COARSE_SOURCE_SHA256
            ),
            "pipeline_source_sha256": pipeline_hash,
            "expected_pipeline_source_sha256": (
                CMMLOC_PUBLIC_PIPELINE_SOURCE_SHA256
            ),
            "source_files_match_pinned_commit": source_files_match,
            "checkpoint_modules_absent_from_public_constructor": (
                missing_checkpoint_modules
            ),
            "public_pipeline_uses_strict_false": uses_unchecked_strict_false,
            # This is intentionally false even when the two source hashes match:
            # the pinned source itself is incompatible with its released state
            # dict and discards 155 keys.
            "compatible": False,
            "reason": (
                "The pinned public CMMLoc constructor omits 155 tensors present "
                "in the official checkpoint, while its evaluation script loads "
                "with strict=False. Reproducing that silent discard would not "
                "satisfy the requested checkpoint/architecture audit."
            ),
        }
    )
    return record


def audit_cmmloc_coarse_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        return {
            "checkpoint_path": str(checkpoint_path),
            "exists": False,
            "compatible_with_official_artifact": False,
            "compatible_with_public_source": False,
            "official_artifact_mismatches": {
                "checkpoint": {
                    "expected": "official CMMLoc coarse.pth",
                    "actual": "missing",
                }
            },
            "missing_keys": ["official CMMLoc coarse checkpoint"],
            "unexpected_keys": [],
            "unexpected_prefix_counts": {},
            "unexpected_key_count": 0,
            "shape_mismatches": [],
        }

    metadata = inspect_pytorch_zip_checkpoint(checkpoint_path)
    checkpoint_hash = sha256_file(checkpoint_path)
    actual_prefix_counts = metadata["prefix_counts"]
    text_projection = metadata["tensors"].get(
        "language_encoder.inter_mlp.0.0.weight", {}
    )
    text_projection_shape = text_projection.get("shape")
    inferred_text_hidden_dim = (
        text_projection_shape[1]
        if isinstance(text_projection_shape, list)
        and len(text_projection_shape) == 2
        else None
    )
    checkpoint_only_keys = sorted(
        key
        for key in metadata["tensors"]
        if key.split(".", 1)[0] in CMMLOC_CHECKPOINT_ONLY_PREFIX_COUNTS
    )
    prefix_mismatches = {
        prefix: {
            "expected": expected_count,
            "actual": actual_prefix_counts.get(prefix, 0),
        }
        for prefix, expected_count in CMMLOC_COARSE_PREFIX_COUNTS.items()
        if actual_prefix_counts.get(prefix, 0) != expected_count
    }
    extra_prefixes = sorted(
        set(actual_prefix_counts) - set(CMMLOC_COARSE_PREFIX_COUNTS)
    )
    artifact_mismatches: dict[str, Any] = {}
    if checkpoint_hash != CMMLOC_COARSE_CHECKPOINT_SHA256:
        artifact_mismatches["sha256"] = {
            "expected": CMMLOC_COARSE_CHECKPOINT_SHA256,
            "actual": checkpoint_hash,
        }
    if checkpoint_path.stat().st_size != CMMLOC_COARSE_CHECKPOINT_BYTES:
        artifact_mismatches["bytes"] = {
            "expected": CMMLOC_COARSE_CHECKPOINT_BYTES,
            "actual": checkpoint_path.stat().st_size,
        }
    if metadata["key_count"] != CMMLOC_COARSE_KEY_COUNT:
        artifact_mismatches["key_count"] = {
            "expected": CMMLOC_COARSE_KEY_COUNT,
            "actual": metadata["key_count"],
        }
    if prefix_mismatches:
        artifact_mismatches["prefix_counts"] = prefix_mismatches
    if extra_prefixes:
        artifact_mismatches["extra_prefixes"] = extra_prefixes

    return {
        "checkpoint_path": str(checkpoint_path),
        "exists": True,
        "checkpoint_sha256": checkpoint_hash,
        "expected_official_sha256": CMMLOC_COARSE_CHECKPOINT_SHA256,
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_key_count": metadata["key_count"],
        "checkpoint_prefix_counts": actual_prefix_counts,
        "checkpoint_text_projection_shape": text_projection_shape,
        "inferred_text_hidden_dim": inferred_text_hidden_dim,
        "t5_large_hidden_dim_matches": inferred_text_hidden_dim == 1024,
        "official_artifact_mismatches": artifact_mismatches,
        "compatible_with_official_artifact": not artifact_mismatches,
        "compatible_with_public_source": False,
        "missing_keys": [],
        "unexpected_keys": checkpoint_only_keys,
        "unexpected_prefix_counts": CMMLOC_CHECKPOINT_ONLY_PREFIX_COUNTS,
        "unexpected_key_count": len(checkpoint_only_keys),
        "shape_mismatches": [],
        "reason": (
            "The file is audited against the official release artifact. It is "
            "not declared source-compatible because 155 checkpoint keys have no "
            "corresponding module in the pinned public constructor."
        ),
    }


def audit_cmmloc_runtime_configuration(
    *,
    requested_text_backbone: str,
    checkpoint_audit: Mapping[str, Any],
    source_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Record source/checkpoint configuration without filling release gaps."""

    requested_is_t5_large = (
        requested_text_backbone.strip().replace("\\", "/").rstrip("/").split("/")[-1]
        == "t5-large"
    )
    exact_text_revision_recorded = False
    source_files_match = bool(
        source_audit.get("source_files_match_pinned_commit")
    )
    record = {
        "text_backbone": {
            "requested": requested_text_backbone,
            "expected_family": "T5-large",
            "requested_name_matches": requested_is_t5_large,
            "checkpoint_hidden_dim": checkpoint_audit.get(
                "inferred_text_hidden_dim"
            ),
            "expected_hidden_dim": 1024,
            "checkpoint_hidden_dim_matches": checkpoint_audit.get(
                "t5_large_hidden_dim_matches", False
            ),
            "public_source_model_path": "PATH_TO_T5",
            "exact_model_revision_recorded": exact_text_revision_recorded,
            "tokenizer_class": "AutoTokenizer",
            "encoder_class": "T5EncoderModel",
        },
        "text_preprocessing": {
            "sentence_splitter": "nltk.tokenize.sent_tokenize",
            "tokenizer_padding": "longest",
            "tokenizer_truncation": False,
            "tokenizer_max_length": None,
        },
        "pointnet": {
            "pointnet_layers": 3,
            "pointnet_variation": 0,
            "pointnet_numpoints": 256,
            "pointnet_features": 2,
            "use_features": ["class", "color", "position", "num"],
            "point_transform": "FixedPoints(256) when --no_pc_augment is set",
            "public_source_pointnet_path": "PATH_TO_POINTNET",
            "exact_pointnet_provenance_recorded": False,
        },
        "dataset_preprocessing": {
            "shuffle_hints": False,
            "flip_poses": False,
            "sample_close_cell": False,
            "test_scene_order": list(EXPECTED_TEST_SCENES),
        },
        "source_files_match_pinned_commit": source_files_match,
    }
    record["compatible"] = bool(
        requested_is_t5_large
        and checkpoint_audit.get("t5_large_hidden_dim_matches")
        and exact_text_revision_recorded
        and record["pointnet"]["exact_pointnet_provenance_recorded"]
        and source_files_match
    )
    record["reason"] = (
        "The checkpoint shape supports a 1024-dimensional T5-large family, "
        "and the pinned source exposes the listed preprocessing/PointNet "
        "defaults. However, it records only PATH_TO_T5/PATH_TO_POINTNET, not "
        "the exact text-model revision or PointNet artifact provenance."
    )
    return record


def build_table8_preflight_report(
    *,
    data_root: Path,
    dataset_signature: Mapping[str, Any],
    cmmloc_coarse_checkpoint: Path,
    cmmloc_source_root: Path,
    vlmloc_report: Mapping[str, Any],
    cmmloc_release_runtime_report: Mapping[str, Any] | None = None,
    allow_public_release_behavior: bool = False,
    requested_text_backbone: str = "t5-large",
    require_exact_paper_count: bool = True,
    split: str = EXPECTED_SPLIT,
    seed: int = EXPECTED_SEED,
    top_k: Sequence[int] = EXPECTED_TOP_K,
    thresholds_m: Sequence[int] = EXPECTED_THRESHOLDS_M,
) -> dict[str, Any]:
    protocol = validate_table8_protocol(
        split=split,
        seed=seed,
        top_k=top_k,
        thresholds_m=thresholds_m,
    )
    dataset = audit_table8_dataset(
        data_root=data_root,
        dataset_signature=dataset_signature,
        require_exact_paper_count=require_exact_paper_count,
    )
    checkpoint = audit_cmmloc_coarse_checkpoint(cmmloc_coarse_checkpoint)
    source = audit_cmmloc_public_source(cmmloc_source_root)
    configuration = audit_cmmloc_runtime_configuration(
        requested_text_backbone=requested_text_backbone,
        checkpoint_audit=checkpoint,
        source_audit=source,
    )

    vlmloc_compatible = bool(vlmloc_report.get("compatible"))
    cmmloc_release_compatible = bool(
        allow_public_release_behavior
        and cmmloc_release_runtime_report
        and cmmloc_release_runtime_report.get("compatible")
        and cmmloc_release_runtime_report.get(
            "public_release_inference_behavior_claimed"
        )
        and not cmmloc_release_runtime_report.get(
            "exact_training_architecture_claimed", True
        )
    )
    checks = {
        "protocol": protocol["compatible"],
        "dataset": dataset["compatible"],
        "cmmloc_coarse_official_artifact": checkpoint[
            "compatible_with_official_artifact"
        ],
        "cmmloc_coarse_source_architecture": (
            cmmloc_release_compatible
            if allow_public_release_behavior
            else checkpoint["compatible_with_public_source"]
            and source["compatible"]
        ),
        "cmmloc_text_pointnet_preprocessing": (
            cmmloc_release_compatible
            if allow_public_release_behavior
            else configuration["compatible"]
        ),
        "vlmloc_table8_fine_backend": vlmloc_compatible,
    }
    blockers = []
    warnings = list(dataset.get("warnings", []))
    if allow_public_release_behavior:
        warnings.append(
            {
                "field": "cmmloc_release_architecture",
                "effect": (
                    "This full-test protocol reproduces the pinned public "
                    "CMMLoc inference constructor. It explicitly reports and "
                    "ignores exactly 155 checkpoint-only tensors, matching "
                    "the public evaluation behavior. It does not claim that "
                    "the unpublished training constructor was recovered."
                ),
                "ignored_prefix_counts": CMMLOC_CHECKPOINT_ONLY_PREFIX_COUNTS,
            }
        )
    if not checks["protocol"]:
        blockers.append(
            {"check": "protocol", "details": protocol["mismatches"]}
        )
    if not checks["dataset"]:
        blockers.append(
            {"check": "dataset", "details": dataset["mismatches"]}
        )
    if not checks["cmmloc_coarse_official_artifact"]:
        blockers.append(
            {
                "check": "cmmloc_coarse_official_artifact",
                "details": checkpoint["official_artifact_mismatches"],
            }
        )
    if not checks["cmmloc_coarse_source_architecture"]:
        blockers.append(
            {
                "check": "cmmloc_coarse_source_architecture",
                "details": (
                    dict(cmmloc_release_runtime_report or {})
                    if allow_public_release_behavior
                    else {
                        "unexpected_key_count": checkpoint.get(
                            "unexpected_key_count", 0
                        ),
                        "unexpected_prefix_counts": checkpoint.get(
                            "unexpected_prefix_counts", {}
                        ),
                        "source_reason": source.get("reason"),
                    }
                ),
            }
        )
    if not checks["cmmloc_text_pointnet_preprocessing"]:
        blockers.append(
            {
                "check": "cmmloc_text_pointnet_preprocessing",
                "details": (
                    dict(cmmloc_release_runtime_report or {})
                    if allow_public_release_behavior
                    else configuration
                ),
            }
        )
    if not checks["vlmloc_table8_fine_backend"]:
        blockers.append(
            {
                "check": "vlmloc_table8_fine_backend",
                "details": {
                    "missing_keys": vlmloc_report.get("missing_keys", []),
                    "unexpected_keys": vlmloc_report.get(
                        "unexpected_keys", []
                    ),
                    "shape_mismatches": vlmloc_report.get(
                        "shape_mismatches", []
                    ),
                    "reason": vlmloc_report.get("reason"),
                },
            }
        )

    return {
        "schema_version": 1,
        "experiment": (
            "CMMLoc baseline coarse Top-1 + VLM-Loc fine on KITTI360Pose "
            + (
                "paper test subset"
                if require_exact_paper_count
                else "full local test set"
            )
        ),
        "comparison_label": (
            "exact VLM-Loc Table-8 reproduction"
            if require_exact_paper_count
            else "Table-8-like full-test evaluation"
        ),
        "protocol": protocol,
        "dataset_audit": dataset,
        "cmmloc_coarse_checkpoint_audit": checkpoint,
        "cmmloc_public_source_audit": source,
        "cmmloc_runtime_configuration_audit": configuration,
        "cmmloc_release_runtime_audit": (
            dict(cmmloc_release_runtime_report)
            if cmmloc_release_runtime_report is not None
            else None
        ),
        "vlmloc_fine_audit": dict(vlmloc_report),
        "checks": checks,
        "all_compatible": all(checks.values()),
        "inference_authorized": all(checks.values()),
        "blockers": blockers,
        "warnings": warnings,
        "safety": {
            "my_model_used": False,
            "retrieval_top_k": 1,
            "strict_false_used_by_this_preflight": bool(
                cmmloc_release_compatible
            ),
            "strict_false_after_exact_mismatch_audit": bool(
                cmmloc_release_compatible
            ),
            "cross_architecture_checkpoint_load_attempted": False,
            "coarse_inference_attempted": False,
            "fine_inference_attempted": False,
        },
    }


def write_table8_preflight_report(
    path: Path, report: Mapping[str, Any]
) -> Path:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)
    return path
