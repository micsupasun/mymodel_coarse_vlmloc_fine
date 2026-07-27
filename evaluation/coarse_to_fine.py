"""Coarse-to-fine evaluation with explicit, fail-closed protocols.

The ``table8-preflight`` command audits CMMLoc Top-1 retrieval plus VLM-Loc
fine localization.  The separate ``stage1``/``stage2`` commands implement the
modified experiment: stage 1 runs only ``my_model`` coarse retrieval and stage
2 consumes that exact manifest without retrieving again.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch_geometric.transforms as T
import tqdm

from dataloading.kitti360pose.cells import (
    Kitti360CoarseDataset,
    Kitti360CoarseDatasetMulti,
)
from dataloading.kitti360pose.eval import Kitti360TopKDataset
from datapreparation.kitti360pose.utils import (
    COLOR_NAMES as COLOR_NAMES_K360,
    KNOWN_CLASS,
    SCENE_NAMES_TEST,
)
from evaluation.checkpoint_loading import audit_and_load_checkpoint
from evaluation.coarse_to_fine_protocol import (
    DEFAULT_SEED,
    REQUIRED_TEST_SCENES,
    REQUIRED_THRESHOLDS,
    REQUIRED_TOP_K,
    dataset_signature,
    load_and_validate_retrieval_manifest,
    seed_everything,
    sha256_file,
    validate_protocol_values,
    write_retrieval_manifest,
)
from evaluation.fine_backends import (
    BackendPreflight,
    FROZEN_TEXT_PREFIXES,
    T5_LARGE_CONFIG,
    preflight_cmmloc,
    preflight_mncl,
    preflight_vlmloc,
    t5_config_record,
    t5_config_mismatches,
    validate_sentence_preprocessing,
)
from evaluation.table8_reproduction import (
    CMMLOC_SOURCE_COMMIT,
    EXPECTED_THRESHOLDS_M as TABLE8_THRESHOLDS_M,
    EXPECTED_TOP_K as TABLE8_TOP_K,
    build_table8_preflight_report,
    write_table8_preflight_report,
)
from evaluation.pipeline import rerank_candidate_cells, run_coarse
from evaluation.utils import calc_sample_accuracies, print_accuracies
from models.coarse.cell_retrieval import CellRetrievalNetwork


class ModelConfig(dict):
    """Dictionary with attribute access, matching the source's mixed args usage."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _model_config(cli: argparse.Namespace) -> ModelConfig:
    rerank_mode = cli.coarse_rerank_mode
    if rerank_mode not in {
        "learned_reranker",
        "structured_rerank",
        "none",
    }:
        raise ValueError(f"unsupported coarse rerank mode: {rerank_mode}")

    return ModelConfig(
        base_path=str(Path(cli.data_root).resolve()),
        batch_size=cli.batch_size,
        top_k=list(cli.top_k),
        threshs=list(cli.thresholds),
        seed=cli.seed,
        dataset="K360",
        ranking_loss="pairwise",
        input_drop=0.2,
        drop=0.2,
        n_heads=4,
        sft_factor=0.6,
        use_features=["class", "color", "position", "num"],
        no_pc_augment=True,
        no_pc_augment_fine=True,
        coarse_embed_dim=256,
        pointnet_layers=3,
        pointnet_variation=0,
        pointnet_numpoints=256,
        pointnet_path="",
        allow_checkpoint_pointnet_init=True,
        pointnet_freeze=False,
        pointnet_features=2,
        class_embed=False,
        color_embed=False,
        object_size=28,
        object_inter_module_num_heads=4,
        object_inter_module_num_layers=2,
        hungging_model=cli.text_backbone,
        fixed_embedding=True,
        text_max_length=128,
        inter_module_num_heads=4,
        inter_module_num_layers=1,
        intra_module_num_heads=4,
        intra_module_num_layers=1,
        semantic_top_objects=12,
        mncl_proj_dim=256,
        use_trainable_reranker=True,
        reranker_hidden_dim=32,
        # Build the learned head so the checkpoint is audited exactly, but
        # reproduce the original ablation script's post-retrieval rerank path.
        trainable_rerank_topn=0,
        rerank_topn=0 if rerank_mode == "none" else 50,
        use_model_reranker=rerank_mode == "learned_reranker",
        rerank_base_weight=1.0,
        rerank_label_weight=0.6,
        rerank_color_weight=1.0,
        coarse_rerank_mode=rerank_mode,
        fine_embed_dim=128,
        fine_num_decoder_heads=4,
        fine_num_decoder_layers=2,
        fine_intra_module_num_heads=4,
        fine_intra_module_num_layers=1,
        pad_size=16,
        num_mentioned=6,
        describe_by="all",
        prealign_pointnet_path="",
        prealign_color_path="",
        prealign_mlp_path="",
    )


def _coarse_rerank_record(args: ModelConfig) -> dict[str, Any]:
    return {
        "coarse_rerank_mode": args.coarse_rerank_mode,
        "post_retrieval_rerank_topn": args.rerank_topn,
        "checkpoint_reranker_head_built": bool(args.use_trainable_reranker),
        "use_model_reranker": args.use_model_reranker,
        "structured_rerank_weights": {
            "base": args.rerank_base_weight,
            "label": args.rerank_label_weight,
            "color_label": args.rerank_color_weight,
        },
    }


def _coarse_rerank_console_text(args: ModelConfig) -> str:
    return (
        f"mode={args.coarse_rerank_mode}, topn={args.rerank_topn}, "
        f"use_model_reranker={args.use_model_reranker}, "
        f"weights=(base={args.rerank_base_weight}, "
        f"label={args.rerank_label_weight}, "
        f"color_label={args.rerank_color_weight})"
    )


def _manifest_coarse_rerank_record(
    manifest: dict[str, Any],
) -> dict[str, Any]:
    configuration = manifest.get("coarse_configuration")
    if not isinstance(configuration, dict):
        raise RuntimeError("manifest has no coarse_configuration record")
    required_keys = (
        "coarse_rerank_mode",
        "post_retrieval_rerank_topn",
        "checkpoint_reranker_head_built",
        "use_model_reranker",
        "structured_rerank_weights",
    )
    missing = [key for key in required_keys if key not in configuration]
    if missing:
        raise RuntimeError(
            "manifest predates explicit coarse-policy provenance; missing "
            f"{missing}. Regenerate Stage 1 with --coarse-rerank-mode."
        )
    return {key: configuration[key] for key in required_keys}


def _independent_structured_rerank(
    pose: Any,
    candidate_cell_ids: np.ndarray,
    candidate_scores: np.ndarray,
    cells_by_id: dict[str, Any],
    args: ModelConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference implementation used only by preflight policy validation."""

    required_labels = Counter(
        description.object_label for description in pose.descriptions
    )
    required_color_labels = Counter(
        (description.object_color_text, description.object_label)
        for description in pose.descriptions
    )

    def coverage(required: Counter[Any], available: Counter[Any]) -> float:
        total = sum(required.values())
        if total == 0:
            return 0.0
        return sum(
            min(count, available.get(key, 0))
            for key, count in required.items()
        ) / total

    combined_scores = []
    for cell_id, base_score in zip(candidate_cell_ids, candidate_scores):
        cell = cells_by_id[str(cell_id)]
        cell_labels = Counter(obj.label for obj in cell.objects)
        cell_color_labels = Counter(
            (obj.get_color_text(), obj.label) for obj in cell.objects
        )
        combined_scores.append(
            args.rerank_base_weight * float(base_score)
            + args.rerank_label_weight
            * coverage(required_labels, cell_labels)
            + args.rerank_color_weight
            * coverage(required_color_labels, cell_color_labels)
        )

    combined_scores_array = np.asarray(combined_scores, dtype=np.float32)
    order = np.argsort(-combined_scores_array)
    return candidate_cell_ids[order], combined_scores_array


def _architecture_record(args: ModelConfig, backend_class: str) -> dict[str, Any]:
    return {
        "backend_class": backend_class,
        "coarse_embed_dim": args.coarse_embed_dim,
        "fine_embed_dim": args.fine_embed_dim,
        "text_backbone_requested": args.hungging_model,
        "fixed_embedding": args.fixed_embedding,
        "text_max_length": args.text_max_length,
        "pointnet_layers": args.pointnet_layers,
        "pointnet_variation": args.pointnet_variation,
        "pointnet_numpoints": args.pointnet_numpoints,
        "pointnet_features": args.pointnet_features,
        "use_features": args.use_features,
        "point_transform": "FixedPoints(256), no NormalizeScale",
        "test_scene_order": list(REQUIRED_TEST_SCENES),
    }


def _load_test_dataset(args: ModelConfig) -> Kitti360CoarseDatasetMulti:
    if tuple(SCENE_NAMES_TEST) != REQUIRED_TEST_SCENES:
        raise RuntimeError(
            f"source test scenes changed: {tuple(SCENE_NAMES_TEST)} != "
            f"{REQUIRED_TEST_SCENES}"
        )
    transform = T.FixedPoints(args.pointnet_numpoints)
    return Kitti360CoarseDatasetMulti(
        args.base_path,
        list(REQUIRED_TEST_SCENES),
        transform,
        shuffle_hints=False,
        flip_poses=False,
        sample_close_cell=False,
    )


def _preflight_my_coarse(
    *, args: ModelConfig, checkpoint_path: Path, report_dir: Path
) -> BackendPreflight:
    report_path = (report_dir / "my_model_coarse_checkpoint_audit.json").resolve()
    try:
        model = CellRetrievalNetwork(KNOWN_CLASS, COLOR_NAMES_K360, args)
        language = model.language_encoder
        llm_config = language.llm_model.config
        sentence_preprocessing = validate_sentence_preprocessing()
        architecture = _architecture_record(
            args, "models.coarse.cell_retrieval.CellRetrievalNetwork"
        )
        architecture.update(
            {
                "text_backbone_config": t5_config_record(llm_config),
                "text_backbone_expected_config": T5_LARGE_CONFIG,
                "tokenizer_class": type(language.tokenizer).__name__,
                "tokenizer_vocab_size": len(language.tokenizer),
                "sentence_preprocessing": sentence_preprocessing,
                "use_trainable_reranker": True,
                "reranker_hidden_dim": args.reranker_hidden_dim,
                **_coarse_rerank_record(args),
            }
        )
        report = audit_and_load_checkpoint(
            model,
            checkpoint_path,
            report_path,
            backend="my_model_coarse",
            allowed_missing_prefixes=FROZEN_TEXT_PREFIXES,
            architecture=architecture,
        )
        config_mismatches = t5_config_mismatches(llm_config)
        if config_mismatches:
            report["compatible"] = False
            report["post_load_validation_succeeded"] = False
            report["text_backbone_config_mismatches"] = config_mismatches
            report["post_load_error"] = (
                f"my_model requires canonical T5-large config; mismatches: "
                f"{config_mismatches}"
            )
            report_path.write_text(
                json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
            )
            return BackendPreflight(
                "my_model_coarse", False, report_path, None
            )
        report["post_load_validation_succeeded"] = True
        report["text_backbone_config_mismatches"] = {}
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        return BackendPreflight("my_model_coarse", True, report_path, model)
    except Exception as error:
        if not report_path.exists():
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "backend": "my_model_coarse",
                        "compatible": False,
                        "load_attempted": False,
                        "load_succeeded": False,
                        "construction_error": f"{type(error).__name__}: {error}",
                        "missing_keys": [],
                        "unexpected_keys": [],
                        "shape_mismatches": [],
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        return BackendPreflight("my_model_coarse", False, report_path)


def _checkpoint_paths(cli: argparse.Namespace) -> dict[str, Path]:
    root = Path(cli.checkpoint_root).resolve()
    return {
        "my_model_coarse": (
            Path(cli.my_coarse_checkpoint).resolve()
            if cli.my_coarse_checkpoint
            else root / "my_model" / "coarse.pth"
        ),
        "cmmloc": (
            Path(cli.cmmloc_fine_checkpoint).resolve()
            if cli.cmmloc_fine_checkpoint
            else root / "CMMLoc" / "fine.pth"
        ),
        "cmmloc_coarse": (
            Path(cli.cmmloc_coarse_checkpoint).resolve()
            if cli.cmmloc_coarse_checkpoint
            else root / "CMMLoc" / "coarse.pth"
        ),
        "mncl": (
            Path(cli.mncl_fine_checkpoint).resolve()
            if cli.mncl_fine_checkpoint
            else root / "MNCL" / "fine.pth"
        ),
        "vlmloc": root / "VLM-Loc",
    }


def _validate_paths(paths: dict[str, Path], selected: Iterable[str]) -> None:
    for name in selected:
        path = paths[name]
        expected = path.is_dir() if name == "vlmloc" else path.is_file()
        if not expected:
            raise FileNotFoundError(f"{name} path does not exist: {path}")


def _write_preflight_summary(
    output_dir: Path,
    *,
    signature: dict[str, Any],
    checks: list[BackendPreflight],
    coarse_rerank_policy: dict[str, Any],
) -> Path:
    path = output_dir / "preflight_summary.json"
    payload = {
        "schema_version": 1,
        "dataset": signature,
        "protocol": {
            "split": "test",
            "seed": DEFAULT_SEED,
            "top_k": list(REQUIRED_TOP_K),
            "thresholds_m": list(REQUIRED_THRESHOLDS),
            "coarse_rerank_policy": coarse_rerank_policy,
        },
        "backends": {
            check.name: {
                "compatible": check.compatible,
                "report_path": str(check.report_path),
            }
            for check in checks
        },
        "all_compatible": all(check.compatible for check in checks),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _run_backend_preflights(
    *,
    names: list[str],
    args: ModelConfig,
    paths: dict[str, Path],
    report_dir: Path,
    signature: dict[str, Any],
) -> list[BackendPreflight]:
    checks = []
    for name in names:
        if name == "cmmloc":
            try:
                check = preflight_cmmloc(
                    args=args,
                    checkpoint_path=paths[name],
                    report_dir=report_dir,
                    known_classes=KNOWN_CLASS,
                    known_colors=COLOR_NAMES_K360,
                )
            except Exception as error:
                report_path = report_dir / "cmmloc_checkpoint_audit.json"
                _write_construction_error(report_path, name, error)
                check = BackendPreflight(name, False, report_path)
        elif name == "mncl":
            try:
                check = preflight_mncl(
                    checkpoint_path=paths[name], report_dir=report_dir
                )
            except Exception as error:
                report_path = report_dir / "mncl_checkpoint_audit.json"
                _write_construction_error(report_path, name, error)
                check = BackendPreflight(name, False, report_path)
        elif name == "vlmloc":
            try:
                check = preflight_vlmloc(
                    vlmloc_root=paths[name],
                    report_dir=report_dir,
                    current_dataset_signature=signature,
                )
            except Exception as error:
                report_path = report_dir / "vlmloc_checkpoint_audit.json"
                _write_construction_error(report_path, name, error)
                check = BackendPreflight(name, False, report_path)
        else:
            raise ValueError(f"unknown backend {name}")
        checks.append(check)
    return checks


def _write_construction_error(path: Path, backend: str, error: Exception) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend": backend,
                "compatible": False,
                "load_attempted": False,
                "load_succeeded": False,
                "construction_error": f"{type(error).__name__}: {error}",
                "missing_keys": [],
                "unexpected_keys": [],
                "shape_mismatches": [],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _record_smoke_result(
    check: BackendPreflight,
    *,
    result: dict[str, Any] | None = None,
    error: Exception | None = None,
) -> None:
    report = json.loads(check.report_path.read_text(encoding="utf-8"))
    if error is None:
        report["smoke_test"] = {"passed": True, **(result or {})}
    else:
        report["smoke_test"] = {
            "passed": False,
            "error": f"{type(error).__name__}: {error}",
        }
        report["compatible"] = False
        check.compatible = False
    check.report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )


def _print_coarse_policy_smoke(check: BackendPreflight) -> None:
    report = json.loads(check.report_path.read_text(encoding="utf-8"))
    smoke = report.get("smoke_test", {})
    if not smoke.get("passed"):
        print(
            "Coarse policy smoke: FAIL "
            f"({smoke.get('error', 'no smoke-test result')})"
        )
        return
    policy = smoke.get("rerank_policy")
    if not isinstance(policy, dict):
        print("Coarse policy smoke: FAIL (rerank_policy evidence missing)")
        return
    print(
        "Coarse policy smoke: PASS, "
        f"mode={policy['coarse_rerank_mode']}, "
        f"candidates={policy['candidate_count']}, "
        f"use_model_reranker={policy['use_model_reranker']}, "
        f"learned_head_exercised={policy['learned_head_exercised']}, "
        f"independent_reference_match="
        f"{policy['independent_reference_match']}, "
        f"changed_candidate_order={policy['changed_candidate_order']}"
    )


def _resolve_device(specification: str) -> torch.device:
    device = torch.device(specification)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


@torch.no_grad()
def _smoke_my_coarse(
    *,
    check: BackendPreflight,
    dataset: Kitti360CoarseDatasetMulti,
    args: ModelConfig,
    device: torch.device,
) -> None:
    if not check.compatible:
        return
    caller_rng_state = _capture_rng_state()
    try:
        seed_everything(args.seed)
        model = check.model.to(device)
        model.eval()
        batch = Kitti360CoarseDataset.collate_fn([dataset[0]])
        text_encoding = model.encode_text(batch["texts"])
        cell_encoding = model.encode_objects(
            batch["objects"], batch["object_points"]
        )
        expected_shape = (1, args.coarse_embed_dim)
        if tuple(text_encoding.shape) != expected_shape:
            raise RuntimeError(
                f"coarse text encoding shape {tuple(text_encoding.shape)}, "
                f"expected {expected_shape}"
            )
        if tuple(cell_encoding.shape) != expected_shape:
            raise RuntimeError(
                f"coarse cell encoding shape {tuple(cell_encoding.shape)}, "
                f"expected {expected_shape}"
            )
        if not torch.isfinite(text_encoding).all() or not torch.isfinite(
            cell_encoding
        ).all():
            raise RuntimeError("coarse smoke test produced NaN or Inf")

        cells_by_id = {str(cell.id): cell for cell in dataset.all_cells}
        candidate_count = max(max(args.top_k), args.rerank_topn)
        if candidate_count > len(dataset.all_cells):
            raise RuntimeError(
                f"coarse policy smoke requests {candidate_count} candidates "
                f"from only {len(dataset.all_cells)} cells"
            )
        candidate_ids = np.asarray(
            [str(cell.id) for cell in dataset.all_cells[:candidate_count]]
        )
        candidate_scores = np.linspace(
            1.0, 0.0, candidate_count, dtype=np.float64
        )

        policy_reference_match: bool | None = None
        learned_head_exercised = False
        structured_all_cell_score_range = None
        if args.coarse_rerank_mode == "structured_rerank":
            class LearnedHeadForbidden:
                reranker_head = object()

                @staticmethod
                def combine_rerank_scores(*_unused: Any) -> Any:
                    raise RuntimeError(
                        "structured_rerank unexpectedly called learned head"
                    )

            all_cell_ids = np.asarray(
                [str(cell.id) for cell in dataset.all_cells]
            )
            _, all_cell_structured_scores = _independent_structured_rerank(
                dataset.all_poses[0],
                all_cell_ids,
                np.zeros(len(all_cell_ids), dtype=np.float64),
                cells_by_id,
                args,
            )
            worst_index = int(np.argmin(all_cell_structured_scores))
            best_index = int(np.argmax(all_cell_structured_scores))
            score_min = float(all_cell_structured_scores[worst_index])
            score_max = float(all_cell_structured_scores[best_index])
            structured_all_cell_score_range = [score_min, score_max]
            if best_index == worst_index or score_max <= score_min:
                raise RuntimeError(
                    "structured policy smoke could not find real cells with "
                    "different fixed structured scores"
                )

            worst_id = str(all_cell_ids[worst_index])
            best_id = str(all_cell_ids[best_index])
            filler_ids = [
                str(cell_id)
                for cell_id in all_cell_ids
                if str(cell_id) not in {worst_id, best_id}
            ][: candidate_count - 2]
            candidate_ids = np.asarray(
                [worst_id, *filler_ids, best_id]
            )
            # Keep base-score differences negligible so the known low/high
            # structured-score cells guarantee a non-trivial reorder.
            candidate_scores = np.linspace(
                1e-6, 0.0, candidate_count, dtype=np.float64
            )
            expected_ids, reference_scores = _independent_structured_rerank(
                dataset.all_poses[0],
                candidate_ids,
                candidate_scores,
                cells_by_id,
                args,
            )
            reranked_ids = rerank_candidate_cells(
                dataset.all_poses[0],
                candidate_ids,
                candidate_scores,
                cells_by_id,
                args,
                LearnedHeadForbidden(),
            )
            policy_reference_match = bool(
                np.array_equal(reranked_ids, expected_ids)
            )
            if not policy_reference_match:
                raise RuntimeError(
                    "structured rerank output differs from independent "
                    "fixed-score reference"
                )
            if np.array_equal(reranked_ids, candidate_ids):
                raise RuntimeError(
                    "structured rerank matched the reference but did not "
                    "change the deliberately ordered real-cell candidates"
                )
            reference_score_range = [
                float(np.min(reference_scores)),
                float(np.max(reference_scores)),
            ]
        elif args.coarse_rerank_mode == "learned_reranker":
            reranked_ids = rerank_candidate_cells(
                dataset.all_poses[0],
                candidate_ids,
                candidate_scores,
                cells_by_id,
                args,
                model,
            )
            learned_head_exercised = True
            reference_score_range = None
        else:
            reranked_ids = rerank_candidate_cells(
                dataset.all_poses[0],
                candidate_ids,
                candidate_scores,
                cells_by_id,
                args,
                model,
            )
            if not np.array_equal(reranked_ids, candidate_ids):
                raise RuntimeError("coarse rerank mode none changed candidate order")
            policy_reference_match = True
            reference_score_range = None

        if len(reranked_ids) != candidate_count:
            raise RuntimeError(
                f"coarse policy smoke returned {len(reranked_ids)} candidates, "
                f"expected {candidate_count}"
            )
        if len(set(str(value) for value in reranked_ids)) != candidate_count:
            raise RuntimeError("coarse policy smoke returned duplicate candidates")

        _record_smoke_result(
            check,
            result={
                "device": str(device),
                "query_index": 0,
                "query_gt_cell_id": str(dataset.all_poses[0].cell_id),
                "text_encoding_shape": list(text_encoding.shape),
                "cell_encoding_shape": list(cell_encoding.shape),
                "similarity": float(
                    (cell_encoding @ text_encoding.transpose(0, 1))
                    .squeeze()
                    .detach()
                    .cpu()
                ),
                "rerank_policy": {
                    **_coarse_rerank_record(args),
                    "candidate_count": candidate_count,
                    "query_index": 0,
                    "input_first_10_cell_ids": [
                        str(value) for value in candidate_ids[:10]
                    ],
                    "output_first_10_cell_ids": [
                        str(value) for value in reranked_ids[:10]
                    ],
                    "changed_candidate_order": bool(
                        not np.array_equal(reranked_ids, candidate_ids)
                    ),
                    "independent_reference_match": policy_reference_match,
                    "learned_head_exercised": learned_head_exercised,
                    "independent_reference_score_range": (
                        reference_score_range
                    ),
                    "structured_all_cell_score_range": (
                        structured_all_cell_score_range
                    ),
                },
            },
        )
    except Exception as error:
        _record_smoke_result(check, error=error)
    finally:
        if check.model is not None:
            check.model.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _restore_rng_state(caller_rng_state)


@torch.no_grad()
def _smoke_fine_backend(
    *,
    check: BackendPreflight,
    dataset: Kitti360CoarseDatasetMulti,
    retrievals: list[list[str]],
    args: ModelConfig,
    device: torch.device,
) -> None:
    if not check.compatible:
        return
    caller_rng_state = _capture_rng_state()
    try:
        seed_everything(args.seed)
        model = check.model.to(device)
        model.eval()
        smoke_dataset = Kitti360TopKDataset(
            dataset.all_poses,
            dataset.all_cells,
            retrievals,
            T.FixedPoints(args.pointnet_numpoints),
            args,
        )
        sample = smoke_dataset[0]
        output = model(
            sample["objects"], sample["texts"], sample["object_points"]
        )
        expected_shape = (max(args.top_k), 2)
        if tuple(output.shape) != expected_shape:
            raise RuntimeError(
                f"fine output shape {tuple(output.shape)}, expected {expected_shape}"
            )
        if not torch.isfinite(output).all():
            raise RuntimeError("fine smoke test produced NaN or Inf")
        observed_ids = [str(cell.id) for cell in sample["cells"]]
        if observed_ids != retrievals[0]:
            raise RuntimeError("fine smoke test changed candidate cell ordering")
        _record_smoke_result(
            check,
            result={
                "device": str(device),
                "query_index": 0,
                "retrieved_cell_ids": observed_ids,
                "output_shape": list(output.shape),
                "finite": True,
            },
        )
    except Exception as error:
        _record_smoke_result(check, error=error)
    finally:
        if check.model is not None:
            check.model.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _restore_rng_state(caller_rng_state)


def _native_metrics(
    metrics: dict[int, dict[int, Any]]
) -> dict[int, dict[int, float]]:
    return {
        int(k): {
            int(threshold): float(value)
            for threshold, value in threshold_values.items()
        }
        for k, threshold_values in metrics.items()
    }


def _exact_cell_retrieval_recall(
    retrievals: list[list[str]],
    dataset: Kitti360CoarseDatasetMulti,
    top_k: list[int],
) -> dict[int, float]:
    if len(retrievals) != len(dataset.all_poses):
        raise RuntimeError(
            f"retrieval count {len(retrievals)} != query count "
            f"{len(dataset.all_poses)}"
        )
    return {
        int(k): float(
            np.mean(
                [
                    str(pose.cell_id) in candidate_ids[:k]
                    for pose, candidate_ids in zip(
                        dataset.all_poses, retrievals
                    )
                ]
            )
        )
        for k in top_k
    }


@torch.no_grad()
def _run_fine(
    *,
    model: torch.nn.Module,
    retrievals: list[list[str]],
    dataset: Kitti360CoarseDatasetMulti,
    args: ModelConfig,
    backend_output_dir: Path,
) -> dict[int, dict[int, float]]:
    model.eval()
    transform = T.FixedPoints(args.pointnet_numpoints)
    dataset_topk = Kitti360TopKDataset(
        dataset.all_poses,
        dataset.all_cells,
        retrievals,
        transform,
        args,
    )
    cells_by_id = {cell.id: cell for cell in dataset.all_cells}
    offsets = []
    observed_cell_ids = []
    observed_poses = []
    started = time.time()
    for sample in tqdm.tqdm(dataset_topk, total=len(dataset_topk)):
        output = model(
            sample["objects"], sample["texts"], sample["object_points"]
        )
        output_array = output.detach().cpu().numpy().astype(np.float32)
        if output_array.shape != (max(args.top_k), 2):
            raise RuntimeError(
                f"fine output shape {output_array.shape}; "
                f"expected {(max(args.top_k), 2)}"
            )
        offsets.append(output_array)
        observed_cell_ids.append([cell.id for cell in sample["cells"]])
        observed_poses.append(np.asarray(sample["poses"][0].pose_w))

    offsets_array = np.stack(offsets)
    observed_cell_ids_array = np.asarray(observed_cell_ids)
    if not np.array_equal(observed_cell_ids_array, np.asarray(retrievals)):
        raise RuntimeError("fine dataset changed retrieval cell ordering")

    accuracies = {
        k: {threshold: [] for threshold in args.threshs} for k in args.top_k
    }
    for index, pose in enumerate(dataset.all_poses):
        if not np.allclose(pose.pose_w, observed_poses[index]):
            raise RuntimeError(f"fine dataset changed pose ordering at query {index}")
        top_cells = [cells_by_id[cell_id] for cell_id in retrievals[index]]
        sample = calc_sample_accuracies(
            pose,
            top_cells,
            offsets_array[index],
            args.top_k,
            args.threshs,
        )
        for k in args.top_k:
            for threshold in args.threshs:
                accuracies[k][threshold].append(sample[k][threshold])

    metrics = {
        k: {
            threshold: float(np.mean(accuracies[k][threshold]))
            for threshold in args.threshs
        }
        for k in args.top_k
    }
    backend_output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        backend_output_dir / "predictions.npz",
        normalized_offsets=offsets_array,
        retrieved_cell_ids=observed_cell_ids_array,
        pose_w=np.asarray(observed_poses),
    )
    (backend_output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "query_count": len(dataset.all_poses),
                "top_k": list(args.top_k),
                "thresholds_m": list(args.threshs),
                "elapsed_seconds": time.time() - started,
                "metrics": metrics,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return metrics


def command_preflight(cli: argparse.Namespace) -> int:
    validate_protocol_values(cli.top_k, cli.thresholds, cli.seed)
    output_dir = Path(cli.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args = _model_config(cli)
    paths = _checkpoint_paths(cli)
    selected = ["my_model_coarse", *cli.backends]
    _validate_paths(paths, selected)
    seed_everything(cli.seed)
    dataset = _load_test_dataset(args)
    signature = dataset_signature(dataset)
    device = _resolve_device(cli.device)
    print(f"Coarse policy preflight: {_coarse_rerank_console_text(args)}")

    checks = [
        _preflight_my_coarse(
            args=args,
            checkpoint_path=paths["my_model_coarse"],
            report_dir=output_dir,
        )
    ]
    _smoke_my_coarse(
        check=checks[0], dataset=dataset, args=args, device=device
    )
    _print_coarse_policy_smoke(checks[0])
    if checks[0].model is not None:
        del checks[0].model
        checks[0].model = None
    checks.extend(
        _run_backend_preflights(
            names=cli.backends,
            args=args,
            paths=paths,
            report_dir=output_dir,
            signature=signature,
        )
    )
    smoke_ids = [str(cell.id) for cell in dataset.all_cells[: max(args.top_k)]]
    smoke_retrievals = [smoke_ids for _ in dataset.all_poses]
    for check in checks[1:]:
        _smoke_fine_backend(
            check=check,
            dataset=dataset,
            retrievals=smoke_retrievals,
            args=args,
            device=device,
        )
    for check in checks:
        if check.model is not None:
            del check.model
            check.model = None
    summary = _write_preflight_summary(
        output_dir,
        signature=signature,
        checks=checks,
        coarse_rerank_policy=_coarse_rerank_record(args),
    )
    print(f"Preflight report: {summary}")
    for check in checks:
        print(f"{check.name}: {'PASS' if check.compatible else 'FAIL'} ({check.report_path})")
    return 0 if all(check.compatible for check in checks) else 2


def command_table8_preflight(cli: argparse.Namespace) -> int:
    """Audit only CMMLoc Top-1 -> VLM-Loc fine Table-8 reproduction."""

    output_dir = Path(cli.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args = _model_config(cli)
    paths = _checkpoint_paths(cli)
    seed_everything(cli.seed)
    dataset = _load_test_dataset(args)
    signature = dataset_signature(dataset)

    vlmloc_dir = output_dir / "vlmloc_fine"
    try:
        vlmloc_check = preflight_vlmloc(
            vlmloc_root=paths["vlmloc"],
            report_dir=vlmloc_dir,
            current_dataset_signature=signature,
        )
    except Exception as error:
        vlmloc_report_path = vlmloc_dir / "vlmloc_checkpoint_audit.json"
        _write_construction_error(
            vlmloc_report_path, "vlmloc_table8_fine", error
        )
        vlmloc_check = BackendPreflight(
            "vlmloc_table8_fine", False, vlmloc_report_path
        )
    vlmloc_report = json.loads(
        vlmloc_check.report_path.read_text(encoding="utf-8")
    )

    checkpoint_root = Path(cli.checkpoint_root).resolve()
    cmmloc_source_root = (
        Path(cli.cmmloc_source_root).resolve()
        if cli.cmmloc_source_root
        else checkpoint_root
        / "CMMLoc"
        / "official_source"
        / f"CMMLoc-{CMMLOC_SOURCE_COMMIT[:12]}"
    )
    report = build_table8_preflight_report(
        data_root=Path(cli.data_root),
        dataset_signature=signature,
        cmmloc_coarse_checkpoint=paths["cmmloc_coarse"],
        cmmloc_source_root=cmmloc_source_root,
        vlmloc_report=vlmloc_report,
        requested_text_backbone=cli.text_backbone,
        split="test",
        seed=cli.seed,
        top_k=cli.top_k,
        thresholds_m=cli.thresholds,
    )
    report_path = write_table8_preflight_report(
        output_dir / "table8_step1_preflight.json", report
    )
    print("Table-8 step 1 protocol: CMMLoc coarse Top-1 -> VLM-Loc fine")
    print(f"Dataset queries: {signature['query_count']} (Table 8: 11404)")
    checkpoint_audit = report["cmmloc_coarse_checkpoint_audit"]
    print(
        "CMMLoc coarse official artifact: "
        f"{'PASS' if checkpoint_audit.get('compatible_with_official_artifact') else 'FAIL'}"
    )
    print(
        "CMMLoc coarse checkpoint/source architecture: "
        f"{'PASS' if report['checks']['cmmloc_coarse_source_architecture'] else 'FAIL'}"
    )
    print(
        "VLM-Loc Table-8 fine backend: "
        f"{'PASS' if report['checks']['vlmloc_table8_fine_backend'] else 'FAIL'}"
    )
    print(f"Table-8 step 1 preflight: {report_path}")
    if not report["all_compatible"]:
        print(
            "Inference was not started because the exact Table-8 protocol is "
            "not fully reproducible from the audited artifacts.",
            file=sys.stderr,
        )
        return 2
    return 0


def command_stage1(cli: argparse.Namespace) -> int:
    validate_protocol_values(cli.top_k, cli.thresholds, cli.seed)
    output_dir = Path(cli.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args = _model_config(cli)
    paths = _checkpoint_paths(cli)
    _validate_paths(paths, ["my_model_coarse"])
    seed_everything(cli.seed)
    dataset = _load_test_dataset(args)
    signature = dataset_signature(dataset)
    coarse_check = _preflight_my_coarse(
        args=args,
        checkpoint_path=paths["my_model_coarse"],
        report_dir=output_dir / "preflight",
    )
    device = _resolve_device(cli.device)
    _smoke_my_coarse(
        check=coarse_check, dataset=dataset, args=args, device=device
    )
    _print_coarse_policy_smoke(coarse_check)
    _write_preflight_summary(
        output_dir / "preflight",
        signature=signature,
        checks=[coarse_check],
        coarse_rerank_policy=_coarse_rerank_record(args),
    )
    if not coarse_check.compatible:
        print(f"Coarse preflight failed: {coarse_check.report_path}", file=sys.stderr)
        return 2

    model = coarse_check.model.to(device)
    print(f"Coarse rerank policy: {_coarse_rerank_console_text(args)}")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=Kitti360CoarseDataset.collate_fn,
        shuffle=False,
        num_workers=0,
    )
    retrievals, metrics = run_coarse(model, dataloader, args)
    retrieval_recall = _exact_cell_retrieval_recall(
        retrievals, dataset, args.top_k
    )
    print(f"my_model exact-cell Retrieval Recall@K: {retrieval_recall}")
    print_accuracies(
        metrics,
        "Cell-center localization baseline (not exact-cell retrieval)",
    )
    native_metrics = _native_metrics(metrics)
    checkpoint_hash = sha256_file(paths["my_model_coarse"])
    manifest_path = output_dir / "retrieval_manifest.json"
    write_retrieval_manifest(
        manifest_path,
        dataset=dataset,
        retrievals=retrievals,
        coarse_checkpoint=paths["my_model_coarse"],
        coarse_checkpoint_sha256=checkpoint_hash,
        seed=cli.seed,
        coarse_configuration={
            **_architecture_record(
                args, "models.coarse.cell_retrieval.CellRetrievalNetwork"
            ),
            **_coarse_rerank_record(args),
            "batch_size": args.batch_size,
            "num_workers": 0,
            "rng_protocol": (
                "seed=42 once before dataset/model construction; preflight "
                "smoke saves and restores caller RNG state"
            ),
        },
    )
    (output_dir / "coarse_metrics.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": signature,
                "checkpoint_sha256": checkpoint_hash,
                "coarse_rerank_policy": _coarse_rerank_record(args),
                "metric_definitions": {
                    "exact_cell_retrieval_recall": (
                        "GT cell ID occurs in the first K retrieved cell IDs"
                    ),
                    "cell_center_localization_recall": (
                        "world-distance R@5/10/15m when every retrieved cell "
                        "predicts its normalized center [0.5, 0.5]; this is a "
                        "diagnostic baseline, not retrieval recall"
                    ),
                },
                "exact_cell_retrieval_recall": retrieval_recall,
                "cell_center_localization_recall": native_metrics,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"Stage-1 manifest: {manifest_path}")
    return 0


def command_stage2(cli: argparse.Namespace) -> int:
    validate_protocol_values(cli.top_k, cli.thresholds, cli.seed)
    output_dir = Path(cli.output_dir).resolve()
    manifest_path = Path(cli.manifest).resolve()
    if output_dir == manifest_path.parent:
        raise ValueError("stage-2 output directory must differ from stage-1")
    output_dir.mkdir(parents=True, exist_ok=True)
    args = _model_config(cli)
    paths = _checkpoint_paths(cli)
    _validate_paths(paths, cli.backends)
    seed_everything(cli.seed)
    dataset = _load_test_dataset(args)
    signature = dataset_signature(dataset)
    manifest, retrievals = load_and_validate_retrieval_manifest(
        manifest_path, dataset=dataset, seed=cli.seed
    )
    manifest_coarse_policy = _manifest_coarse_rerank_record(manifest)
    print(
        "Shared manifest coarse rerank policy: "
        f"{json.dumps(manifest_coarse_policy, sort_keys=True)}"
    )
    coarse_retrieval_recall = _exact_cell_retrieval_recall(
        retrievals, dataset, args.top_k
    )
    print(
        "Shared manifest exact-cell Retrieval Recall@K: "
        f"{coarse_retrieval_recall}"
    )

    preflight_dir = output_dir / "preflight"
    checks = _run_backend_preflights(
        names=cli.backends,
        args=args,
        paths=paths,
        report_dir=preflight_dir,
        signature=signature,
    )
    device = _resolve_device(cli.device)
    for check in checks:
        _smoke_fine_backend(
            check=check,
            dataset=dataset,
            retrievals=retrievals,
            args=args,
            device=device,
        )
    summary = _write_preflight_summary(
        preflight_dir,
        signature=signature,
        checks=checks,
        coarse_rerank_policy=manifest_coarse_policy,
    )
    if not all(check.compatible for check in checks):
        print(
            f"Stage-2 aborted before inference; incompatible backend(s). See {summary}",
            file=sys.stderr,
        )
        return 2

    combined = {
        "schema_version": 1,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "retrieval_rows_sha256": manifest["retrieval_rows_sha256"],
        "dataset": signature,
        "seed": cli.seed,
        "coarse_rerank_policy": manifest_coarse_policy,
        "coarse_exact_cell_retrieval_recall": coarse_retrieval_recall,
        "metrics": {},
    }
    for check in checks:
        # Reset every RNG before each backend so FixedPoints selects identical
        # point subsets for the identical ordered query/cell stream.
        seed_everything(cli.seed)
        model = check.model.to(device)
        metrics = _run_fine(
            model=model,
            retrievals=retrievals,
            dataset=dataset,
            args=args,
            backend_output_dir=output_dir / check.name,
        )
        print_accuracies(metrics, check.name)
        combined["metrics"][check.name] = metrics
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result_path = output_dir / "comparison_metrics.json"
    result_path.write_text(
        json.dumps(combined, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Stage-2 comparison: {result_path}")
    return 0


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        default=r"data\k360_30-10_scG_pd10_pc4_spY_all",
    )
    parser.add_argument(
        "--checkpoint-root",
        default=r"checkpoints\k360_30-10_scG_pd10_pc4_spY_all",
    )
    parser.add_argument(
        "--text-backbone",
        default="t5-large",
        help="Exact local directory or Hugging Face identifier for the T5-large backbone.",
    )
    parser.add_argument("--my-coarse-checkpoint", default="")
    parser.add_argument("--cmmloc-coarse-checkpoint", default="")
    parser.add_argument("--cmmloc-fine-checkpoint", default="")
    parser.add_argument("--mncl-fine-checkpoint", default="")
    parser.add_argument(
        "--cmmloc-source-root",
        default="",
        help=(
            "Pinned official CMMLoc source root. If omitted, use "
            "CHECKPOINT_ROOT/CMMLoc/official_source/CMMLoc-d49458963d4c."
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--top-k", type=int, nargs="+", default=list(REQUIRED_TOP_K)
    )
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=list(REQUIRED_THRESHOLDS),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--coarse-rerank-mode",
        choices=("learned_reranker", "structured_rerank", "none"),
        default="learned_reranker",
        help=(
            "learned_reranker reproduces ablation v1.1; structured_rerank "
            "reproduces v1.4 with fixed base/label/color-label weights; none "
            "uses the base retrieval ranking."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fair shared-retrieval coarse-to-fine evaluation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    _add_common_arguments(preflight)
    preflight.add_argument("--output-dir", required=True)
    preflight.add_argument(
        "--backends",
        nargs="+",
        choices=("vlmloc", "cmmloc", "mncl"),
        default=["vlmloc", "cmmloc", "mncl"],
    )
    preflight.set_defaults(handler=command_preflight)

    table8_preflight = subparsers.add_parser(
        "table8-preflight",
        help=(
            "Fail-closed audit for CMMLoc coarse Top-1 + VLM-Loc fine "
            "(VLM-Loc Table 8)."
        ),
    )
    _add_common_arguments(table8_preflight)
    table8_preflight.set_defaults(
        handler=command_table8_preflight,
        top_k=list(TABLE8_TOP_K),
        thresholds=list(TABLE8_THRESHOLDS_M),
        coarse_rerank_mode="none",
    )
    table8_preflight.add_argument("--output-dir", required=True)

    stage1 = subparsers.add_parser("stage1")
    _add_common_arguments(stage1)
    stage1.add_argument("--output-dir", required=True)
    stage1.set_defaults(handler=command_stage1)

    stage2 = subparsers.add_parser("stage2")
    _add_common_arguments(stage2)
    stage2.add_argument("--manifest", required=True)
    stage2.add_argument("--output-dir", required=True)
    stage2.add_argument(
        "--backends",
        nargs="+",
        choices=("vlmloc", "cmmloc", "mncl"),
        default=["vlmloc", "cmmloc", "mncl"],
    )
    stage2.set_defaults(handler=command_stage2)
    return parser


def main() -> int:
    cli = build_parser().parse_args()
    return int(cli.handler(cli))


if __name__ == "__main__":
    raise SystemExit(main())
