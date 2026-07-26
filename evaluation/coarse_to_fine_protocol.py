"""Shared, deterministic protocol and retrieval-manifest handling."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


PROTOCOL_NAME = "my-model-coarse/shared-fine-kitti360pose-test"
PROTOCOL_VERSION = 1
DEFAULT_SEED = 42
REQUIRED_TOP_K = (1, 3, 5, 10)
REQUIRED_THRESHOLDS = (5, 10, 15)
REQUIRED_TEST_SCENES = (
    "2013_05_28_drive_0003_sync",
    "2013_05_28_drive_0005_sync",
    "2013_05_28_drive_0009_sync",
)


def validate_protocol_values(
    top_k: Sequence[int], thresholds: Sequence[int], seed: int
) -> None:
    if tuple(top_k) != REQUIRED_TOP_K:
        raise ValueError(f"top-k must be exactly {REQUIRED_TOP_K}, got {tuple(top_k)}")
    if tuple(thresholds) != REQUIRED_THRESHOLDS:
        raise ValueError(
            f"thresholds must be exactly {REQUIRED_THRESHOLDS}, got {tuple(thresholds)}"
        )
    if seed != DEFAULT_SEED:
        raise ValueError(f"seed must be exactly {DEFAULT_SEED}, got {seed}")


def seed_everything(seed: int) -> None:
    """Set deterministic RNG state; torch is imported only when available."""

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _float_tokens(values: Iterable[Any]) -> list[str]:
    return [float(value).hex() for value in values]


def _description_record(description: Any) -> dict[str, Any]:
    return {
        "direction": str(description.direction),
        "object_color_text": str(description.object_color_text),
        "object_label": str(description.object_label),
        "object_id": str(getattr(description, "object_id", "")),
        "is_matched": bool(getattr(description, "is_matched", False)),
    }


def _update_hash(digest: Any, value: Any) -> None:
    digest.update(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    )
    digest.update(b"\n")


def _update_array_hash(digest: Any, value: Any) -> None:
    array = np.ascontiguousarray(value)
    _update_hash(
        digest,
        {"dtype": array.dtype.str, "shape": list(array.shape)},
    )
    digest.update(array.view(np.uint8).tobytes())
    digest.update(b"\n")


def dataset_signature(dataset: Any) -> dict[str, Any]:
    """Fingerprint ordered test queries and cells, including their text inputs."""

    query_digest = hashlib.sha256()
    cell_digest = hashlib.sha256()
    scene_query_counts: dict[str, int] = {}
    scene_cell_counts: dict[str, int] = {}
    description_count_histogram: dict[str, int] = {}

    for index, pose in enumerate(dataset.all_poses):
        scene = str(pose.scene_name)
        scene_query_counts[scene] = scene_query_counts.get(scene, 0) + 1
        descriptions = [_description_record(item) for item in pose.descriptions]
        description_count = str(len(descriptions))
        description_count_histogram[description_count] = (
            description_count_histogram.get(description_count, 0) + 1
        )
        query_text = " ".join(
            f"The pose is {item['direction']} of a "
            f"{item['object_color_text']} {item['object_label']}."
            for item in descriptions
        )
        _update_hash(
            query_digest,
            {
                "index": index,
                "scene": scene,
                "cell_id": str(pose.cell_id),
                "pose_w": _float_tokens(pose.pose_w),
                "query_text": query_text,
                "descriptions": descriptions,
            },
        )

    for index, cell in enumerate(dataset.all_cells):
        scene = str(cell.scene_name)
        scene_cell_counts[scene] = scene_cell_counts.get(scene, 0) + 1
        _update_hash(
            cell_digest,
            {
                "index": index,
                "scene": scene,
                "cell_id": str(cell.id),
                "bbox_w": _float_tokens(cell.bbox_w),
                "cell_size": float(cell.cell_size).hex(),
                "object_count": len(cell.objects),
            },
        )
        for object_index, object_3d in enumerate(cell.objects):
            _update_hash(
                cell_digest,
                {
                    "object_index": object_index,
                    "object_id": str(object_3d.id),
                    "instance_id": str(object_3d.instance_id),
                    "label": str(object_3d.label),
                },
            )
            _update_array_hash(cell_digest, object_3d.xyz)
            _update_array_hash(cell_digest, object_3d.rgb)

    ordered_scenes = tuple(str(scene) for scene in dataset.scene_names)
    if ordered_scenes != REQUIRED_TEST_SCENES:
        raise RuntimeError(
            f"dataset scene ordering is {ordered_scenes}, expected {REQUIRED_TEST_SCENES}"
        )

    return {
        "ordered_scenes": list(ordered_scenes),
        "query_count": len(dataset.all_poses),
        "cell_count": len(dataset.all_cells),
        "scene_query_counts": scene_query_counts,
        "scene_cell_counts": scene_cell_counts,
        "description_count_histogram": description_count_histogram,
        "ordered_query_sha256": query_digest.hexdigest(),
        "ordered_cell_sha256": cell_digest.hexdigest(),
    }


def _retrieval_rows(
    dataset: Any, retrievals: Sequence[Sequence[str]]
) -> list[dict[str, Any]]:
    max_k = max(REQUIRED_TOP_K)
    if len(retrievals) != len(dataset.all_poses):
        raise RuntimeError(
            f"retrieval query count {len(retrievals)} does not match "
            f"dataset query count {len(dataset.all_poses)}"
        )

    known_cell_ids = {str(cell.id) for cell in dataset.all_cells}
    rows = []
    for query_index, (pose, candidates) in enumerate(
        zip(dataset.all_poses, retrievals)
    ):
        candidate_ids = [str(value) for value in candidates]
        if len(candidate_ids) != max_k:
            raise RuntimeError(
                f"query {query_index} has {len(candidate_ids)} candidates; expected {max_k}"
            )
        if len(set(candidate_ids)) != len(candidate_ids):
            raise RuntimeError(f"query {query_index} contains duplicate candidate cells")
        unknown = sorted(set(candidate_ids) - known_cell_ids)
        if unknown:
            raise RuntimeError(
                f"query {query_index} contains unknown candidate cells: {unknown}"
            )
        rows.append(
            {
                "query_index": query_index,
                "query_scene": str(pose.scene_name),
                "query_gt_cell_id": str(pose.cell_id),
                "retrieved_cell_ids": candidate_ids,
            }
        )
    return rows


def _rows_sha256(rows: Sequence[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        _update_hash(digest, row)
    return digest.hexdigest()


def write_retrieval_manifest(
    path: str | Path,
    *,
    dataset: Any,
    retrievals: Sequence[Sequence[str]],
    coarse_checkpoint: str | Path,
    coarse_checkpoint_sha256: str,
    seed: int,
    coarse_configuration: dict[str, Any],
) -> dict[str, Any]:
    validate_protocol_values(REQUIRED_TOP_K, REQUIRED_THRESHOLDS, seed)
    rows = _retrieval_rows(dataset, retrievals)
    manifest = {
        "schema_version": PROTOCOL_VERSION,
        "protocol_name": PROTOCOL_NAME,
        "split": "test",
        "seed": seed,
        "top_k": list(REQUIRED_TOP_K),
        "thresholds_m": list(REQUIRED_THRESHOLDS),
        "dataset": dataset_signature(dataset),
        "coarse_checkpoint": str(Path(coarse_checkpoint).resolve()),
        "coarse_checkpoint_sha256": coarse_checkpoint_sha256,
        "coarse_configuration": coarse_configuration,
        "retrieval_rows_sha256": _rows_sha256(rows),
        "rows": rows,
    }

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)
    return manifest


def load_and_validate_retrieval_manifest(
    path: str | Path, *, dataset: Any, seed: int
) -> tuple[dict[str, Any], list[list[str]]]:
    validate_protocol_values(REQUIRED_TOP_K, REQUIRED_THRESHOLDS, seed)
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))

    expected_header = {
        "schema_version": PROTOCOL_VERSION,
        "protocol_name": PROTOCOL_NAME,
        "split": "test",
        "seed": seed,
        "top_k": list(REQUIRED_TOP_K),
        "thresholds_m": list(REQUIRED_THRESHOLDS),
    }
    for key, expected in expected_header.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"manifest {key!r} is {manifest.get(key)!r}, expected {expected!r}"
            )

    current_signature = dataset_signature(dataset)
    if manifest.get("dataset") != current_signature:
        raise RuntimeError(
            "manifest dataset signature/order does not match the current test dataset"
        )
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError("manifest rows are missing or are not a list")
    if manifest.get("retrieval_rows_sha256") != _rows_sha256(rows):
        raise RuntimeError("manifest retrieval rows checksum is invalid")

    retrievals = [row["retrieved_cell_ids"] for row in rows]
    canonical_rows = _retrieval_rows(dataset, retrievals)
    if rows != canonical_rows:
        raise RuntimeError("manifest query metadata/order is inconsistent with the dataset")
    return manifest, retrievals


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
