"""Manifest format for CMMLoc Top-1 -> VLM-Loc full-test evaluation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from evaluation.coarse_to_fine_protocol import dataset_signature


PROTOCOL_NAME = (
    "table8-like/cmmloc-release-coarse-top1/"
    "vlmloc-kitti360pose-30m-fine/full-test"
)
PROTOCOL_VERSION = 1
SEED = 42
TOP_K = (1,)
THRESHOLDS_M = (5, 10, 15)
QUERY_COUNT = 11_505


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _rows_sha256(rows: Sequence[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(_canonical_json(row))
        digest.update(b"\n")
    return digest.hexdigest()


def _canonical_rows(
    dataset: Any, retrievals: Sequence[Sequence[str]]
) -> list[dict[str, Any]]:
    if len(dataset.all_poses) != QUERY_COUNT:
        raise RuntimeError(
            f"Table-8-like manifest requires {QUERY_COUNT} queries, got "
            f"{len(dataset.all_poses)}"
        )
    if len(retrievals) != len(dataset.all_poses):
        raise RuntimeError(
            "CMMLoc retrieval count does not match ordered test queries"
        )
    known_cells = {str(cell.id) for cell in dataset.all_cells}
    rows = []
    for query_index, (pose, candidates) in enumerate(
        zip(dataset.all_poses, retrievals)
    ):
        candidate_ids = [str(value) for value in candidates]
        if len(candidate_ids) != 1:
            raise RuntimeError(
                f"query {query_index} must contain exactly one CMMLoc cell"
            )
        if candidate_ids[0] not in known_cells:
            raise RuntimeError(
                f"query {query_index} contains unknown cell "
                f"{candidate_ids[0]}"
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


def write_table8_like_manifest(
    path: str | Path,
    *,
    dataset: Any,
    retrievals: Sequence[Sequence[str]],
    coarse_checkpoint: str | Path,
    coarse_checkpoint_sha256: str,
    coarse_audit_path: str | Path,
    coarse_configuration: dict[str, Any],
) -> dict[str, Any]:
    rows = _canonical_rows(dataset, retrievals)
    manifest = {
        "schema_version": PROTOCOL_VERSION,
        "protocol_name": PROTOCOL_NAME,
        "split": "test",
        "seed": SEED,
        "top_k": list(TOP_K),
        "thresholds_m": list(THRESHOLDS_M),
        "dataset": dataset_signature(dataset),
        "coarse_backend": "cmmloc_release_coarse",
        "coarse_checkpoint": str(Path(coarse_checkpoint).resolve()),
        "coarse_checkpoint_sha256": coarse_checkpoint_sha256,
        "coarse_audit_path": str(Path(coarse_audit_path).resolve()),
        "coarse_configuration": coarse_configuration,
        "retrieval_rows_sha256": _rows_sha256(rows),
        "rows": rows,
    }
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, destination)
    return manifest


def load_table8_like_manifest(
    path: str | Path, *, dataset: Any
) -> tuple[dict[str, Any], list[list[str]]]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "schema_version": PROTOCOL_VERSION,
        "protocol_name": PROTOCOL_NAME,
        "split": "test",
        "seed": SEED,
        "top_k": list(TOP_K),
        "thresholds_m": list(THRESHOLDS_M),
        "coarse_backend": "cmmloc_release_coarse",
    }
    for key, expected_value in expected.items():
        if manifest.get(key) != expected_value:
            raise RuntimeError(
                f"manifest {key!r} is {manifest.get(key)!r}, expected "
                f"{expected_value!r}"
            )
    if manifest.get("dataset") != dataset_signature(dataset):
        raise RuntimeError(
            "manifest dataset signature/order differs from current test data"
        )
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError("manifest rows are absent or invalid")
    if manifest.get("retrieval_rows_sha256") != _rows_sha256(rows):
        raise RuntimeError("manifest retrieval row checksum is invalid")
    retrievals = [row.get("retrieved_cell_ids", []) for row in rows]
    if rows != _canonical_rows(dataset, retrievals):
        raise RuntimeError("manifest rows do not match ordered dataset metadata")
    return manifest, retrievals
