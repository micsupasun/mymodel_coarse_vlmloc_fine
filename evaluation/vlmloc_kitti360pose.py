"""Prepare and score the Table-8-like KITTI360Pose VLM-Loc fine stage.

The public VLM-Loc release only provides CityLoc 50 m data/adapters.  This
module creates a separate 30 m training/validation/test dataset from the
ordered local KITTI360Pose pickles.  Test examples use the exact CMMLoc Top-1
candidate stored in the checksummed Table-8-like manifest.

The local processed cells do not contain the dense ``xyz_raw``/``rgb_raw``
fields used by the public CityLoc renderer.  Images generated here therefore
use the cell's downsampled normalized points and are labeled accordingly in
the preparation audit.  They must be used with an adapter retrained on these
same images; a public CityLoc adapter is never substituted.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from evaluation.coarse_to_fine_protocol import dataset_signature
from evaluation.table8_like_protocol import (
    QUERY_COUNT,
    load_table8_like_manifest,
)
from evaluation.vlmloc_release import (
    QWEN3_VL_8B_ARCHITECTURE,
    inspect_adapter,
    inspect_base_model,
    inspect_official_source,
    inspect_python_environment,
)


IMAGE_SIZE = 224
BEV_RANGE_M = 30.0
METERS_PER_PIXEL = BEV_RANGE_M / IMAGE_SIZE
RENDERER_NAME = "processed_cell_downsampled_points_v1"
SCENE_NAMES_TRAIN = (
    "2013_05_28_drive_0000_sync",
    "2013_05_28_drive_0002_sync",
    "2013_05_28_drive_0004_sync",
    "2013_05_28_drive_0006_sync",
    "2013_05_28_drive_0007_sync",
)
SCENE_NAMES_VAL = ("2013_05_28_drive_0010_sync",)
SCENE_NAMES_TEST = (
    "2013_05_28_drive_0003_sync",
    "2013_05_28_drive_0005_sync",
    "2013_05_28_drive_0009_sync",
)
STUFF_CLASSES = frozenset(
    {
        "sidewalk",
        "road",
        "parking",
        "wall",
        "fence",
        "guard rail",
        "bridge",
        "tunnel",
        "vegetation",
        "terrain",
    }
)


@dataclass
class OrderedDataset:
    scene_names: list[str]
    all_poses: list[Any]
    all_cells: list[Any]


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_pickle(path: Path) -> Any:
    # Import lazily so geometry/projection unit tests do not require OpenCV.
    from dataloading.kitti360pose.compat import (
        load_kitti360pose_pickle,
    )

    return load_kitti360pose_pickle(path)


def load_ordered_dataset(
    data_root: Path, scene_names: Sequence[str]
) -> OrderedDataset:
    data_root = Path(data_root).resolve()
    poses: list[Any] = []
    cells: list[Any] = []
    for scene in scene_names:
        pose_path = data_root / "poses" / f"{scene}.pkl"
        cell_path = data_root / "cells" / f"{scene}.pkl"
        if not pose_path.is_file() or not cell_path.is_file():
            raise FileNotFoundError(
                f"Missing ordered KITTI360Pose files for {scene}: "
                f"{pose_path}, {cell_path}"
            )
        scene_poses = list(_load_pickle(pose_path))
        scene_cells = list(_load_pickle(cell_path))
        if any(str(pose.scene_name) != scene for pose in scene_poses):
            raise RuntimeError(f"pose scene metadata mismatch in {pose_path}")
        if any(str(cell.scene_name) != scene for cell in scene_cells):
            raise RuntimeError(f"cell scene metadata mismatch in {cell_path}")
        poses.extend(scene_poses)
        cells.extend(scene_cells)
    cell_ids = [str(cell.id) for cell in cells]
    if len(cell_ids) != len(set(cell_ids)):
        raise RuntimeError("ordered dataset contains duplicate cell IDs")
    return OrderedDataset(list(scene_names), poses, cells)


def normalized_xy_to_pixel(
    xy: np.ndarray | Sequence[float],
    *,
    image_size: int = IMAGE_SIZE,
    clip: bool = True,
) -> tuple[int, int]:
    values = np.asarray(xy, dtype=np.float64).reshape(2)
    eps = 1e-9
    x = int(math.floor(values[0] * image_size - eps))
    y_unflipped = int(math.floor(values[1] * image_size - eps))
    if clip:
        x = min(max(x, 0), image_size - 1)
        y_unflipped = min(max(y_unflipped, 0), image_size - 1)
    elif not (
        0 <= x < image_size and 0 <= y_unflipped < image_size
    ):
        raise ValueError(f"normalized point is outside the cell: {values}")
    return x, (image_size - 1) - y_unflipped


def world_xy_to_pixel(
    world_xy: np.ndarray | Sequence[float],
    bbox_w: np.ndarray | Sequence[float],
    *,
    image_size: int = IMAGE_SIZE,
    clip: bool = True,
) -> tuple[tuple[int, int], bool]:
    bbox = np.asarray(bbox_w, dtype=np.float64)
    xy = np.asarray(world_xy, dtype=np.float64).reshape(2)
    spans = bbox[3:5] - bbox[0:2]
    if np.any(spans <= 0):
        raise ValueError(f"invalid cell bbox: {bbox.tolist()}")
    normalized = (xy - bbox[0:2]) / spans
    inside = bool(np.all((normalized >= 0.0) & (normalized <= 1.0)))
    return (
        normalized_xy_to_pixel(
            normalized, image_size=image_size, clip=clip
        ),
        inside,
    )


def pixel_to_world_xy(
    pixel_xy: Sequence[float],
    bbox_w: np.ndarray | Sequence[float],
    *,
    image_size: int = IMAGE_SIZE,
) -> np.ndarray:
    pixel = np.asarray(pixel_xy, dtype=np.float64).reshape(2)
    if np.any(pixel < 0) or np.any(pixel > image_size - 1):
        raise ValueError(f"predicted pixel is outside the BEV: {pixel}")
    bbox = np.asarray(bbox_w, dtype=np.float64)
    spans = bbox[3:5] - bbox[0:2]
    x_norm = (pixel[0] + 0.5) / image_size
    y_unflipped = (image_size - 1) - pixel[1]
    y_norm = (y_unflipped + 0.5) / image_size
    return bbox[0:2] + np.asarray([x_norm, y_norm]) * spans


def _rgb_u8(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb)
    if values.size == 0:
        return np.asarray([128, 128, 128], dtype=np.uint8)
    mean = np.nanmean(values.astype(np.float64), axis=0)
    if np.nanmax(mean) <= 1.5:
        mean = mean * 255.0
    return np.clip(np.rint(mean), 0, 255).astype(np.uint8)


def render_processed_cell(
    cell: Any, *, image_size: int = IMAGE_SIZE
) -> tuple[np.ndarray, dict[str, Any]]:
    image = np.full((image_size, image_size, 3), 255, dtype=np.uint8)
    entries: list[tuple[bool, np.ndarray, np.ndarray, Any]] = []
    raw_attribute_count = 0
    point_count = 0
    for obj in cell.objects:
        xyz = np.asarray(obj.xyz)
        rgb = np.asarray(obj.rgb)
        if hasattr(obj, "xyz_raw") or hasattr(obj, "rgb_raw"):
            raw_attribute_count += 1
        if xyz.ndim != 2 or xyz.shape[1] < 2 or not len(xyz):
            continue
        xy = np.clip(xyz[:, :2].astype(np.float64), 0.0, 1.0)
        x = np.floor(xy[:, 0] * image_size - 1e-9).astype(np.int64)
        y0 = np.floor(xy[:, 1] * image_size - 1e-9).astype(np.int64)
        np.clip(x, 0, image_size - 1, out=x)
        np.clip(y0, 0, image_size - 1, out=y0)
        y = (image_size - 1) - y0
        linear = np.unique(y * image_size + x)
        center_xy = np.mean(xy, axis=0)
        entries.append(
            (
                str(obj.label) in STUFF_CLASSES,
                linear,
                center_xy,
                obj,
            )
        )
        point_count += len(xyz)

    # Match the public renderer's draw order: stuff, then instances.
    ordered = [entry for entry in entries if entry[0]]
    ordered.extend(entry for entry in entries if not entry[0])
    flat = image.reshape(-1, 3)
    for _, linear, _, obj in ordered:
        flat[linear] = _rgb_u8(obj.rgb)

    nodes = []
    for node_id, (_, _, center_xy, obj) in enumerate(entries):
        px, py = normalized_xy_to_pixel(
            center_xy, image_size=image_size, clip=True
        )
        nodes.append(
            {
                "node_id": node_id,
                "label": str(obj.label).lower(),
                "pixel_center": [px, py],
                "object_id": str(obj.id),
                "instance_id": str(obj.instance_id),
            }
        )
    return image, {
        "nodes": nodes,
        "object_count": len(cell.objects),
        "rendered_object_count": len(entries),
        "downsampled_point_count": point_count,
        "objects_with_raw_attributes": raw_attribute_count,
    }


def _join_phrases(phrases: Sequence[str]) -> str:
    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return ", ".join(phrases[:-1]) + f", and {phrases[-1]}"


def _query_message(pose: Any) -> str:
    phrases = [
        f"{description.direction} of a "
        f"{description.object_color_text} {description.object_label}"
        for description in pose.descriptions
    ]
    return f" The target location is {_join_phrases(phrases)}."


def _assignments(pose: Any, cell: Any, nodes: Sequence[dict[str, Any]]):
    node_by_object_id = {
        (node["label"], node["object_id"]): node for node in nodes
    }
    nodes_by_instance: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for node in nodes:
        nodes_by_instance.setdefault(
            (node["label"], node["instance_id"]), []
        ).append(node)

    assignments = []
    for description in pose.descriptions:
        label = str(description.object_label).lower()
        node = None
        object_id = getattr(description, "object_id", None)
        if object_id is not None:
            node = node_by_object_id.get((label, str(object_id)))
        if node is None:
            candidates = nodes_by_instance.get(
                (
                    label,
                    str(getattr(description, "object_instance_id", "")),
                ),
                [],
            )
            if candidates:
                node = candidates[0]
        assignments.append(
            {
                "object_label": str(description.object_label),
                "grounded": node is not None,
                "matched_node": node["node_id"] if node else None,
            }
        )
    return assignments


def _sample(
    *,
    pose: Any,
    cell: Any,
    image_path: Path,
    render_info: Mapping[str, Any],
    sample_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    point, inside = world_xy_to_pixel(
        pose.pose_w[0:2], cell.bbox_w, clip=True
    )
    nodes = list(render_info["nodes"])
    public_nodes = [
        {
            "node_id": node["node_id"],
            "label": node["label"],
            "pixel_center": node["pixel_center"],
        }
        for node in nodes
    ]
    answer = {
        "assignments": _assignments(pose, cell, nodes),
        "point_2d": [point[0], point[1]],
    }
    sample = {
        "id": sample_id,
        "messages": [
            {
                "role": "user",
                "content": json.dumps(
                    {"nodes": public_nodes}, ensure_ascii=False
                ),
            },
            {
                "role": "user",
                "content": f"<image>{_query_message(pose)}",
            },
            {
                "role": "assistant",
                "content": json.dumps(answer, ensure_ascii=False),
            },
        ],
        "images": [str(image_path.resolve())],
    }
    index = {
        "id": sample_id,
        "pose_world_xy": [
            float(pose.pose_w[0]),
            float(pose.pose_w[1]),
        ],
        "candidate_cell_id": str(cell.id),
        "candidate_bbox_w": [
            float(value) for value in np.asarray(cell.bbox_w).tolist()
        ],
        "target_pixel_clipped": [point[0], point[1]],
        "target_inside_candidate": inside,
        "image_path": str(image_path.resolve()),
    }
    return sample, index


def _render_cell_once(
    cell: Any,
    *,
    image_root: Path,
    overwrite: bool,
) -> tuple[Path, dict[str, Any]]:
    scene_dir = image_root / str(cell.scene_name)
    scene_dir.mkdir(parents=True, exist_ok=True)
    image_path = scene_dir / f"{cell.id}.png"
    image, info = render_processed_cell(cell)
    if overwrite or not image_path.is_file():
        Image.fromarray(image, mode="RGB").save(image_path)
    else:
        with Image.open(image_path) as existing:
            if existing.size != (IMAGE_SIZE, IMAGE_SIZE):
                raise RuntimeError(
                    f"cached image has wrong size: {image_path}"
                )
    return image_path, info


def _prepare_gt_split(
    data_root: Path,
    scenes: Sequence[str],
    *,
    output_dir: Path,
    split_name: str,
    overwrite_images: bool,
) -> tuple[Path, dict[str, Any]]:
    image_root = output_dir / "images" / split_name
    samples = []
    outside_count = 0
    cell_count = 0
    sample_index = 0
    # Keep only one scene's point arrays in memory at a time. The JSON sample
    # order remains the declared split order.
    for scene in scenes:
        dataset = load_ordered_dataset(data_root, [scene])
        cells = {str(cell.id): cell for cell in dataset.all_cells}
        rendered: dict[str, tuple[Path, dict[str, Any]]] = {}
        for cell in dataset.all_cells:
            rendered[str(cell.id)] = _render_cell_once(
                cell,
                image_root=image_root,
                overwrite=overwrite_images,
            )
        cell_count += len(dataset.all_cells)
        for pose in dataset.all_poses:
            cell = cells.get(str(pose.cell_id))
            if cell is None:
                raise RuntimeError(
                    f"{split_name} pose {sample_index} references unknown GT "
                    f"cell {pose.cell_id}"
                )
            image_path, render_info = rendered[str(cell.id)]
            sample, metadata = _sample(
                pose=pose,
                cell=cell,
                image_path=image_path,
                render_info=render_info,
                sample_id=f"{split_name}_{sample_index:06d}",
            )
            outside_count += int(
                not metadata["target_inside_candidate"]
            )
            samples.append(sample)
            sample_index += 1
    if outside_count:
        raise RuntimeError(
            f"{split_name} contains {outside_count} GT poses outside their "
            "own cells"
        )

    json_path = output_dir / f"vlmloc_{split_name}_data.json"
    _write_json_atomic(json_path, samples)
    stats = {
        "scenes": list(scenes),
        "sample_count": len(samples),
        "cell_count": cell_count,
        "dataset_json_path": str(json_path.resolve()),
        "dataset_json_sha256": _sha256_file(json_path),
    }
    return json_path, stats


def prepare_vlmloc_data(
    *,
    data_root: Path,
    manifest_path: Path,
    output_dir: Path,
    overwrite_images: bool = False,
) -> Path:
    from datapreparation.kitti360pose import utils as source_utils

    source_split_configuration = {
        "train": tuple(source_utils.SCENE_NAMES_TRAIN),
        "validation": tuple(source_utils.SCENE_NAMES_VAL),
        "test": tuple(source_utils.SCENE_NAMES_TEST),
        "stuff_classes": frozenset(source_utils.STUFF_CLASSES),
    }
    expected_split_configuration = {
        "train": SCENE_NAMES_TRAIN,
        "validation": SCENE_NAMES_VAL,
        "test": SCENE_NAMES_TEST,
        "stuff_classes": STUFF_CLASSES,
    }
    if source_split_configuration != expected_split_configuration:
        raise RuntimeError(
            "KITTI360Pose source split/stuff-class ordering changed: "
            f"{source_split_configuration}"
        )

    data_root = Path(data_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    training_path, training_stats = _prepare_gt_split(
        data_root,
        SCENE_NAMES_TRAIN,
        output_dir=output_dir,
        split_name="training",
        overwrite_images=overwrite_images,
    )
    validation_path, validation_stats = _prepare_gt_split(
        data_root,
        SCENE_NAMES_VAL,
        output_dir=output_dir,
        split_name="validation",
        overwrite_images=overwrite_images,
    )

    test_dataset = load_ordered_dataset(data_root, SCENE_NAMES_TEST)
    signature = dataset_signature(test_dataset)
    if signature["query_count"] != QUERY_COUNT:
        raise RuntimeError(
            f"expected {QUERY_COUNT} ordered test queries, got "
            f"{signature['query_count']}"
        )
    manifest, retrievals = load_table8_like_manifest(
        manifest_path, dataset=test_dataset
    )
    cells = {str(cell.id): cell for cell in test_dataset.all_cells}
    candidate_ids = [row[0] for row in retrievals]
    unique_candidate_ids = list(dict.fromkeys(candidate_ids))
    image_root = output_dir / "images" / "testing_cmmloc_top1"
    rendered = {}
    for cell_id in unique_candidate_ids:
        cell = cells[cell_id]
        rendered[cell_id] = _render_cell_once(
            cell,
            image_root=image_root,
            overwrite=overwrite_images,
        )

    test_samples = []
    test_index = []
    outside_count = 0
    for query_index, (pose, candidate_id) in enumerate(
        zip(test_dataset.all_poses, candidate_ids)
    ):
        cell = cells[candidate_id]
        image_path, render_info = rendered[candidate_id]
        sample, index = _sample(
            pose=pose,
            cell=cell,
            image_path=image_path,
            render_info=render_info,
            sample_id=f"test_{query_index:06d}",
        )
        index.update(
            {
                "query_index": query_index,
                "query_scene": str(pose.scene_name),
                "query_gt_cell_id": str(pose.cell_id),
            }
        )
        outside_count += int(not index["target_inside_candidate"])
        test_samples.append(sample)
        test_index.append(index)

    testing_path = output_dir / "vlmloc_testing_data.json"
    index_path = output_dir / "vlmloc_testing_index.json"
    smoke_path = output_dir / "vlmloc_testing_smoke_1.json"
    _write_json_atomic(testing_path, test_samples)
    _write_json_atomic(index_path, test_index)
    _write_json_atomic(smoke_path, test_samples[:1])
    audit = {
        "schema_version": 1,
        "backend": "vlmloc_kitti360pose_30m_retrained",
        "comparison_scope": "table8_like_full_test_11505",
        "renderer": RENDERER_NAME,
        "renderer_exactly_matches_public_dense_raw_renderer": False,
        "renderer_reason": (
            "The local processed KITTI360Pose cells contain normalized "
            "downsampled xyz/rgb arrays but no xyz_raw/rgb_raw fields. The "
            "adapter must be retrained on these generated images."
        ),
        "image_size_px": IMAGE_SIZE,
        "bev_range_m": BEV_RANGE_M,
        "meters_per_pixel": METERS_PER_PIXEL,
        "training": training_stats,
        "validation": validation_stats,
        "testing": {
            "scenes": list(SCENE_NAMES_TEST),
            "sample_count": len(test_samples),
            "unique_candidate_cell_count": len(unique_candidate_ids),
            "target_outside_candidate_count": outside_count,
            "dataset_json_path": str(testing_path.resolve()),
            "dataset_json_sha256": _sha256_file(testing_path),
            "index_path": str(index_path.resolve()),
            "index_sha256": _sha256_file(index_path),
            "smoke_dataset_path": str(smoke_path.resolve()),
            "smoke_dataset_sha256": _sha256_file(smoke_path),
            "ordered_dataset_signature": signature,
        },
        "cmmloc_manifest_path": str(Path(manifest_path).resolve()),
        "cmmloc_manifest_sha256": _sha256_file(Path(manifest_path)),
        "cmmloc_retrieval_rows_sha256": manifest[
            "retrieval_rows_sha256"
        ],
        "training_dataset_path": str(training_path.resolve()),
        "validation_dataset_path": str(validation_path.resolve()),
        "testing_dataset_path": str(testing_path.resolve()),
    }
    audit_path = output_dir / "vlmloc_data_preparation_audit.json"
    _write_json_atomic(audit_path, audit)
    return audit_path


def audit_vlmloc_runtime_preflight(
    *,
    adapter_dir: Path,
    base_model_dir: Path,
    official_source_dir: Path,
    data_dir: Path,
    smoke_predictions_path: Path,
    output_dir: Path,
) -> Path:
    """Audit a retrained adapter after a one-query ``swift infer`` smoke."""

    adapter_dir = Path(adapter_dir).resolve()
    base_model_dir = Path(base_model_dir).resolve()
    official_source_dir = Path(official_source_dir).resolve()
    data_dir = Path(data_dir).resolve()
    smoke_predictions_path = Path(smoke_predictions_path).resolve()
    preparation_path = data_dir / "vlmloc_data_preparation_audit.json"
    preparation = (
        json.loads(preparation_path.read_text(encoding="utf-8"))
        if preparation_path.is_file()
        else {}
    )
    adapter = inspect_adapter(adapter_dir)
    base_model = inspect_base_model(base_model_dir)
    source = inspect_official_source(official_source_dir)
    environment = inspect_python_environment()
    smoke_rows = (
        _load_jsonl(smoke_predictions_path)
        if smoke_predictions_path.is_file()
        else []
    )
    smoke_point = (
        _response_point(smoke_rows[0]) if len(smoke_rows) == 1 else None
    )

    expected_data_files = {
        "training": data_dir / "vlmloc_training_data.json",
        "validation": data_dir / "vlmloc_validation_data.json",
        "testing": data_dir / "vlmloc_testing_data.json",
        "test_index": data_dir / "vlmloc_testing_index.json",
        "smoke": data_dir / "vlmloc_testing_smoke_1.json",
    }
    data_file_audit = {
        name: {
            "path": str(path.resolve()),
            "exists": path.is_file(),
            "sha256": _sha256_file(path) if path.is_file() else None,
        }
        for name, path in expected_data_files.items()
    }
    recorded_hashes = {
        "training": preparation.get("training", {}).get(
            "dataset_json_sha256"
        ),
        "validation": preparation.get("validation", {}).get(
            "dataset_json_sha256"
        ),
        "testing": preparation.get("testing", {}).get(
            "dataset_json_sha256"
        ),
        "test_index": preparation.get("testing", {}).get("index_sha256"),
        "smoke": preparation.get("testing", {}).get(
            "smoke_dataset_sha256"
        ),
    }
    data_hashes_match = all(
        record["exists"] and record["sha256"] == recorded_hashes[name]
        for name, record in data_file_audit.items()
    )

    adapter_mismatches = {}
    expected_adapter = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "rank": 8,
        "lora_alpha": 16,
        "training_model_type": "qwen3_vl",
        "training_template": "qwen3_vl",
        "training_type": "lora",
        "torch_dtype": "bfloat16",
        "training_target_modules": ["all-linear"],
        "training_freeze_vit": False,
        "training_freeze_aligner": False,
        "training_num_epochs": 5.0,
        "training_learning_rate": 1e-4,
        "training_gradient_accumulation_steps": 2,
        "training_seed": 42,
        "data_seed": 42,
    }
    for key, expected in expected_adapter.items():
        if adapter.get(key) != expected:
            adapter_mismatches[key] = {
                "expected": expected,
                "actual": adapter.get(key),
            }
    training_paths = {
        str(Path(value).resolve())
        for value in (adapter.get("training_datasets") or [])
        if isinstance(value, str) and Path(value).exists()
    }
    validation_paths = {
        str(Path(value).resolve())
        for value in (adapter.get("validation_datasets") or [])
        if isinstance(value, str) and Path(value).exists()
    }
    expected_training = str(
        expected_data_files["training"].resolve()
    )
    expected_validation = str(
        expected_data_files["validation"].resolve()
    )
    if expected_training not in training_paths:
        adapter_mismatches["training_dataset"] = {
            "expected": expected_training,
            "actual": sorted(training_paths),
        }
    if expected_validation not in validation_paths:
        adapter_mismatches["validation_dataset"] = {
            "expected": expected_validation,
            "actual": sorted(validation_paths),
        }

    checks = {
        "prepared_data_complete_and_hashes_match": bool(
            preparation
            and preparation.get("renderer") == RENDERER_NAME
            and preparation.get("testing", {}).get("sample_count")
            == QUERY_COUNT
            and data_hashes_match
        ),
        "adapter_metadata_and_internal_lora_shapes": bool(
            adapter.get("adapter_config_exists")
            and adapter.get("adapter_weights_exists")
            and adapter.get("args_exists")
            and not adapter.get("internal_shape_mismatches", ["missing"])
            and not adapter.get("unpaired_lora_keys", ["missing"])
            and not adapter_mismatches
        ),
        "base_model_complete": bool(
            base_model.get("architecture_matches_qwen3_vl_8b")
            and not base_model.get("missing_base_shards", ["not-audited"])
            and isinstance(base_model.get("resolved_model_revision"), str)
            and bool(
                re.fullmatch(
                    r"[0-9a-f]{40}",
                    base_model["resolved_model_revision"],
                )
            )
        ),
        "official_source_and_environment": bool(
            source.get("source_commit_matches")
            and not source.get("missing_required_source_files", ["missing"])
            and environment.get("package_versions", {}).get("ms-swift")
            and environment.get("package_versions", {}).get("transformers")
            and environment.get("package_versions", {}).get("peft")
        ),
        "one_query_runtime_model_load_and_generation_smoke": bool(
            len(smoke_rows) == 1 and smoke_point is not None
        ),
    }
    report = {
        "schema_version": 1,
        "backend": "vlmloc_kitti360pose_30m_retrained",
        "comparison_scope": "table8_like_full_test_11505",
        "adapter_audit": adapter,
        "adapter_mismatches": adapter_mismatches,
        "base_model_audit": base_model,
        "expected_base_architecture": QWEN3_VL_8B_ARCHITECTURE,
        "official_source_audit": source,
        "python_environment_audit": environment,
        "preparation_audit_path": str(preparation_path),
        "preparation_audit": preparation,
        "prepared_data_files": data_file_audit,
        "prepared_data_recorded_hashes": recorded_hashes,
        "smoke_predictions_path": str(smoke_predictions_path),
        "smoke_predictions_sha256": (
            _sha256_file(smoke_predictions_path)
            if smoke_predictions_path.is_file()
            else None
        ),
        "smoke_prediction_count": len(smoke_rows),
        "smoke_point_2d": list(smoke_point) if smoke_point else None,
        "checks": checks,
        "compatible": all(checks.values()),
        "load_attempted": bool(smoke_rows),
        "load_succeeded": checks[
            "one_query_runtime_model_load_and_generation_smoke"
        ],
        "shape_validation": {
            "lora_pair_internal_shapes": (
                "PASS"
                if not adapter.get(
                    "internal_shape_mismatches", ["missing"]
                )
                else "FAIL"
            ),
            "runtime_adapter_to_base_load": (
                "PASS"
                if checks[
                    "one_query_runtime_model_load_and_generation_smoke"
                ]
                else "FAIL"
            ),
        },
        "important_scope": (
            "This validates the separately retrained Table-8-like 30 m "
            "adapter and processed-point renderer, not the unavailable exact "
            "Table-8 adapter/dense-raw renderer."
        ),
    }
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "vlmloc_runtime_preflight.json"
    _write_json_atomic(report_path, report)
    return report_path


def _safe_response_json(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            return None
        try:
            parsed, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None


def _response_point(row: Mapping[str, Any]) -> tuple[float, float] | None:
    parsed = _safe_response_json(
        row.get("response", row.get("output", row.get("prediction")))
    )
    if not parsed:
        return None
    point = parsed.get("point_2d")
    if (
        not isinstance(point, (list, tuple))
        or len(point) != 2
    ):
        return None
    try:
        values = (float(point[0]), float(point[1]))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    if not all(0 <= value <= IMAGE_SIZE - 1 for value in values):
        return None
    return values


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"invalid prediction JSONL at line {line_number}"
                ) from error
            if not isinstance(row, dict):
                raise RuntimeError(
                    f"prediction line {line_number} is not an object"
                )
            rows.append(row)
    return rows


def evaluate_vlmloc_predictions(
    *,
    predictions_path: Path,
    test_index_path: Path,
    output_dir: Path,
) -> Path:
    predictions = _load_jsonl(predictions_path)
    test_index = json.loads(
        Path(test_index_path).read_text(encoding="utf-8")
    )
    if len(test_index) != QUERY_COUNT:
        raise RuntimeError(
            f"test index has {len(test_index)} rows, expected {QUERY_COUNT}"
        )
    if len(predictions) != len(test_index):
        raise RuntimeError(
            f"prediction count {len(predictions)} does not match test index "
            f"count {len(test_index)}"
        )

    errors_m = []
    invalid = 0
    per_query = []
    for index, (prediction, metadata) in enumerate(
        zip(predictions, test_index)
    ):
        expected_id = metadata["id"]
        observed_id = prediction.get("id")
        if observed_id is not None and str(observed_id) != expected_id:
            raise RuntimeError(
                f"prediction order/id mismatch at {index}: "
                f"{observed_id!r} != {expected_id!r}"
            )
        point = _response_point(prediction)
        error_m = None
        if point is None:
            invalid += 1
        else:
            predicted_world = pixel_to_world_xy(
                point, metadata["candidate_bbox_w"]
            )
            target_world = np.asarray(
                metadata["pose_world_xy"], dtype=np.float64
            )
            error_m = float(np.linalg.norm(predicted_world - target_world))
            errors_m.append(error_m)
        per_query.append(
            {
                "query_index": index,
                "id": expected_id,
                "candidate_cell_id": metadata["candidate_cell_id"],
                "valid_prediction": error_m is not None,
                "error_m": error_m,
            }
        )

    total = len(test_index)
    metrics = {
        str(threshold): sum(
            error <= threshold for error in errors_m
        )
        / total
        for threshold in (5, 10, 15)
    }
    paper_reference = {"5": 0.4036, "10": 0.5169, "15": 0.5474}
    result = {
        "schema_version": 1,
        "backend": "vlmloc_kitti360pose_30m_retrained",
        "protocol": "CMMLoc release coarse Top-1 -> VLM-Loc fine",
        "query_count": total,
        "valid_prediction_count": len(errors_m),
        "invalid_prediction_count": invalid,
        "invalid_predictions_count_as_misses": True,
        "metric": "world-coordinate localization recall",
        "thresholds_m": [5, 10, 15],
        "recall": metrics,
        "table8_reference_recall_not_exact_target": paper_reference,
        "difference_from_table8_reference_percentage_points": {
            threshold: 100.0 * (metrics[threshold] - reference)
            for threshold, reference in paper_reference.items()
        },
        "comparison_warning": (
            "The reference uses the paper's 11,404-query/dense-raw-renderer "
            "setup. This result uses all 11,505 local queries and a retrained "
            "processed-point renderer, so closeness is a sanity check rather "
            "than an exact reproduction criterion."
        ),
        "mean_error_m_valid_only": (
            float(np.mean(errors_m)) if errors_m else None
        ),
        "median_error_m_valid_only": (
            float(np.median(errors_m)) if errors_m else None
        ),
        "predictions_path": str(Path(predictions_path).resolve()),
        "predictions_sha256": _sha256_file(Path(predictions_path)),
        "test_index_path": str(Path(test_index_path).resolve()),
        "test_index_sha256": _sha256_file(Path(test_index_path)),
        "per_query": per_query,
    }
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "vlmloc_fine_metrics.json"
    _write_json_atomic(result_path, result)
    return result_path
