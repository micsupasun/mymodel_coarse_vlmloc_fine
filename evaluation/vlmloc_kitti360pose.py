"""Prepare and score the Table-8-like KITTI360Pose VLM-Loc fine stage.

The public VLM-Loc release only provides CityLoc 50 m data/adapters.  This
module creates a separate bundle of VLM input representations (BEV images and
JSON prompts) from the ordered local KITTI360Pose pickles.  It does not create
new KITTI360Pose samples or modify the source dataset.  Test examples use the
exact CMMLoc Top-1 candidate stored in the checksummed Table-8-like manifest.

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
import time
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
    audit_merged_lora_application,
    inspect_adapter,
    inspect_base_model,
    inspect_full_model,
    inspect_official_source,
    inspect_python_environment,
)


IMAGE_SIZE = 224
BEV_RANGE_M = 30.0
METERS_PER_PIXEL = BEV_RANGE_M / IMAGE_SIZE
RENDERER_NAME = "processed_cell_downsampled_points_v1"
DENSE_RAW_RENDERER_NAME = (
    "official_dense_raw_points_on_origin_kitti360pose_cells_v1"
)
HYBRID_RAW_RENDERER_NAME = (
    "official_dense_raw_with_audited_processed_missing_objects_v1"
)
SUPPORTED_RENDERERS = frozenset(
    {
        RENDERER_NAME,
        DENSE_RAW_RENDERER_NAME,
        HYBRID_RAW_RENDERER_NAME,
    }
)
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
RAW_INSTANCE_KEY_STRIDE = 10_000_000


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


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return False
    return True


def _assert_separate_derived_output(
    *,
    data_root: Path,
    output_dir: Path,
    raw_kitti360_root: Path | None,
) -> None:
    """Fail closed if generated VLM inputs could overlap source data."""

    source_roots = [("KITTI360Pose", Path(data_root).resolve())]
    if raw_kitti360_root is not None:
        source_roots.append(
            ("raw KITTI-360", Path(raw_kitti360_root).resolve())
        )
    resolved_output = Path(output_dir).resolve()
    for label, source_root in source_roots:
        if _path_is_within(resolved_output, source_root) or _path_is_within(
            source_root, resolved_output
        ):
            raise RuntimeError(
                "VLM-Loc derived output must be separate from the "
                f"{label} source tree: output={resolved_output}, "
                f"source={source_root}"
            )


def _source_tree_metadata_snapshot(root: Path) -> dict[str, Any]:
    """Fingerprint names/sizes/timestamps without reading 19 GB of payloads."""

    root = Path(root).resolve()
    digest = hashlib.sha256()
    file_count = 0
    byte_count = 0
    for path in sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    ):
        stat = path.stat()
        relative = path.relative_to(root).as_posix()
        digest.update(
            (
                f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n"
            ).encode("utf-8")
        )
        file_count += 1
        byte_count += stat.st_size
    return {
        "root": str(root),
        "file_count": file_count,
        "byte_count": byte_count,
        "relative_path_size_mtime_sha256": digest.hexdigest(),
    }


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
        scene_short = scene.split("_")[-2]
        accepted_scene_names = {scene, scene_short}
        if any(
            str(pose.scene_name) not in accepted_scene_names
            for pose in scene_poses
        ):
            raise RuntimeError(f"pose scene metadata mismatch in {pose_path}")
        if any(
            str(cell.scene_name) not in accepted_scene_names
            for cell in scene_cells
        ):
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
    entries: list[tuple[bool, np.ndarray, Any]] = []
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
        entries.append(
            (
                str(obj.label) in STUFF_CLASSES,
                linear,
                obj,
            )
        )
        point_count += len(xyz)

    # Match the public renderer's draw order: stuff, then instances.
    ordered = [entry for entry in entries if entry[0]]
    ordered.extend(entry for entry in entries if not entry[0])
    flat = image.reshape(-1, 3)
    nodes = []
    for node_id, (_, linear, obj) in enumerate(ordered):
        flat[linear] = _rgb_u8(obj.rgb)
        ys = (linear // image_size).astype(np.float64)
        xs = (linear % image_size).astype(np.float64)
        x_mean = float(np.mean(xs))
        y_mean = float(np.mean(ys))
        world_center = pixel_to_world_xy(
            (x_mean, y_mean),
            cell.bbox_w,
            image_size=image_size,
        )
        nodes.append(
            {
                "node_id": node_id,
                "label": str(obj.label).lower(),
                "pixel_center": [
                    int(round(x_mean)),
                    int(round(y_mean)),
                ],
                "world_center": [
                    float(world_center[0]),
                    float(world_center[1]),
                ],
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


def _raw_semantic_scene_directory(
    raw_kitti360_root: Path, scene: str
) -> Path:
    root = Path(raw_kitti360_root).resolve()
    candidates = (
        root / "data_3d_semantics" / scene / "static",
        root / "data_3d_semantics" / "train" / scene / "static",
        root / "data_3d_semantics" / "test" / scene / "static",
        root / scene / "static",
    )
    matches = [
        candidate
        for candidate in candidates
        if candidate.is_dir() and any(candidate.glob("*.ply"))
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            "Expected exactly one KITTI-360 semantic PLY directory for "
            f"{scene}; checked {[str(path) for path in candidates]}, "
            f"found {[str(path) for path in matches]}. Download/extract and "
            "audit the official raw files first with: python "
            "scripts\\setup_kitti360_raw_vlmloc_gpu.py --raw-root "
            '"data\\KITTI-360-raw" --output-dir '
            '"evaluation_outputs\\kitti360_raw_setup"'
        )
    return matches[0]


def _required_raw_object_keys(
    cells: Sequence[Any],
) -> set[tuple[str, int]]:
    return {
        (str(obj.label).lower(), int(obj.instance_id))
        for cell in cells
        for obj in cell.objects
    }


def _pack_semantic_instance(
    semantic: np.ndarray | Sequence[int],
    instance: np.ndarray | Sequence[int],
) -> np.ndarray:
    """Pack exact pairs for one vectorized membership test."""

    semantic_values = np.asarray(semantic, dtype=np.int64)
    instance_values = np.asarray(instance, dtype=np.int64)
    if semantic_values.shape != instance_values.shape:
        raise RuntimeError(
            "semantic and instance arrays have different shapes: "
            f"{semantic_values.shape}, {instance_values.shape}"
        )
    if np.any(instance_values < 0) or np.any(
        instance_values >= RAW_INSTANCE_KEY_STRIDE
    ):
        raise RuntimeError(
            "raw instance ID is outside the collision-free packed range "
            f"[0, {RAW_INSTANCE_KEY_STRIDE})"
        )
    return semantic_values * RAW_INSTANCE_KEY_STRIDE + instance_values


def _group_selected_indices_by_packed_key(
    packed_keys: np.ndarray,
    selected_indices: np.ndarray,
) -> dict[int, np.ndarray]:
    """Group selected vertex rows in one sort instead of one scan per key."""

    indices = np.asarray(selected_indices, dtype=np.int64)
    if not len(indices):
        return {}
    selected_keys = np.asarray(packed_keys, dtype=np.int64)[indices]
    order = np.argsort(selected_keys, kind="stable")
    sorted_keys = selected_keys[order]
    unique_keys, starts = np.unique(sorted_keys, return_index=True)
    ends = np.concatenate(
        (starts[1:], np.asarray([len(sorted_keys)], dtype=np.int64))
    )
    return {
        int(key): indices[order[start:end]]
        for key, start, end in zip(unique_keys, starts, ends)
    }


def _rgb_points_u8(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise RuntimeError(f"invalid per-point RGB shape: {values.shape}")
    if values.size and np.nanmax(values) <= 1.5:
        values = values * 255.0
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def _processed_fallback_objects(
    cells: Sequence[Any],
    missing_keys: set[tuple[str, int]],
) -> tuple[
    dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    list[dict[str, Any]],
]:
    """Recover only raw-release gaps from authoritative processed cells."""

    occurrences: dict[
        tuple[str, int],
        list[tuple[str, np.ndarray, np.ndarray]],
    ] = {key: [] for key in missing_keys}
    for cell in cells:
        bbox = np.asarray(cell.bbox_w, dtype=np.float64)
        for obj in cell.objects:
            key = (str(obj.label).lower(), int(obj.instance_id))
            if key not in occurrences:
                continue
            normalized = np.asarray(obj.xyz, dtype=np.float64)
            if (
                normalized.ndim != 2
                or normalized.shape[1] != 3
                or not len(normalized)
            ):
                raise RuntimeError(
                    f"invalid processed points for {key} in cell {cell.id}: "
                    f"{normalized.shape}"
                )
            world = (
                normalized * float(cell.cell_size) + bbox[0:3]
            ).astype(np.float32)
            occurrences[key].append(
                (str(cell.id), world, _rgb_points_u8(obj.rgb))
            )

    recovered = {}
    audit = []
    for key in sorted(missing_keys):
        rows = occurrences[key]
        if not rows:
            raise RuntimeError(
                f"No processed KITTI360Pose points exist for missing raw "
                f"object {key}"
            )
        label, instance_id = key
        if label not in STUFF_CLASSES:
            canonical = max(rows, key=lambda row: len(row[1]))
            canonical_order = np.lexsort(
                (
                    canonical[1][:, 2],
                    canonical[1][:, 1],
                    canonical[1][:, 0],
                )
            )
            canonical_sorted = canonical[1][canonical_order]
            inconsistent = [
                cell_id
                for cell_id, xyz, _ in rows
                if len(xyz) != len(canonical[1])
                or np.max(
                    np.abs(
                        xyz[
                            np.lexsort(
                                (xyz[:, 2], xyz[:, 1], xyz[:, 0])
                            )
                        ]
                        - canonical_sorted
                    )
                )
                > 1e-3
            ]
            if inconsistent:
                raise RuntimeError(
                    "Processed non-stuff fallback occurrences are not the "
                    f"same full object for {key}: {inconsistent[:20]}"
                )
            xyz, rgb = canonical[1], canonical[2]
        else:
            xyz = np.concatenate([row[1] for row in rows], axis=0)
            rgb = np.concatenate([row[2] for row in rows], axis=0)
            _, unique_indices = np.unique(
                np.round(xyz.astype(np.float64), decimals=4),
                axis=0,
                return_index=True,
            )
            unique_indices.sort()
            xyz = xyz[unique_indices]
            rgb = rgb[unique_indices]
        recovered[key] = (xyz, rgb)
        audit.append(
            {
                "label": label,
                "instance_id": instance_id,
                "reason": "absent_from_current_official_raw_archive",
                "source": "original_processed_kitti360pose_cell_points",
                "occurrence_count": len(rows),
                "source_cell_ids": [row[0] for row in rows],
                "point_count": len(xyz),
            }
        )
    return recovered, audit


def _load_raw_scene_objects(
    raw_kitti360_root: Path,
    scene: str,
    cells: Sequence[Any],
    *,
    allow_processed_missing_raw_fallback: bool = False,
) -> tuple[dict[tuple[str, int], tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    try:
        from plyfile import PlyData
    except ImportError as error:
        raise RuntimeError(
            "Dense VLM-Loc rendering requires plyfile. Install it in the "
            "CMMLoc environment without upgrading its NumPy-1.x ABI: "
            'python -m pip install --force-reinstall "numpy==1.26.4" '
            '"plyfile==1.1.3"'
        ) from error

    from datapreparation.kitti360pose.utils import CLASS_TO_LABEL

    required = _required_raw_object_keys(cells)
    label_to_semantic = {
        str(label).lower(): int(semantic)
        for label, semantic in CLASS_TO_LABEL.items()
    }
    unknown_labels = sorted(
        label for label, _ in required if label not in label_to_semantic
    )
    if unknown_labels:
        raise RuntimeError(
            f"Raw KITTI-360 semantic mapping lacks labels: {unknown_labels}"
        )
    key_by_numeric: dict[tuple[int, int], tuple[str, int]] = {}
    for label, instance_id in required:
        semantic = label_to_semantic[label]
        key_by_numeric[(semantic, instance_id)] = (label, instance_id)
    required_packed_keys = _pack_semantic_instance(
        [semantic for semantic, _ in key_by_numeric],
        [instance for _, instance in key_by_numeric],
    )

    scene_dir = _raw_semantic_scene_directory(
        raw_kitti360_root, scene
    )
    ply_paths = sorted(scene_dir.glob("*.ply"))
    xyz_parts: dict[tuple[str, int], list[np.ndarray]] = {
        key: [] for key in required
    }
    rgb_parts: dict[tuple[str, int], list[np.ndarray]] = {
        key: [] for key in required
    }
    inventory = []
    total_vertices = 0
    selected_vertices = 0
    required_fields = {
        "x",
        "y",
        "z",
        "red",
        "green",
        "blue",
        "semantic",
        "instance",
    }
    print(
        f"Raw scene scan: scene={scene}, ply_files={len(ply_paths)}, "
        f"required_object_keys={len(required)}",
        flush=True,
    )
    scene_scan_started = time.perf_counter()
    for ply_index, ply_path in enumerate(ply_paths, start=1):
        ply_started = time.perf_counter()
        vertex = PlyData.read(str(ply_path))["vertex"].data
        fields = set(vertex.dtype.names or ())
        missing_fields = sorted(required_fields - fields)
        if missing_fields:
            raise RuntimeError(
                f"{ply_path} lacks required vertex fields: {missing_fields}"
            )
        semantic = np.asarray(vertex["semantic"], dtype=np.int64)
        instance = np.asarray(vertex["instance"], dtype=np.int64)
        total_vertices += len(vertex)
        packed_keys = _pack_semantic_instance(semantic, instance)
        selected_mask = np.isin(packed_keys, required_packed_keys)
        selected_indices = np.flatnonzero(selected_mask)
        selected_vertices += len(selected_indices)
        if len(selected_indices):
            grouped_rows = _group_selected_indices_by_packed_key(
                packed_keys, selected_indices
            )
            for packed_key, rows in grouped_rows.items():
                semantic_id = packed_key // RAW_INSTANCE_KEY_STRIDE
                instance_id = packed_key % RAW_INSTANCE_KEY_STRIDE
                key = key_by_numeric.get(
                    (int(semantic_id), int(instance_id))
                )
                if key is None:
                    continue
                xyz_parts[key].append(
                    np.column_stack(
                        (
                            vertex["x"][rows],
                            vertex["y"][rows],
                            vertex["z"][rows],
                        )
                    ).astype(np.float32, copy=False)
                )
                rgb_parts[key].append(
                    np.column_stack(
                        (
                            vertex["red"][rows],
                            vertex["green"][rows],
                            vertex["blue"][rows],
                        )
                    ).astype(np.uint8, copy=False)
                )
        stat = ply_path.stat()
        inventory.append(
            {
                "path": str(ply_path.resolve()),
                "size_bytes": stat.st_size,
                "vertex_count": len(vertex),
            }
        )
        print(
            f"Raw PLY {ply_index}/{len(ply_paths)}: {ply_path.name}, "
            f"vertices={len(vertex)}, selected={len(selected_indices)}, "
            f"seconds={time.perf_counter() - ply_started:.2f}",
            flush=True,
        )
    print(
        f"Raw scene scan complete: scene={scene}, "
        f"vertices={total_vertices}, selected={selected_vertices}, "
        f"seconds={time.perf_counter() - scene_scan_started:.2f}",
        flush=True,
    )

    missing_key_tuples = {
        key for key, parts in xyz_parts.items() if not parts
    }
    missing_keys = sorted(
        f"{label}:{instance_id}"
        for label, instance_id in missing_key_tuples
    )
    if missing_keys and not allow_processed_missing_raw_fallback:
        raise RuntimeError(
            f"Raw KITTI-360 files for {scene} lack required objects: "
            f"{missing_keys[:30]}. This is a raw-release compatibility gap, "
            "not a missing scene. To use only the original processed "
            "KITTI360Pose points for these explicitly audited missing keys, "
            "rerun with --allow-processed-missing-raw-fallback."
        )
    fallback_objects: dict[
        tuple[str, int], tuple[np.ndarray, np.ndarray]
    ] = {}
    fallback_audit: list[dict[str, Any]] = []
    if missing_key_tuples:
        fallback_objects, fallback_audit = _processed_fallback_objects(
            cells, missing_key_tuples
        )
    raw_objects = {}
    for key in required:
        if xyz_parts[key]:
            xyz = np.concatenate(xyz_parts[key], axis=0)
            rgb = np.concatenate(rgb_parts[key], axis=0)
        else:
            xyz, rgb = fallback_objects[key]
        if key[0] in STUFF_CLASSES and len(xyz):
            order = np.argsort(xyz[:, 0], kind="stable")
            xyz = xyz[order]
            rgb = rgb[order]
        raw_objects[key] = (xyz, rgb)
    inventory_signature = hashlib.sha256(
        json.dumps(
            inventory,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return raw_objects, {
        "scene": scene,
        "semantic_directory": str(scene_dir.resolve()),
        "ply_file_count": len(ply_paths),
        "total_vertex_count": total_vertices,
        "selected_vertex_count": selected_vertices,
        "required_object_key_count": len(required),
        "exact_raw_object_key_count": (
            len(required) - len(missing_key_tuples)
        ),
        "processed_fallback_object_key_count": len(missing_key_tuples),
        "processed_fallback_objects": fallback_audit,
        "stuff_point_arrays_sorted_by_world_x": True,
        "allow_processed_missing_raw_fallback": (
            allow_processed_missing_raw_fallback
        ),
        "inventory_metadata_sha256": inventory_signature,
        "inventory_is_content_hash": False,
        "files": inventory,
    }


def _world_points_to_linear_pixels(
    xyz: np.ndarray,
    bbox_w: np.ndarray,
    image_size: int,
) -> np.ndarray:
    points = np.asarray(xyz, dtype=np.float64)
    bbox = np.asarray(bbox_w, dtype=np.float64)
    spans = bbox[3:5] - bbox[0:2]
    if points.ndim != 2 or points.shape[1] < 2 or not len(points):
        return np.empty(0, dtype=np.int64)
    if np.any(spans <= 0):
        raise RuntimeError(f"invalid cell bbox: {bbox.tolist()}")
    x = np.floor(
        (points[:, 0] - bbox[0]) * image_size / spans[0] - 1e-9
    ).astype(np.int64)
    y_unflipped = np.floor(
        (points[:, 1] - bbox[1]) * image_size / spans[1] - 1e-9
    ).astype(np.int64)
    np.clip(x, 0, image_size - 1, out=x)
    np.clip(y_unflipped, 0, image_size - 1, out=y_unflipped)
    y = (image_size - 1) - y_unflipped
    return np.unique(y * image_size + x)


def _crop_raw_points_to_bbox(
    xyz: np.ndarray,
    rgb: np.ndarray,
    bbox: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop identically, using binary search when points are x-sorted."""

    points = np.asarray(xyz)
    colors = np.asarray(rgb)
    if len(points) != len(colors):
        raise RuntimeError(
            f"raw xyz/rgb length mismatch: {len(points)}, {len(colors)}"
        )
    if not len(points):
        return points, colors
    if np.all(points[1:, 0] >= points[:-1, 0]):
        start = int(np.searchsorted(points[:, 0], bbox[0], side="left"))
        end = int(np.searchsorted(points[:, 0], bbox[3], side="right"))
        candidate_xyz = points[start:end]
        candidate_rgb = colors[start:end]
    else:
        candidate_xyz = points
        candidate_rgb = colors
    inside = np.bitwise_and.reduce(
        (
            candidate_xyz[:, 0] >= bbox[0],
            candidate_xyz[:, 1] >= bbox[1],
            candidate_xyz[:, 2] >= bbox[2],
            candidate_xyz[:, 0] <= bbox[3],
            candidate_xyz[:, 1] <= bbox[4],
            candidate_xyz[:, 2] <= bbox[5],
        )
    )
    return candidate_xyz[inside], candidate_rgb[inside]


def render_dense_raw_cell(
    cell: Any,
    raw_scene_objects: Mapping[
        tuple[str, int], tuple[np.ndarray, np.ndarray]
    ],
    *,
    image_size: int = IMAGE_SIZE,
    allow_processed_empty_crop_fallback: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    bbox = np.asarray(cell.bbox_w, dtype=np.float64)
    image = np.full((image_size, image_size, 3), 255, dtype=np.uint8)
    object_groups: dict[tuple[str, int], list[Any]] = {}
    for obj in cell.objects:
        object_groups.setdefault(
            (str(obj.label).lower(), int(obj.instance_id)), []
        ).append(obj)

    object_raw: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    processed_cell_fallbacks: list[dict[str, Any]] = []
    for key, objects in object_groups.items():
        raw_xyz, raw_rgb = raw_scene_objects[key]
        label = key[0]
        if label not in STUFF_CLASSES:
            if len(objects) != 1:
                raise RuntimeError(
                    "Multiple non-stuff objects share one semantic/instance "
                    f"key in cell {cell.id}: {key}"
                )
            object_raw[id(objects[0])] = (raw_xyz, raw_rgb)
            continue

        cropped_xyz, cropped_rgb = _crop_raw_points_to_bbox(
            raw_xyz, raw_rgb, bbox
        )
        if not len(cropped_xyz):
            if not allow_processed_empty_crop_fallback:
                raise RuntimeError(
                    f"Raw stuff object {key} has no points in cell {cell.id}"
                )
            for obj in objects:
                normalized = np.asarray(obj.xyz, dtype=np.float64)
                world = (
                    normalized * float(cell.cell_size) + bbox[0:3]
                ).astype(np.float32)
                object_raw[id(obj)] = (
                    world,
                    _rgb_points_u8(obj.rgb),
                )
                processed_cell_fallbacks.append(
                    {
                        "cell_id": str(cell.id),
                        "label": key[0],
                        "instance_id": key[1],
                        "object_id": str(obj.id),
                        "reason": "raw_bbox_crop_empty",
                        "source": (
                            "original_processed_kitti360pose_cell_points"
                        ),
                        "point_count": len(world),
                    }
                )
            continue
        try:
            from sklearn.neighbors import KDTree
        except ImportError as error:
            raise RuntimeError(
                "Dense VLM-Loc rendering requires scikit-learn for the same "
                "nearest-cluster assignment used by the public source. Keep "
                "NumPy pinned below 2 for the existing PyTorch/PyG binaries."
            ) from error
        cluster_points = []
        cluster_ids = []
        for cluster_id, obj in enumerate(objects):
            normalized = np.asarray(obj.xyz, dtype=np.float64)
            world = normalized * float(cell.cell_size) + bbox[0:3]
            cluster_points.append(world[:, :2])
            cluster_ids.extend([cluster_id] * len(world))
        tree = KDTree(np.concatenate(cluster_points, axis=0))
        nearest = tree.query(
            cropped_xyz[:, :2], k=1, return_distance=False
        )[:, 0]
        assignments = np.asarray(cluster_ids, dtype=np.int64)[nearest]
        for cluster_id, obj in enumerate(objects):
            selected = assignments == cluster_id
            if not np.any(selected):
                if not allow_processed_empty_crop_fallback:
                    raise RuntimeError(
                        f"Raw cluster mapping produced no points for object "
                        f"{obj.id} in cell {cell.id}"
                    )
                normalized = np.asarray(obj.xyz, dtype=np.float64)
                world = (
                    normalized * float(cell.cell_size) + bbox[0:3]
                ).astype(np.float32)
                object_raw[id(obj)] = (
                    world,
                    _rgb_points_u8(obj.rgb),
                )
                processed_cell_fallbacks.append(
                    {
                        "cell_id": str(cell.id),
                        "label": key[0],
                        "instance_id": key[1],
                        "object_id": str(obj.id),
                        "reason": "raw_nearest_cluster_assignment_empty",
                        "source": (
                            "original_processed_kitti360pose_cell_points"
                        ),
                        "point_count": len(world),
                    }
                )
            else:
                object_raw[id(obj)] = (
                    cropped_xyz[selected],
                    cropped_rgb[selected],
                )

    entries = []
    dense_point_count = 0
    for obj in cell.objects:
        xyz, rgb = object_raw[id(obj)]
        linear = _world_points_to_linear_pixels(
            xyz, bbox, image_size
        )
        if not len(linear):
            continue
        entries.append(
            (
                str(obj.label).lower() in STUFF_CLASSES,
                linear,
                _rgb_u8(rgb),
                obj,
            )
        )
        dense_point_count += len(xyz)

    ordered = [entry for entry in entries if entry[0]]
    ordered.extend(entry for entry in entries if not entry[0])
    flat = image.reshape(-1, 3)
    nodes = []
    for node_id, (_, linear, color, obj) in enumerate(ordered):
        flat[linear] = color
        ys = (linear // image_size).astype(np.float64)
        xs = (linear % image_size).astype(np.float64)
        x_mean = float(np.mean(xs))
        y_mean = float(np.mean(ys))
        world_center = pixel_to_world_xy(
            (x_mean, y_mean), bbox, image_size=image_size
        )
        nodes.append(
            {
                "node_id": node_id,
                "label": str(obj.label).lower(),
                "pixel_center": [
                    int(round(x_mean)),
                    int(round(y_mean)),
                ],
                "world_center": [
                    float(world_center[0]),
                    float(world_center[1]),
                ],
                "object_id": str(obj.id),
                "instance_id": str(obj.instance_id),
            }
        )
    return image, {
        "nodes": nodes,
        "object_count": len(cell.objects),
        "rendered_object_count": len(entries),
        "dense_raw_point_count": dense_point_count,
        "raw_object_key_count": len(object_groups),
        "processed_cell_fallback_count": len(
            processed_cell_fallbacks
        ),
        "processed_cell_fallbacks": processed_cell_fallbacks,
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
    # Match the released ms-swift dataset items exactly: create_train_info()
    # adds one leading space to a user_message that already begins with one.
    return f"  The target location is {_join_phrases(phrases)}."


def _assignments(pose: Any, cell: Any, nodes: Sequence[dict[str, Any]]):
    del cell
    nodes_by_label: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        nodes_by_label.setdefault(node["label"], []).append(node)

    assignments = []
    for description in pose.descriptions:
        label = str(description.object_label).lower()
        candidates = nodes_by_label.get(label, [])
        node = None
        best_distance = math.inf
        object_center = getattr(description, "object_center", None)
        if candidates and object_center is not None:
            center_in_pose_cell = np.asarray(
                object_center, dtype=np.float64
            )[:2]
            object_center_world = (
                center_in_pose_cell * BEV_RANGE_M
                + np.asarray(pose.pose_w[:2], dtype=np.float64)
                - BEV_RANGE_M / 2.0
            )
            distances = [
                float(
                    np.linalg.norm(
                        np.asarray(
                            candidate["world_center"],
                            dtype=np.float64,
                        )
                        - object_center_world
                    )
                )
                for candidate in candidates
            ]
            best_index = int(np.argmin(distances))
            best_distance = distances[best_index]
            threshold = (
                50.0
                if label == "road"
                else 15.0
                if label in STUFF_CLASSES
                else 5.0
            )
            if best_distance <= threshold:
                node = candidates[best_index]
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
    raw_scene_objects: Mapping[
        tuple[str, int], tuple[np.ndarray, np.ndarray]
    ]
    | None = None,
    renderer_name: str | None = None,
    allow_processed_cell_fallback: bool = False,
) -> tuple[Path, dict[str, Any]]:
    scene_dir = image_root / str(cell.scene_name)
    scene_dir.mkdir(parents=True, exist_ok=True)
    image_path = scene_dir / f"{cell.id}.png"
    cache_path = scene_dir / f"{cell.id}.render.json"
    selected_renderer = renderer_name or (
        DENSE_RAW_RENDERER_NAME
        if raw_scene_objects is not None
        else RENDERER_NAME
    )
    if (
        not overwrite
        and image_path.is_file()
        and cache_path.is_file()
    ):
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            cached.get("schema_version") == 1
            and cached.get("cell_id") == str(cell.id)
            and cached.get("renderer") == selected_renderer
            and cached.get("image_size_px") == IMAGE_SIZE
            and isinstance(cached.get("render_info"), dict)
        ):
            with Image.open(image_path) as existing:
                if existing.size != (IMAGE_SIZE, IMAGE_SIZE):
                    raise RuntimeError(
                        f"cached image has wrong size: {image_path}"
                    )
            return image_path, cached["render_info"]
    if raw_scene_objects is None:
        image, info = render_processed_cell(cell)
        info["renderer"] = selected_renderer
    else:
        image, info = render_dense_raw_cell(
            cell,
            raw_scene_objects,
            allow_processed_empty_crop_fallback=(
                allow_processed_cell_fallback
            ),
        )
        info["renderer"] = selected_renderer
    if overwrite or not image_path.is_file():
        Image.fromarray(image, mode="RGB").save(image_path)
    else:
        with Image.open(image_path) as existing:
            if existing.size != (IMAGE_SIZE, IMAGE_SIZE):
                raise RuntimeError(
                    f"cached image has wrong size: {image_path}"
                )
    _write_json_atomic(
        cache_path,
        {
            "schema_version": 1,
            "cell_id": str(cell.id),
            "renderer": selected_renderer,
            "image_size_px": IMAGE_SIZE,
            "render_info": info,
        },
    )
    return image_path, info


def _prepare_gt_split(
    data_root: Path,
    scenes: Sequence[str],
    *,
    output_dir: Path,
    split_name: str,
    overwrite_images: bool,
    raw_kitti360_root: Path | None,
    allow_processed_missing_raw_fallback: bool,
) -> tuple[Path, dict[str, Any]]:
    renderer_name = (
        (
            HYBRID_RAW_RENDERER_NAME
            if allow_processed_missing_raw_fallback
            else DENSE_RAW_RENDERER_NAME
        )
        if raw_kitti360_root is not None
        else RENDERER_NAME
    )
    image_root = output_dir / "images" / renderer_name / split_name
    samples = []
    outside_count = 0
    cell_count = 0
    sample_index = 0
    raw_scene_audits = []
    processed_cell_fallbacks: list[dict[str, Any]] = []
    # Keep only one scene's point arrays in memory at a time. The JSON sample
    # order remains the declared split order.
    for scene in scenes:
        print(
            f"Preparing VLM inputs: split={split_name}, scene={scene}",
            flush=True,
        )
        dataset = load_ordered_dataset(data_root, [scene])
        cells = {str(cell.id): cell for cell in dataset.all_cells}
        raw_scene_objects = None
        if raw_kitti360_root is not None:
            raw_scene_objects, raw_scene_audit = (
                _load_raw_scene_objects(
                    raw_kitti360_root,
                    scene,
                    dataset.all_cells,
                    allow_processed_missing_raw_fallback=(
                        allow_processed_missing_raw_fallback
                    ),
                )
            )
            raw_scene_audits.append(raw_scene_audit)
            print(
                f"Raw object audit: scene={scene}, exact_raw_keys="
                f"{raw_scene_audit['exact_raw_object_key_count']}, "
                f"processed_fallback_keys="
                f"{raw_scene_audit['processed_fallback_object_key_count']}",
                flush=True,
            )
        rendered: dict[str, tuple[Path, dict[str, Any]]] = {}
        render_started = time.perf_counter()
        for cell_index, cell in enumerate(dataset.all_cells, start=1):
            render_result = _render_cell_once(
                cell,
                image_root=image_root,
                overwrite=overwrite_images,
                raw_scene_objects=raw_scene_objects,
                renderer_name=renderer_name,
                allow_processed_cell_fallback=(
                    allow_processed_missing_raw_fallback
                ),
            )
            rendered[str(cell.id)] = render_result
            cell_fallbacks = render_result[1].get(
                "processed_cell_fallbacks", []
            )
            if cell_fallbacks:
                processed_cell_fallbacks.extend(cell_fallbacks)
                print(
                    f"Audited per-cell processed fallback: "
                    f"cell={cell.id}, count={len(cell_fallbacks)}",
                    flush=True,
                )
            if (
                cell_index == 1
                or cell_index % 100 == 0
                or cell_index == len(dataset.all_cells)
            ):
                print(
                    f"Rendered {split_name} {scene}: "
                    f"{cell_index}/{len(dataset.all_cells)} cells, "
                    f"seconds={time.perf_counter() - render_started:.1f}",
                    flush=True,
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
        "raw_scene_audits": raw_scene_audits,
        "processed_cell_fallback_count": len(
            processed_cell_fallbacks
        ),
        "processed_cell_fallbacks": processed_cell_fallbacks,
    }
    return json_path, stats


def prepare_vlmloc_data(
    *,
    data_root: Path,
    manifest_path: Path,
    output_dir: Path,
    overwrite_images: bool = False,
    raw_kitti360_root: Path | None = None,
    require_dense_raw: bool = False,
    allow_processed_missing_raw_fallback: bool = False,
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
    raw_kitti360_root = (
        Path(raw_kitti360_root).resolve()
        if raw_kitti360_root is not None
        else None
    )
    _assert_separate_derived_output(
        data_root=data_root,
        output_dir=output_dir,
        raw_kitti360_root=raw_kitti360_root,
    )
    if require_dense_raw and raw_kitti360_root is None:
        raise RuntimeError(
            "--require-dense-raw was requested but "
            "--raw-kitti360-root was not provided."
        )
    source_snapshot_before = _source_tree_metadata_snapshot(data_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_path, training_stats = _prepare_gt_split(
        data_root,
        SCENE_NAMES_TRAIN,
        output_dir=output_dir,
        split_name="training",
        overwrite_images=overwrite_images,
        raw_kitti360_root=raw_kitti360_root,
        allow_processed_missing_raw_fallback=(
            allow_processed_missing_raw_fallback
        ),
    )
    validation_path, validation_stats = _prepare_gt_split(
        data_root,
        SCENE_NAMES_VAL,
        output_dir=output_dir,
        split_name="validation",
        overwrite_images=overwrite_images,
        raw_kitti360_root=raw_kitti360_root,
        allow_processed_missing_raw_fallback=(
            allow_processed_missing_raw_fallback
        ),
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
    selected_renderer_name = (
        (
            HYBRID_RAW_RENDERER_NAME
            if allow_processed_missing_raw_fallback
            else DENSE_RAW_RENDERER_NAME
        )
        if raw_kitti360_root is not None
        else RENDERER_NAME
    )
    image_root = (
        output_dir
        / "images"
        / selected_renderer_name
        / "testing_cmmloc_top1"
    )
    rendered = {}
    testing_raw_scene_audits = []
    testing_processed_cell_fallbacks: list[dict[str, Any]] = []
    for scene in SCENE_NAMES_TEST:
        print(
            "Preparing VLM inputs: split=testing_cmmloc_top1, "
            f"scene={scene}",
            flush=True,
        )
        scene_short = scene.split("_")[-2]
        scene_cell_ids = [
            cell_id
            for cell_id in unique_candidate_ids
            if str(cells[cell_id].scene_name) in {scene, scene_short}
        ]
        if not scene_cell_ids:
            continue
        raw_scene_objects = None
        if raw_kitti360_root is not None:
            raw_scene_objects, raw_scene_audit = (
                _load_raw_scene_objects(
                    raw_kitti360_root,
                    scene,
                    [cells[cell_id] for cell_id in scene_cell_ids],
                    allow_processed_missing_raw_fallback=(
                        allow_processed_missing_raw_fallback
                    ),
                )
            )
            testing_raw_scene_audits.append(raw_scene_audit)
            print(
                f"Raw object audit: scene={scene}, exact_raw_keys="
                f"{raw_scene_audit['exact_raw_object_key_count']}, "
                f"processed_fallback_keys="
                f"{raw_scene_audit['processed_fallback_object_key_count']}",
                flush=True,
            )
        render_started = time.perf_counter()
        for cell_index, cell_id in enumerate(scene_cell_ids, start=1):
            render_result = _render_cell_once(
                cells[cell_id],
                image_root=image_root,
                overwrite=overwrite_images,
                raw_scene_objects=raw_scene_objects,
                renderer_name=selected_renderer_name,
                allow_processed_cell_fallback=(
                    allow_processed_missing_raw_fallback
                ),
            )
            rendered[cell_id] = render_result
            cell_fallbacks = render_result[1].get(
                "processed_cell_fallbacks", []
            )
            if cell_fallbacks:
                testing_processed_cell_fallbacks.extend(cell_fallbacks)
                print(
                    f"Audited per-cell processed fallback: "
                    f"cell={cell_id}, count={len(cell_fallbacks)}",
                    flush=True,
                )
            if (
                cell_index == 1
                or cell_index % 100 == 0
                or cell_index == len(scene_cell_ids)
            ):
                print(
                    f"Rendered testing {scene}: "
                    f"{cell_index}/{len(scene_cell_ids)} candidate cells, "
                    f"seconds={time.perf_counter() - render_started:.1f}",
                    flush=True,
                )
    missing_rendered = sorted(set(unique_candidate_ids) - set(rendered))
    if missing_rendered:
        raise RuntimeError(
            "Candidate cells could not be associated with the declared "
            f"test scenes: {missing_rendered[:30]}"
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
                "renderer": render_info["renderer"],
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
    source_snapshot_after = _source_tree_metadata_snapshot(data_root)
    if source_snapshot_after != source_snapshot_before:
        raise RuntimeError(
            "KITTI360Pose source tree changed while preparing VLM-Loc "
            "derived inputs. The source dataset must remain immutable."
        )
    renderer_name = selected_renderer_name
    audit = {
        "schema_version": 1,
        "backend": "vlmloc_kitti360pose_30m_retrained",
        "comparison_scope": "table8_like_full_test_11505",
        "source_dataset_immutability": {
            "source_dataset_modified": False,
            "new_kitti360pose_samples_added": 0,
            "snapshot_before": source_snapshot_before,
            "snapshot_after": source_snapshot_after,
            "generated_files_are_derived_inputs_only": True,
            "derived_output_root": str(output_dir),
        },
        "renderer": renderer_name,
        "renderer_exactly_matches_public_dense_raw_renderer": False,
        "allow_processed_missing_raw_fallback": (
            allow_processed_missing_raw_fallback
        ),
        "renderer_reason": (
            (
                "Uses official KITTI-360 dense semantic/RGB points wherever "
                "the current raw release contains the exact semantic/instance "
                "key, and uses original processed KITTI360Pose object points "
                "only for explicitly listed raw-release gaps. Every fallback "
                "key and source cell is recorded in the split audits. This "
                "localized compatibility layer is not claimed to be the "
                "authors' unpublished exact renderer."
            )
            if renderer_name == HYBRID_RAW_RENDERER_NAME
            else (
                "Uses official KITTI-360 dense semantic/RGB points, public "
                "stuff-before-object rasterization, footprint centroids, and "
                "PNA thresholds while preserving the original ordered 30 m "
                "KITTI360Pose cells. Raw stuff points are deterministically "
                "mapped to the already-published cell clusters, so this is "
                "closer to the paper but is not claimed to be the authors' "
                "unreleased exact Table-8 generator."
            )
            if renderer_name == DENSE_RAW_RENDERER_NAME
            else (
                "The local processed KITTI360Pose cells contain normalized "
                "downsampled xyz/rgb arrays but no xyz_raw/rgb_raw fields. "
                "This fallback is not the closest-public-data renderer."
            )
        ),
        "uses_dense_raw_kitti360_points": raw_kitti360_root is not None,
        "raw_kitti360_root": (
            str(raw_kitti360_root) if raw_kitti360_root else None
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
            "raw_scene_audits": testing_raw_scene_audits,
            "processed_cell_fallback_count": len(
                testing_processed_cell_fallbacks
            ),
            "processed_cell_fallbacks": (
                testing_processed_cell_fallbacks
            ),
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
    require_dense_raw: bool = False,
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
        "training_num_epochs": 2.0,
        "training_learning_rate": 1e-4,
        "training_attention_implementation": "flash_attn",
        "training_per_device_batch_size": 1,
        "training_gradient_accumulation_steps": 4,
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
            and preparation.get("renderer") in SUPPORTED_RENDERERS
            and (
                not require_dense_raw
                or preparation.get("renderer")
                == DENSE_RAW_RENDERER_NAME
                or (
                    preparation.get("renderer")
                    == HYBRID_RAW_RENDERER_NAME
                    and preparation.get(
                        "allow_processed_missing_raw_fallback"
                    )
                    is True
                )
            )
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
        "require_dense_raw": require_dense_raw,
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
            (
                "This validates the separately retrained Table-8-like 30 m "
                "adapter with official dense KITTI-360 points plus the "
                "explicitly audited original-cell fallback only for raw "
                "release gaps. It is not the unavailable authors' exact "
                "Table-8 adapter or exact unpublished renderer."
            )
            if preparation.get("renderer") == HYBRID_RAW_RENDERER_NAME
            else (
                "This validates the separately retrained Table-8-like 30 m "
                "adapter with official dense KITTI-360 points on the original "
                "ordered KITTI360Pose cells. It is the closest audited public "
                "path, not the unavailable authors' exact Table-8 adapter."
            )
            if preparation.get("renderer") == DENSE_RAW_RENDERER_NAME
            else (
                "This validates the separately retrained Table-8-like 30 m "
                "adapter with the downsampled-point fallback, not the closest "
                "dense-raw path or unavailable exact Table-8 adapter."
            )
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


def audit_vlmloc_merged_checkpoint(
    *,
    adapter_dir: Path,
    base_model_dir: Path,
    merged_model_dir: Path,
    adapter_smoke_predictions_path: Path,
    merged_smoke_predictions_path: Path,
    output_dir: Path,
) -> Path:
    """Audit a standalone Qwen checkpoint produced by merging the LoRA."""

    adapter_dir = Path(adapter_dir).resolve()
    base_model_dir = Path(base_model_dir).resolve()
    merged_model_dir = Path(merged_model_dir).resolve()
    adapter_smoke_path = Path(adapter_smoke_predictions_path).resolve()
    merged_smoke_path = Path(merged_smoke_predictions_path).resolve()

    adapter = inspect_adapter(adapter_dir)
    base = inspect_base_model(base_model_dir)
    merged = inspect_full_model(merged_model_dir)
    merge_application = audit_merged_lora_application(
        adapter_dir=adapter_dir,
        base_model_dir=base_model_dir,
        merged_model_dir=merged_model_dir,
        sample_count=5,
    )
    adapter_rows = (
        _load_jsonl(adapter_smoke_path)
        if adapter_smoke_path.is_file()
        else []
    )
    merged_rows = (
        _load_jsonl(merged_smoke_path)
        if merged_smoke_path.is_file()
        else []
    )
    adapter_point = (
        _response_point(adapter_rows[0]) if len(adapter_rows) == 1 else None
    )
    merged_point = (
        _response_point(merged_rows[0]) if len(merged_rows) == 1 else None
    )
    tensor_namespace_matches = bool(
        base.get("base_tensor_key_sha256")
        and base.get("base_tensor_key_sha256")
        == merged.get("tensor_key_sha256")
        and base.get("base_tensor_key_count")
        == merged.get("header_tensor_key_count")
    )
    merged_inventory_complete = bool(
        merged.get("architecture_matches_qwen3_vl_8b")
        and int(merged.get("header_tensor_key_count") or 0) > 0
        and not merged.get("missing_required_inference_files", ["missing"])
        and not merged.get("missing_shards", ["missing"])
        and not merged.get("declared_but_missing_header_keys", ["missing"])
        and not merged.get("undeclared_header_keys", ["missing"])
        and not merged.get("wrong_shard_keys", ["missing"])
        and not merged.get("duplicate_header_keys", ["missing"])
        and not merged.get("invalid_header_records", ["missing"])
        and not merged.get("inventory_error")
    )
    checks = {
        "source_lora_adapter_structure": bool(
            adapter.get("adapter_config_exists")
            and adapter.get("adapter_weights_exists")
            and adapter.get("peft_type") == "LORA"
            and adapter.get("task_type") == "CAUSAL_LM"
            and not adapter.get("unpaired_lora_keys", ["missing"])
            and not adapter.get("internal_shape_mismatches", ["missing"])
        ),
        "base_model_complete_and_qwen3_vl_8b": bool(
            base.get("architecture_matches_qwen3_vl_8b")
            and base.get("safetensors_index_exists")
            and not base.get("missing_base_shards", ["missing"])
            and base.get("revision_marker_exists")
        ),
        "merged_full_checkpoint_inventory": merged_inventory_complete,
        "merged_tensor_namespace_matches_base": tensor_namespace_matches,
        "merged_has_no_adapter_or_lora_tensor_namespace": bool(
            not merged.get("adapter_artifacts_present", ["missing"])
            and not merged.get("lora_tensor_keys", ["missing"])
        ),
        "sampled_lora_targets_changed_from_base": bool(
            merge_application.get("compatible")
        ),
        "adapter_runtime_smoke": bool(
            len(adapter_rows) == 1 and adapter_point is not None
        ),
        "merged_runtime_smoke": bool(
            len(merged_rows) == 1 and merged_point is not None
        ),
        "merged_and_adapter_smoke_prediction_match": bool(
            adapter_point is not None
            and merged_point is not None
            and np.allclose(adapter_point, merged_point, atol=0.0, rtol=0.0)
        ),
    }
    report = {
        "schema_version": 1,
        "backend": "vlmloc_kitti360pose_30m_merged_full_checkpoint",
        "adapter_audit": adapter,
        "base_model_audit": base,
        "merged_model_audit": merged,
        "merge_application_audit": merge_application,
        "adapter_smoke_predictions_path": str(adapter_smoke_path),
        "adapter_smoke_predictions_sha256": (
            _sha256_file(adapter_smoke_path)
            if adapter_smoke_path.is_file()
            else None
        ),
        "merged_smoke_predictions_path": str(merged_smoke_path),
        "merged_smoke_predictions_sha256": (
            _sha256_file(merged_smoke_path)
            if merged_smoke_path.is_file()
            else None
        ),
        "adapter_smoke_point_2d": (
            list(adapter_point) if adapter_point is not None else None
        ),
        "merged_smoke_point_2d": (
            list(merged_point) if merged_point is not None else None
        ),
        "checks": checks,
        "compatible": all(checks.values()),
        "checkpoint_kind": (
            "standalone full-weight Hugging Face checkpoint for inference; "
            "the separate LoRA training checkpoint is retained for resuming "
            "training and provenance"
        ),
    }
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "vlmloc_merged_checkpoint_audit.json"
    _write_json_atomic(report_path, report)
    return report_path


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
    candidate_min_errors_m = []
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
        target_world = np.asarray(
            metadata["pose_world_xy"], dtype=np.float64
        )
        candidate_bbox = np.asarray(
            metadata["candidate_bbox_w"], dtype=np.float64
        )
        closest_in_candidate = np.clip(
            target_world,
            candidate_bbox[0:2],
            candidate_bbox[3:5],
        )
        candidate_min_error = float(
            np.linalg.norm(closest_in_candidate - target_world)
        )
        candidate_min_errors_m.append(candidate_min_error)
        if point is None:
            invalid += 1
        else:
            predicted_world = pixel_to_world_xy(
                point, metadata["candidate_bbox_w"]
            )
            error_m = float(np.linalg.norm(predicted_world - target_world))
            errors_m.append(error_m)
        per_query.append(
            {
                "query_index": index,
                "id": expected_id,
                "candidate_cell_id": metadata["candidate_cell_id"],
                "candidate_min_possible_error_m": candidate_min_error,
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
    paper_query_count = 11_404
    added_query_count = (
        total - paper_query_count if total >= paper_query_count else None
    )
    sample_count_only_envelope = (
        {
            threshold: {
                "approx_min_recall_if_all_added_queries_fail": (
                    reference * paper_query_count / total
                ),
                "approx_max_recall_if_all_added_queries_succeed": (
                    reference * paper_query_count + added_query_count
                )
                / total,
            }
            for threshold, reference in paper_reference.items()
        }
        if added_query_count is not None
        else None
    )
    renderers = sorted(
        {
            str(row.get("renderer", "unknown"))
            for row in test_index
        }
    )
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
        "candidate_geometry_upper_bound_recall": {
            str(threshold): sum(
                error <= threshold for error in candidate_min_errors_m
            )
            / total
            for threshold in (5, 10, 15)
        },
        "table8_reference_recall_not_exact_target": paper_reference,
        "table8_reference_query_count": paper_query_count,
        "additional_local_query_count": added_query_count,
        "sample_count_only_reference_envelope_approximate": (
            sample_count_only_envelope
        ),
        "sample_count_only_envelope_note": (
            "This narrow envelope applies only if the original 11,404 "
            "predictions are unchanged and the sole difference is appending "
            "101 queries. It does not apply when the checkpoint, renderer, "
            "CMMLoc candidates, or preprocessing differ."
        ),
        "renderers": renderers,
        "difference_from_table8_reference_percentage_points": {
            threshold: 100.0 * (metrics[threshold] - reference)
            for threshold, reference in paper_reference.items()
        },
        "comparison_warning": (
            "The reference uses the paper's 11,404-query/dense-raw-renderer "
            "setup. This result uses all 11,505 local queries and a separately "
            "retrained checkpoint. Even with dense raw points, closeness is a "
            "sanity check rather than an exact reproduction criterion."
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
