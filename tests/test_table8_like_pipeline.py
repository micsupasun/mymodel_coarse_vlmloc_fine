from __future__ import annotations

import json
import shutil
import struct
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from evaluation.table8_reproduction import (
    CMMLOC_CHECKPOINT_ONLY_PREFIX_COUNTS,
)
from evaluation.table8_like_protocol import (
    load_table8_like_manifest,
    write_table8_like_manifest,
)
from evaluation.vlmloc_kitti360pose import (
    HYBRID_RAW_RENDERER_NAME,
    _assignments,
    _assert_separate_derived_output,
    _crop_raw_points_to_bbox,
    _group_selected_indices_by_packed_key,
    _pack_semantic_instance,
    _processed_fallback_objects,
    _query_message,
    _render_cell_once,
    _source_tree_metadata_snapshot,
    audit_vlmloc_merged_checkpoint,
    evaluate_vlmloc_predictions,
    load_ordered_dataset,
    normalized_xy_to_pixel,
    pixel_to_world_xy,
    render_dense_raw_cell,
    render_processed_cell,
    world_xy_to_pixel,
)
from evaluation.vlmloc_release import inspect_full_model


def _temporary_directory():
    root = Path.cwd() / "evaluation_outputs" / "_table8_like_tests"
    root.mkdir(parents=True, exist_ok=True)
    path = root / uuid.uuid4().hex
    path.mkdir()
    return root, path


def _description():
    return SimpleNamespace(
        direction="north",
        object_color_text="green",
        object_label="vegetation",
        object_center=np.asarray([0.5, 0.5, 0.0]),
        object_id="1",
        is_matched=True,
    )


def _write_safetensors(path, tensors):
    offset = 0
    header = {}
    payload = bytearray()
    for key, (shape, dtype, raw) in tensors.items():
        raw = bytes(raw)
        header[key] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        payload.extend(raw)
        offset += len(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _write_fake_full_model(directory, payload, *, revision=False):
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_vl",
                "architectures": ["Qwen3VLForConditionalGeneration"],
            }
        ),
        encoding="utf-8",
    )
    for name in (
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    ):
        (directory / name).write_text("{}", encoding="utf-8")
    key = "model.language_model.layers.0.self_attn.q_proj.weight"
    shard_name = "model-00001-of-00001.safetensors"
    _write_safetensors(
        directory / shard_name,
        {key: ([2, 2], "F32", payload)},
    )
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: shard_name}}),
        encoding="utf-8",
    )
    if revision:
        (directory / ".huggingface_model_revision").write_text(
            "a" * 40 + "\n", encoding="utf-8"
        )


def _write_fake_adapter(directory):
    directory.mkdir(parents=True)
    (directory / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 1,
                "lora_alpha": 2,
            }
        ),
        encoding="utf-8",
    )
    prefix = (
        "base_model.model.model.language_model.layers.0.self_attn.q_proj"
    )
    _write_safetensors(
        directory / "adapter_model.safetensors",
        {
            f"{prefix}.lora_A.weight": ([1, 2], "F32", b"\x01" * 8),
            f"{prefix}.lora_B.weight": ([2, 1], "F32", b"\x02" * 8),
        },
    )


def _dataset():
    scenes = [
        "2013_05_28_drive_0003_sync",
        "2013_05_28_drive_0005_sync",
        "2013_05_28_drive_0009_sync",
    ]
    cells = []
    poses = []
    for index, scene in enumerate(scenes[:2]):
        obj = SimpleNamespace(
            id=index + 1,
            instance_id=index + 11,
            label="vegetation",
            xyz=np.asarray([[0.25, 0.25, 0.0]], dtype=np.float32),
            rgb=np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
        )
        cell = SimpleNamespace(
            id=f"000{index}_00000",
            scene_name=scene,
            bbox_w=np.asarray([0, 0, 0, 30, 30, 5], dtype=np.float64),
            cell_size=30.0,
            objects=[obj],
        )
        pose = SimpleNamespace(
            scene_name=scene,
            cell_id=cell.id,
            pose_w=np.asarray([15.0, 15.0, 0.0]),
            descriptions=[_description()],
        )
        cells.append(cell)
        poses.append(pose)
    return SimpleNamespace(
        scene_names=scenes, all_cells=cells, all_poses=poses
    )


class Table8LikePipelineTests(unittest.TestCase):
    def test_grouped_raw_rows_match_pairwise_reference(self):
        packed = np.asarray([30, 10, 30, 20, 10, 40], dtype=np.int64)
        selected = np.asarray([0, 1, 2, 4, 5], dtype=np.int64)
        grouped = _group_selected_indices_by_packed_key(packed, selected)
        reference = {
            int(key): selected[packed[selected] == key]
            for key in np.unique(packed[selected])
        }
        self.assertEqual(set(grouped), set(reference))
        for key in reference:
            np.testing.assert_array_equal(grouped[key], reference[key])

    def test_x_sorted_raw_crop_matches_full_boolean_crop(self):
        xyz = np.asarray(
            [
                [35, 15, 2],
                [5, 5, 1],
                [25, 25, 3],
                [15, 15, 2],
                [20, 50, 2],
            ],
            dtype=np.float32,
        )
        rgb = np.arange(15, dtype=np.uint8).reshape(5, 3)
        order = np.argsort(xyz[:, 0], kind="stable")
        sorted_xyz, sorted_rgb = xyz[order], rgb[order]
        bbox = np.asarray([10, 10, 0, 30, 30, 5], dtype=np.float64)
        cropped_xyz, cropped_rgb = _crop_raw_points_to_bbox(
            sorted_xyz, sorted_rgb, bbox
        )
        reference_mask = np.bitwise_and.reduce(
            (
                sorted_xyz[:, 0] >= bbox[0],
                sorted_xyz[:, 1] >= bbox[1],
                sorted_xyz[:, 2] >= bbox[2],
                sorted_xyz[:, 0] <= bbox[3],
                sorted_xyz[:, 1] <= bbox[4],
                sorted_xyz[:, 2] <= bbox[5],
            )
        )
        np.testing.assert_array_equal(
            cropped_xyz, sorted_xyz[reference_mask]
        )
        np.testing.assert_array_equal(
            cropped_rgb, sorted_rgb[reference_mask]
        )

    def test_packed_raw_object_filter_matches_pairwise_reference(self):
        semantic = np.asarray([17, 17, 37, 41, 39, 7], dtype=np.int64)
        instance = np.asarray(
            [17004, 17346, 37015, 41083, 39108, 0],
            dtype=np.int64,
        )
        required_pairs = {(17, 17004), (41, 41083), (7, 0)}
        required_packed = _pack_semantic_instance(
            [pair[0] for pair in required_pairs],
            [pair[1] for pair in required_pairs],
        )
        optimized = np.isin(
            _pack_semantic_instance(semantic, instance),
            required_packed,
        )
        reference = np.asarray(
            [
                (int(semantic_id), int(instance_id)) in required_pairs
                for semantic_id, instance_id in zip(semantic, instance)
            ]
        )
        np.testing.assert_array_equal(optimized, reference)

    def test_packed_raw_object_filter_rejects_collision_range(self):
        with self.assertRaisesRegex(RuntimeError, "collision-free"):
            _pack_semantic_instance([17], [10_000_000])

    def test_missing_raw_compatibility_uses_only_original_cell_points(self):
        key = ("pole", 17004)
        object_a = SimpleNamespace(
            label=key[0],
            instance_id=key[1],
            xyz=np.asarray([[0.5, 0.5, 0.25]], dtype=np.float32),
            rgb=np.asarray([[0.5, 0.25, 0.0]], dtype=np.float32),
        )
        object_b = SimpleNamespace(
            label=key[0],
            instance_id=key[1],
            xyz=np.asarray([[1 / 6, 1 / 6, 0.25]], dtype=np.float32),
            rgb=np.asarray([[0.5, 0.25, 0.0]], dtype=np.float32),
        )
        cells = [
            SimpleNamespace(
                id="0000_00000",
                bbox_w=np.asarray([0, 0, 0, 30, 30, 30]),
                cell_size=30.0,
                objects=[object_a],
            ),
            SimpleNamespace(
                id="0000_00001",
                bbox_w=np.asarray([10, 10, 0, 40, 40, 30]),
                cell_size=30.0,
                objects=[object_b],
            ),
        ]
        recovered, audit = _processed_fallback_objects(cells, {key})
        xyz, rgb = recovered[key]
        np.testing.assert_allclose(xyz, [[15.0, 15.0, 7.5]])
        np.testing.assert_array_equal(rgb, [[128, 64, 0]])
        self.assertEqual(audit[0]["occurrence_count"], 2)
        self.assertEqual(
            audit[0]["source"],
            "original_processed_kitti360pose_cell_points",
        )
        self.assertIn(
            "audited_processed_missing_objects",
            HYBRID_RAW_RENDERER_NAME,
        )

    def test_missing_raw_compatibility_rejects_inconsistent_instances(self):
        key = ("pole", 17004)
        cells = []
        for index, x in enumerate((0.25, 0.75)):
            obj = SimpleNamespace(
                label=key[0],
                instance_id=key[1],
                xyz=np.asarray([[x, 0.5, 0.25]], dtype=np.float32),
                rgb=np.asarray([[0.5, 0.25, 0.0]], dtype=np.float32),
            )
            cells.append(
                SimpleNamespace(
                    id=f"0000_{index:05d}",
                    bbox_w=np.asarray([0, 0, 0, 30, 30, 30]),
                    cell_size=30.0,
                    objects=[obj],
                )
            )
        with self.assertRaisesRegex(RuntimeError, "not the same full object"):
            _processed_fallback_objects(cells, {key})

    def test_derived_output_cannot_overlap_source_dataset(self):
        root, directory = _temporary_directory()
        try:
            data_root = directory / "source_data"
            data_root.mkdir()
            with self.assertRaisesRegex(
                RuntimeError, "derived output must be separate"
            ):
                _assert_separate_derived_output(
                    data_root=data_root,
                    output_dir=data_root / "generated_vlm_inputs",
                    raw_kitti360_root=None,
                )
        finally:
            shutil.rmtree(directory)
            try:
                root.rmdir()
            except OSError:
                pass

    def test_source_snapshot_detects_changes_but_not_derived_outputs(self):
        root, directory = _temporary_directory()
        try:
            data_root = directory / "source_data"
            output_root = directory / "derived_inputs"
            data_root.mkdir()
            output_root.mkdir()
            (data_root / "pose.pkl").write_bytes(b"original")
            before = _source_tree_metadata_snapshot(data_root)
            (output_root / "bev.png").write_bytes(b"derived")
            after_output_write = _source_tree_metadata_snapshot(data_root)
            self.assertEqual(before, after_output_write)
            (data_root / "new_query.pkl").write_bytes(b"not-allowed")
            after_source_write = _source_tree_metadata_snapshot(data_root)
            self.assertNotEqual(before, after_source_write)
        finally:
            shutil.rmtree(directory)
            try:
                root.rmdir()
            except OSError:
                pass

    def test_cmmloc_release_allowlist_has_exact_public_counts(self):
        self.assertEqual(
            sum(CMMLOC_CHECKPOINT_ONLY_PREFIX_COUNTS.values()), 155
        )
        self.assertEqual(
            CMMLOC_CHECKPOINT_ONLY_PREFIX_COUNTS,
            {
                "cell_encoder2": 130,
                "modular_vector_mapping": 1,
                "obj_inter_module": 24,
            },
        )

    def test_top1_manifest_roundtrip_and_checksum(self):
        root, directory = _temporary_directory()
        try:
            dataset = _dataset()
            retrievals = [
                [dataset.all_cells[0].id],
                [dataset.all_cells[1].id],
            ]
            path = directory / "manifest.json"
            with patch(
                "evaluation.table8_like_protocol.QUERY_COUNT", 2
            ):
                written = write_table8_like_manifest(
                    path,
                    dataset=dataset,
                    retrievals=retrievals,
                    coarse_checkpoint=directory / "coarse.pth",
                    coarse_checkpoint_sha256="a" * 64,
                    coarse_audit_path=directory / "audit.json",
                    coarse_configuration={"backend": "cmmloc-release"},
                )
                loaded, loaded_retrievals = load_table8_like_manifest(
                    path, dataset=dataset
                )
            self.assertEqual(
                loaded["retrieval_rows_sha256"],
                written["retrieval_rows_sha256"],
            )
            self.assertEqual(loaded_retrievals, retrievals)
        finally:
            shutil.rmtree(directory)
            try:
                root.rmdir()
            except OSError:
                pass

    def test_manifest_tamper_is_rejected(self):
        root, directory = _temporary_directory()
        try:
            dataset = _dataset()
            path = directory / "manifest.json"
            with patch(
                "evaluation.table8_like_protocol.QUERY_COUNT", 2
            ):
                write_table8_like_manifest(
                    path,
                    dataset=dataset,
                    retrievals=[
                        [dataset.all_cells[0].id],
                        [dataset.all_cells[1].id],
                    ],
                    coarse_checkpoint=directory / "coarse.pth",
                    coarse_checkpoint_sha256="a" * 64,
                    coarse_audit_path=directory / "audit.json",
                    coarse_configuration={},
                )
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["rows"][0]["retrieved_cell_ids"] = [
                    dataset.all_cells[1].id
                ]
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "checksum"):
                    load_table8_like_manifest(path, dataset=dataset)
        finally:
            shutil.rmtree(directory)
            try:
                root.rmdir()
            except OSError:
                pass

    def test_30m_pixel_world_roundtrip_is_within_half_pixel(self):
        bbox = np.asarray([100, 200, 0, 130, 230, 5], dtype=np.float64)
        target = np.asarray([112.5, 219.25], dtype=np.float64)
        pixel, inside = world_xy_to_pixel(target, bbox)
        recovered = pixel_to_world_xy(pixel, bbox)
        self.assertTrue(inside)
        self.assertLessEqual(
            float(np.linalg.norm(recovered - target)),
            np.sqrt(2) * (30.0 / 224.0) / 2 + 1e-9,
        )
        self.assertEqual(normalized_xy_to_pixel([0.5, 0.5]), (111, 112))

    def test_renderer_uses_instance_over_stuff_draw_order(self):
        stuff = SimpleNamespace(
            id=1,
            instance_id=1,
            label="road",
            xyz=np.asarray([[0.5, 0.5, 0.0]], dtype=np.float32),
            rgb=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        )
        instance = SimpleNamespace(
            id=2,
            instance_id=2,
            label="pole",
            xyz=np.asarray([[0.5, 0.5, 0.0]], dtype=np.float32),
            rgb=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        )
        image, audit = render_processed_cell(
            SimpleNamespace(
                objects=[instance, stuff],
                bbox_w=np.asarray([0, 0, 0, 30, 30, 5]),
            )
        )
        x, y = normalized_xy_to_pixel([0.5, 0.5])
        self.assertEqual(image[y, x].tolist(), [0, 0, 255])
        self.assertEqual(
            [node["label"] for node in audit["nodes"]],
            ["road", "pole"],
        )
        self.assertEqual(
            audit["nodes"][0]["pixel_center"],
            [x, y],
        )
        self.assertEqual(audit["objects_with_raw_attributes"], 0)
        self.assertEqual(audit["downsampled_point_count"], 2)

    def test_dense_renderer_uses_raw_footprint_and_public_centroid(self):
        obj = SimpleNamespace(
            id=2,
            instance_id=17002,
            label="pole",
            xyz=np.asarray([[0.5, 0.5, 0.0]], dtype=np.float32),
            rgb=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        )
        cell = SimpleNamespace(
            id="0003_00000",
            objects=[obj],
            bbox_w=np.asarray([0, 0, 0, 30, 30, 5]),
            cell_size=30.0,
        )
        image, audit = render_dense_raw_cell(
            cell,
            {
                ("pole", 17002): (
                    np.asarray(
                        [[1, 1, 0], [29, 29, 0]], dtype=np.float32
                    ),
                    np.asarray(
                        [[255, 0, 0], [255, 0, 0]], dtype=np.uint8
                    ),
                )
            },
        )
        self.assertEqual(audit["dense_raw_point_count"], 2)
        self.assertEqual(audit["rendered_object_count"], 1)
        self.assertEqual(audit["nodes"][0]["pixel_center"], [112, 112])
        first = world_xy_to_pixel([1, 1], cell.bbox_w)[0]
        self.assertEqual(
            image[first[1], first[0]].tolist(), [255, 0, 0]
        )

    def test_dense_renderer_audits_empty_raw_cell_crop_fallback(self):
        obj = SimpleNamespace(
            id=6,
            instance_id=13000,
            label="fence",
            xyz=np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32),
            rgb=np.asarray([[0.25, 0.5, 0.75]], dtype=np.float32),
        )
        cell = SimpleNamespace(
            id="0000_1835",
            objects=[obj],
            bbox_w=np.asarray([0, 0, 0, 30, 30, 30]),
            cell_size=30.0,
        )
        raw_objects = {
            ("fence", 13000): (
                np.asarray([[100, 100, 100]], dtype=np.float32),
                np.asarray([[255, 255, 255]], dtype=np.uint8),
            )
        }
        with self.assertRaisesRegex(RuntimeError, "has no points"):
            render_dense_raw_cell(cell, raw_objects)
        image, audit = render_dense_raw_cell(
            cell,
            raw_objects,
            allow_processed_empty_crop_fallback=True,
        )
        self.assertEqual(audit["processed_cell_fallback_count"], 1)
        self.assertEqual(
            audit["processed_cell_fallbacks"][0]["reason"],
            "raw_bbox_crop_empty",
        )
        x, y = normalized_xy_to_pixel([0.5, 0.5])
        self.assertEqual(image[y, x].tolist(), [64, 128, 191])

    def test_render_sidecar_resumes_without_recomputing(self):
        root, directory = _temporary_directory()
        try:
            obj = SimpleNamespace(
                id=1,
                instance_id=17001,
                label="pole",
                xyz=np.asarray([[0.5, 0.5, 0.0]], dtype=np.float32),
                rgb=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            )
            cell = SimpleNamespace(
                id="0000_00001",
                scene_name="0000",
                objects=[obj],
                bbox_w=np.asarray([0, 0, 0, 30, 30, 30]),
                cell_size=30.0,
            )
            first_path, first_info = _render_cell_once(
                cell,
                image_root=directory / "images",
                overwrite=False,
            )
            with patch(
                "evaluation.vlmloc_kitti360pose.render_processed_cell",
                side_effect=RuntimeError("must not recompute"),
            ):
                second_path, second_info = _render_cell_once(
                    cell,
                    image_root=directory / "images",
                    overwrite=False,
                )
            self.assertEqual(first_path, second_path)
            self.assertEqual(first_info, second_info)
            self.assertTrue(
                first_path.with_name(
                    f"{cell.id}.render.json"
                ).is_file()
            )
        finally:
            shutil.rmtree(directory)
            try:
                root.rmdir()
            except OSError:
                pass

    def test_partial_node_assignment_uses_public_distance_thresholds(self):
        pose = SimpleNamespace(
            pose_w=np.asarray([100.0, 100.0, 0.0]),
            descriptions=[
                SimpleNamespace(
                    object_label="vegetation",
                    object_center=np.asarray([0.5, 0.5, 0.0]),
                    object_id="does-not-control-grounding",
                ),
                SimpleNamespace(
                    object_label="pole",
                    object_center=np.asarray([0.5, 0.5, 0.0]),
                ),
            ],
        )
        nodes = [
            {
                "node_id": 0,
                "label": "vegetation",
                "world_center": [110.0, 100.0],
            },
            {
                "node_id": 1,
                "label": "vegetation",
                "world_center": [103.0, 100.0],
            },
            {
                "node_id": 2,
                "label": "pole",
                "world_center": [106.0, 100.0],
            },
        ]
        assignments = _assignments(pose, None, nodes)
        self.assertTrue(assignments[0]["grounded"])
        self.assertEqual(assignments[0]["matched_node"], 1)
        self.assertFalse(assignments[1]["grounded"])
        self.assertIsNone(assignments[1]["matched_node"])

    def test_query_format_matches_released_double_space_after_image(self):
        pose = SimpleNamespace(descriptions=[_description()])
        self.assertEqual(
            f"<image>{_query_message(pose)}",
            "<image>  The target location is north of a green vegetation.",
        )

    def test_ordered_loader_accepts_original_short_scene_metadata(self):
        root, directory = _temporary_directory()
        try:
            scene = "2013_05_28_drive_0003_sync"
            (directory / "poses").mkdir()
            (directory / "cells").mkdir()
            (directory / "poses" / f"{scene}.pkl").touch()
            (directory / "cells" / f"{scene}.pkl").touch()
            pose = SimpleNamespace(scene_name="0003")
            cell = SimpleNamespace(
                scene_name="0003", id="0003_00000"
            )
            with patch(
                "evaluation.vlmloc_kitti360pose._load_pickle",
                side_effect=[[pose], [cell]],
            ):
                dataset = load_ordered_dataset(directory, [scene])
            self.assertEqual(dataset.all_poses, [pose])
            self.assertEqual(dataset.all_cells, [cell])
        finally:
            shutil.rmtree(directory)
            try:
                root.rmdir()
            except OSError:
                pass

    def test_world_metric_counts_invalid_generation_as_miss(self):
        root, directory = _temporary_directory()
        try:
            index = [
                {
                    "id": "test_000000",
                    "candidate_cell_id": "0003_00000",
                    "candidate_bbox_w": [0, 0, 0, 30, 30, 5],
                    "pose_world_xy": [15, 15],
                },
                {
                    "id": "test_000001",
                    "candidate_cell_id": "0005_00000",
                    "candidate_bbox_w": [0, 0, 0, 30, 30, 5],
                    "pose_world_xy": [15, 15],
                },
            ]
            index_path = directory / "index.json"
            index_path.write_text(json.dumps(index), encoding="utf-8")
            predictions_path = directory / "predictions.jsonl"
            predictions_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "test_000000",
                                "response": json.dumps(
                                    {"point_2d": [111, 112]}
                                ),
                            }
                        ),
                        json.dumps(
                            {
                                "id": "test_000001",
                                "response": "not json",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with patch(
                "evaluation.vlmloc_kitti360pose.QUERY_COUNT", 2
            ):
                result_path = evaluate_vlmloc_predictions(
                    predictions_path=predictions_path,
                    test_index_path=index_path,
                    output_dir=directory / "metrics",
                )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["valid_prediction_count"], 1)
            self.assertEqual(result["invalid_prediction_count"], 1)
            self.assertEqual(result["recall"]["5"], 0.5)
            self.assertEqual(result["recall"]["10"], 0.5)
            self.assertEqual(result["recall"]["15"], 0.5)
            self.assertEqual(
                result["candidate_geometry_upper_bound_recall"]["5"],
                1.0,
            )
        finally:
            shutil.rmtree(directory)
            try:
                root.rmdir()
            except OSError:
                pass

    def test_merged_full_checkpoint_audit_proves_lora_application(self):
        root, directory = _temporary_directory()
        try:
            adapter = directory / "adapter"
            base = directory / "base"
            merged = directory / "merged"
            _write_fake_adapter(adapter)
            _write_fake_full_model(base, b"\x00" * 16, revision=True)
            _write_fake_full_model(merged, b"\x03" * 16)
            smoke_row = {
                "id": "test_000000",
                "response": json.dumps({"point_2d": [111, 112]}),
            }
            adapter_smoke = directory / "adapter_smoke.jsonl"
            merged_smoke = directory / "merged_smoke.jsonl"
            adapter_smoke.write_text(
                json.dumps(smoke_row) + "\n", encoding="utf-8"
            )
            merged_smoke.write_text(
                json.dumps(smoke_row) + "\n", encoding="utf-8"
            )
            report_path = audit_vlmloc_merged_checkpoint(
                adapter_dir=adapter,
                base_model_dir=base,
                merged_model_dir=merged,
                adapter_smoke_predictions_path=adapter_smoke,
                merged_smoke_predictions_path=merged_smoke,
                output_dir=directory / "audit",
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(report["compatible"])
            self.assertTrue(
                report["checks"]["sampled_lora_targets_changed_from_base"]
            )
            self.assertEqual(
                report["merged_model_audit"]["lora_tensor_keys"], []
            )
        finally:
            shutil.rmtree(directory)
            try:
                root.rmdir()
            except OSError:
                pass

    def test_full_checkpoint_inspection_rejects_adapter_artifacts(self):
        root, directory = _temporary_directory()
        try:
            merged = directory / "merged"
            _write_fake_full_model(merged, b"\x03" * 16)
            (merged / "adapter_config.json").write_text(
                "{}", encoding="utf-8"
            )
            audit = inspect_full_model(merged)
            self.assertEqual(
                audit["adapter_artifacts_present"], ["adapter_config.json"]
            )
            self.assertEqual(audit["lora_tensor_keys"], [])
            self.assertEqual(audit["header_tensor_key_count"], 1)
        finally:
            shutil.rmtree(directory)
            try:
                root.rmdir()
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main()
