from __future__ import annotations

import json
import shutil
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
    evaluate_vlmloc_predictions,
    normalized_xy_to_pixel,
    pixel_to_world_xy,
    render_processed_cell,
    world_xy_to_pixel,
)


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
        object_id="1",
        is_matched=True,
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
            SimpleNamespace(objects=[instance, stuff])
        )
        x, y = normalized_xy_to_pixel([0.5, 0.5])
        self.assertEqual(image[y, x].tolist(), [0, 0, 255])
        self.assertEqual(audit["objects_with_raw_attributes"], 0)
        self.assertEqual(audit["downsampled_point_count"], 2)

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
        finally:
            shutil.rmtree(directory)
            try:
                root.rmdir()
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main()
