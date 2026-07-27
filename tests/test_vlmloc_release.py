from __future__ import annotations

import json
import shutil
import struct
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

from evaluation.vlmloc_release import (
    EXPECTED_DATASET_TOKEN,
    EXPECTED_TEST_SCENES,
    OFFICIAL_SOURCE_COMMIT,
    QWEN3_VL_8B_MODEL_ID,
    TABLE8_PROVENANCE_NAME,
    dataset_tokens,
    default_vlmloc_paths,
    inspect_python_environment,
    read_safetensors_header,
    sha256_file,
    validate_table8_artifact_hashes,
    validate_table8_provenance,
)


@contextmanager
def workspace_temp_directory():
    root = Path.cwd() / "evaluation_outputs" / "_vlmloc_release_tests"
    root.mkdir(parents=True, exist_ok=True)
    directory = root / uuid.uuid4().hex
    directory.mkdir()
    try:
        yield directory
    finally:
        shutil.rmtree(directory)
        try:
            root.rmdir()
        except OSError:
            pass


class VlmLocReleaseTests(unittest.TestCase):
    def test_dataset_tokens_are_normalized_and_deduplicated(self) -> None:
        value = {
            "windows": r"C:\data\k360_30-10_scG_pd10_pc4_spY_all\cells",
            "unix": "/data/k360_50-10_gridCells_pd10_pc2/images",
            "duplicate": "/x/k360_30-10_scG_pd10_pc4_spY_all/poses",
        }
        self.assertEqual(
            dataset_tokens(value),
            [
                "k360_30-10_scG_pd10_pc4_spY_all",
                "k360_50-10_gridCells_pd10_pc2",
            ],
        )

    def test_safetensors_header_is_read_without_torch(self) -> None:
        header = {
            "layer.lora_A.weight": {
                "dtype": "BF16",
                "shape": [8, 16],
                "data_offsets": [0, 256],
            }
        }
        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        with workspace_temp_directory() as directory:
            path = directory / "adapter_model.safetensors"
            path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(256))
            self.assertEqual(read_safetensors_header(path), header)

    def test_invalid_safetensors_header_is_rejected(self) -> None:
        with workspace_temp_directory() as directory:
            path = directory / "broken.safetensors"
            path.write_bytes(b"short")
            with self.assertRaisesRegex(ValueError, "too short"):
                read_safetensors_header(path)

    def test_missing_table8_provenance_fails_closed(self) -> None:
        with workspace_temp_directory() as directory:
            result = validate_table8_provenance(
                directory / TABLE8_PROVENANCE_NAME,
                current_query_count=11505,
                current_query_order_sha256="a" * 64,
                current_cell_order_sha256="b" * 64,
            )
            self.assertFalse(result["compatible"])
            self.assertFalse(result["exists"])

    def test_exact_table8_provenance_passes(self) -> None:
        with workspace_temp_directory() as directory:
            path = directory / TABLE8_PROVENANCE_NAME
            record = {
                "schema_version": 1,
                "backend": "vlmloc",
                "dataset_token": EXPECTED_DATASET_TOKEN,
                "split": "test",
                "query_count": 11505,
                "test_scenes": list(EXPECTED_TEST_SCENES),
                "cell_size_m": 30.0,
                "bev_range_m": 30.0,
                "image_size_px": 224,
                "base_model_id": QWEN3_VL_8B_MODEL_ID,
                "source_commit": OFFICIAL_SOURCE_COMMIT,
                "query_order_sha256": "a" * 64,
                "cell_order_sha256": "b" * 64,
                "adapter_config_sha256": "c" * 64,
                "adapter_weights_sha256": "d" * 64,
                "args_sha256": "1" * 64,
                "base_model_config_sha256": "2" * 64,
                "system_prompt_sha256": "e" * 64,
                "training_dataset_sha256": "f" * 64,
                "validation_dataset_sha256": "0" * 64,
            }
            path.write_text(json.dumps(record), encoding="utf-8")
            result = validate_table8_provenance(
                path,
                current_query_count=11505,
                current_query_order_sha256="a" * 64,
                current_cell_order_sha256="b" * 64,
            )
            self.assertTrue(result["compatible"])
            self.assertEqual(result["mismatches"], {})

    def test_query_order_mismatch_is_reported(self) -> None:
        with workspace_temp_directory() as directory:
            path = directory / TABLE8_PROVENANCE_NAME
            record = {
                "schema_version": 1,
                "backend": "vlmloc",
                "dataset_token": EXPECTED_DATASET_TOKEN,
                "split": "test",
                "query_count": 11505,
                "test_scenes": list(EXPECTED_TEST_SCENES),
                "cell_size_m": 30.0,
                "bev_range_m": 30.0,
                "image_size_px": 224,
                "base_model_id": QWEN3_VL_8B_MODEL_ID,
                "source_commit": OFFICIAL_SOURCE_COMMIT,
                "query_order_sha256": "9" * 64,
                "cell_order_sha256": "b" * 64,
                "adapter_config_sha256": "c" * 64,
                "adapter_weights_sha256": "d" * 64,
                "args_sha256": "1" * 64,
                "base_model_config_sha256": "2" * 64,
                "system_prompt_sha256": "e" * 64,
                "training_dataset_sha256": "f" * 64,
                "validation_dataset_sha256": "0" * 64,
            }
            path.write_text(json.dumps(record), encoding="utf-8")
            result = validate_table8_provenance(
                path,
                current_query_count=11505,
                current_query_order_sha256="a" * 64,
                current_cell_order_sha256="b" * 64,
            )
            self.assertFalse(result["compatible"])
            self.assertIn("query_order_sha256", result["mismatches"])

    def test_artifact_hashes_are_cross_checked(self) -> None:
        with workspace_temp_directory() as directory:
            source = directory / "source"
            source.mkdir()
            prompt = source / "system_prompt.txt"
            prompt.write_text("strict prompt", encoding="utf-8")
            train = directory / "train.json"
            train.write_text("[]", encoding="utf-8")
            validation = directory / "validation.json"
            validation.write_text("[]", encoding="utf-8")
            provenance_path = directory / TABLE8_PROVENANCE_NAME
            record = {
                "adapter_config_sha256": "a" * 64,
                "adapter_weights_sha256": "b" * 64,
                "args_sha256": "c" * 64,
                "base_model_config_sha256": "d" * 64,
                "system_prompt_sha256": sha256_file(prompt),
                "training_dataset_path": "train.json",
                "training_dataset_sha256": sha256_file(train),
                "validation_dataset_path": "validation.json",
                "validation_dataset_sha256": sha256_file(validation),
            }
            provenance_path.write_text(json.dumps(record), encoding="utf-8")
            audit = validate_table8_artifact_hashes(
                {
                    "record": record,
                    "provenance_path": str(provenance_path),
                },
                adapter_audit={
                    "adapter_config_sha256": "a" * 64,
                    "adapter_weights_sha256": "b" * 64,
                    "args_sha256": "c" * 64,
                },
                base_model_audit={"config_sha256": "d" * 64},
                official_source_audit={"vlmloc_source_root": str(source)},
            )
            self.assertTrue(audit["compatible"])

            bad_audit = validate_table8_artifact_hashes(
                {
                    "record": record,
                    "provenance_path": str(provenance_path),
                },
                adapter_audit={
                    "adapter_config_sha256": "9" * 64,
                    "adapter_weights_sha256": "b" * 64,
                    "args_sha256": "c" * 64,
                },
                base_model_audit={"config_sha256": "d" * 64},
                official_source_audit={"vlmloc_source_root": str(source)},
            )
            self.assertFalse(bad_audit["compatible"])
            self.assertIn("adapter_config_sha256", bad_audit["mismatches"])

    def test_public_and_table8_adapter_paths_are_separate(self) -> None:
        paths = default_vlmloc_paths(Path("VLM-Loc"))
        self.assertNotEqual(paths["public_adapter"], paths["table8_adapter"])
        self.assertIn("output", paths["public_adapter"].parts)
        self.assertIn("table8_kitti360pose_30m", paths["table8_adapter"].parts)

    def test_environment_audit_does_not_install_or_guess_versions(self) -> None:
        audit = inspect_python_environment()
        self.assertFalse(audit["automatic_environment_install_allowed"])
        self.assertFalse(audit["exact_ms_swift_version_recorded"])
        self.assertEqual(audit["checkpoint_recorded_peft_version"], "0.11.1")
        self.assertIn("transformers", audit["package_versions"])


if __name__ == "__main__":
    unittest.main()
