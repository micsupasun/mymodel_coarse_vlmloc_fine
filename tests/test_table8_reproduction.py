from __future__ import annotations

import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

from evaluation.table8_reproduction import (
    CMMLOC_COARSE_CHECKPOINT_SHA256,
    EXPECTED_TEST_SCENES,
    audit_cmmloc_coarse_checkpoint,
    audit_cmmloc_public_source,
    audit_cmmloc_runtime_configuration,
    audit_table8_dataset,
    build_table8_preflight_report,
    validate_table8_protocol,
)


def _signature(query_count: int) -> dict[str, object]:
    return {
        "query_count": query_count,
        "ordered_scenes": list(EXPECTED_TEST_SCENES),
        "ordered_query_sha256": "a" * 64,
        "ordered_cell_sha256": "b" * 64,
        "scene_query_counts": {},
        "scene_cell_counts": {},
    }


@contextmanager
def workspace_temp_directory():
    root = Path.cwd() / "evaluation_outputs" / "_table8_tests"
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


class Table8ReproductionTests(unittest.TestCase):
    def test_protocol_is_cmmloc_top1_vlmloc_fine(self) -> None:
        result = validate_table8_protocol(
            split="test", seed=42, top_k=[1], thresholds_m=[5, 10, 15]
        )
        self.assertTrue(result["compatible"])
        self.assertEqual(result["coarse_backend"], "cmmloc")
        self.assertEqual(result["fine_backend"], "vlmloc")

    def test_modified_my_model_top_k_protocol_is_rejected(self) -> None:
        result = validate_table8_protocol(
            split="test",
            seed=42,
            top_k=[1, 3, 5, 10],
            thresholds_m=[5, 10, 15],
        )
        self.assertFalse(result["compatible"])
        self.assertIn("top_k", result["mismatches"])

    def test_local_11505_query_split_is_not_table8_split(self) -> None:
        result = audit_table8_dataset(
            data_root=Path("k360_30-10_scG_pd10_pc4_spY_all"),
            dataset_signature=_signature(11_505),
        )
        self.assertFalse(result["compatible"])
        self.assertEqual(
            result["mismatches"]["query_count"],
            {"expected": 11_404, "actual": 11_505},
        )

    def test_exact_table8_dataset_header_passes(self) -> None:
        result = audit_table8_dataset(
            data_root=Path("k360_30-10_scG_pd10_pc4_spY_all"),
            dataset_signature=_signature(11_404),
        )
        self.assertTrue(result["compatible"])
        self.assertEqual(result["mismatches"], {})

    def test_missing_cmmloc_source_fails_closed(self) -> None:
        with workspace_temp_directory() as directory:
            result = audit_cmmloc_public_source(directory)
        self.assertFalse(result["compatible"])
        self.assertEqual(len(result["missing_files"]), 2)

    def test_combined_report_never_authorizes_inference_with_missing_assets(
        self,
    ) -> None:
        with workspace_temp_directory() as root:
            report = build_table8_preflight_report(
                data_root=root / "k360_30-10_scG_pd10_pc4_spY_all",
                dataset_signature=_signature(11_404),
                cmmloc_coarse_checkpoint=root / "coarse.pth",
                cmmloc_source_root=root / "source",
                vlmloc_report={
                    "compatible": False,
                    "missing_keys": ["Table-8 adapter"],
                    "unexpected_keys": [],
                    "shape_mismatches": [],
                },
            )
        self.assertFalse(report["all_compatible"])
        self.assertFalse(report["inference_authorized"])
        self.assertFalse(report["safety"]["my_model_used"])
        self.assertFalse(
            report["safety"]["cross_architecture_checkpoint_load_attempted"]
        )

    def test_local_cmmloc_checkpoint_matches_official_artifact_when_present(
        self,
    ) -> None:
        checkpoint = (
            Path("checkpoints")
            / "k360_30-10_scG_pd10_pc4_spY_all"
            / "CMMLoc"
            / "coarse.pth"
        )
        if not checkpoint.is_file():
            self.skipTest("local CMMLoc checkpoint is not present")
        result = audit_cmmloc_coarse_checkpoint(checkpoint)
        self.assertTrue(result["compatible_with_official_artifact"])
        self.assertEqual(
            result["checkpoint_sha256"], CMMLOC_COARSE_CHECKPOINT_SHA256
        )
        self.assertEqual(result["unexpected_key_count"], 155)
        self.assertFalse(result["compatible_with_public_source"])

    def test_text_and_pointnet_release_gaps_are_not_guessed(self) -> None:
        result = audit_cmmloc_runtime_configuration(
            requested_text_backbone="t5-large",
            checkpoint_audit={
                "inferred_text_hidden_dim": 1024,
                "t5_large_hidden_dim_matches": True,
            },
            source_audit={"source_files_match_pinned_commit": True},
        )
        self.assertTrue(result["text_backbone"]["requested_name_matches"])
        self.assertTrue(
            result["text_backbone"]["checkpoint_hidden_dim_matches"]
        )
        self.assertFalse(
            result["text_backbone"]["exact_model_revision_recorded"]
        )
        self.assertFalse(
            result["pointnet"]["exact_pointnet_provenance_recorded"]
        )
        self.assertFalse(result["compatible"])

    def test_pinned_public_source_exposes_release_mismatch_when_present(
        self,
    ) -> None:
        source = (
            Path("evaluation_outputs") / "_source_audit" / "CMMLoc"
        )
        if not source.is_dir():
            self.skipTest("pinned CMMLoc source audit clone is not present")
        result = audit_cmmloc_public_source(source)
        self.assertTrue(result["source_files_match_pinned_commit"])
        self.assertTrue(result["public_pipeline_uses_strict_false"])
        self.assertTrue(
            all(
                result[
                    "checkpoint_modules_absent_from_public_constructor"
                ].values()
            )
        )
        self.assertFalse(result["compatible"])


if __name__ == "__main__":
    unittest.main()
