"""Download and audit the Qwen3-VL-32B/VLM-Loc assets on the GPU PC.

The public 32B adapter was trained on CityLoc-K/50 m. It can be evaluated on
the unchanged local KITTI360Pose/30 m test set as a cross-dataset zero-shot
run, but that run must not be labelled as a Table-8 reproduction.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.vlmloc_release import (
    BASE_MODEL_REVISION_MARKER,
    DEFAULT_BASE_MODEL_RELATIVE,
    DEFAULT_OFFICIAL_SOURCE_RELATIVE,
    OFFICIAL_HF_DATASET,
    OFFICIAL_SOURCE_COMMIT,
    OFFICIAL_SOURCE_REPOSITORY,
    PUBLIC_QWEN32_ADAPTER_RELATIVE,
    PUBLIC_QWEN32_ADAPTER_SHA256,
    PUBLIC_RELEASE_FILES,
    QWEN3_VL_32B_MODEL_ID,
    default_vlmloc_paths,
    inspect_adapter,
    inspect_base_model,
    inspect_official_source,
    inspect_python_environment,
)


MIN_FREE_BYTES_FOR_BASE_MODEL = 75 * 1024**3


def _disk_free(path: Path) -> int:
    existing = path.resolve()
    while not existing.exists():
        existing = existing.parent
    return shutil.disk_usage(existing).free


def _require_free_space(path: Path, required: int, label: str) -> None:
    free = _disk_free(path)
    if free < required:
        raise RuntimeError(
            f"Not enough free disk space for {label}: "
            f"need at least {required / 1024**3:.2f} GiB, "
            f"have {free / 1024**3:.2f} GiB at {path.resolve()}."
        )


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            try:
                target.relative_to(destination)
            except ValueError as error:
                raise RuntimeError(
                    f"Unsafe path in {archive}: {member.name}"
                ) from error
        bundle.extractall(destination)


def _find_public_qwen32_adapter(root: Path) -> Path | None:
    exact = root / PUBLIC_QWEN32_ADAPTER_RELATIVE
    if (exact / "adapter_config.json").is_file():
        return exact
    matches = sorted(
        path.parent
        for path in root.glob("**/adapter_config.json")
        if "qwen3_32b" in str(path).lower()
        and path.parent.name == "checkpoint-3300"
    )
    return matches[0] if matches else None


def _download_public_adapter(vlmloc_root: Path) -> Path:
    existing = _find_public_qwen32_adapter(vlmloc_root)
    if existing is not None:
        print(f"Public CityLoc Qwen3-VL-32B adapter already exists: {existing}")
        return existing

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub is required. Install it in the GPU environment "
            "with: python -m pip install huggingface_hub"
        ) from error

    archive_size = PUBLIC_RELEASE_FILES["checkpoints.tar.gz"]
    _require_free_space(vlmloc_root, archive_size * 3, "VLM-Loc checkpoint archive")
    cache_dir = vlmloc_root / "_downloads"
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = Path(
        hf_hub_download(
            repo_id=OFFICIAL_HF_DATASET,
            repo_type="dataset",
            filename="checkpoints.tar.gz",
            local_dir=str(cache_dir),
        )
    )
    extract_dir = vlmloc_root / "_public_checkpoint_extract"
    extract_dir.mkdir(parents=True, exist_ok=True)
    _safe_extract_tar(archive, extract_dir)
    found = _find_public_qwen32_adapter(extract_dir)
    if found is None:
        raise RuntimeError(
            f"Qwen3-VL-32B checkpoint-3300 not found after extracting {archive}"
        )
    target = vlmloc_root / PUBLIC_QWEN32_ADAPTER_RELATIVE
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise RuntimeError(f"Refusing to overwrite existing adapter directory: {target}")
    shutil.copytree(found, target)
    return target


def _download_base_model(vlmloc_root: Path) -> Path:
    target = vlmloc_root / DEFAULT_BASE_MODEL_RELATIVE
    revision_marker = target / BASE_MODEL_REVISION_MARKER
    if (target / "config.json").is_file() and (
        target / "model.safetensors.index.json"
    ).is_file() and revision_marker.is_file():
        print(f"Qwen3-VL-32B base model already exists: {target}")
        return target
    if target.exists() and any(target.iterdir()):
        raise RuntimeError(
            "Refusing to stamp or overwrite a pre-existing Qwen base-model "
            f"directory without an immutable revision marker: {target}. "
            "Move it aside and rerun, or audit it manually."
        )
    _require_free_space(
        target, MIN_FREE_BYTES_FOR_BASE_MODEL, QWEN3_VL_32B_MODEL_ID
    )
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub is required. Install it in the GPU environment "
            "with: python -m pip install huggingface_hub"
        ) from error

    resolved_revision = HfApi().model_info(QWEN3_VL_32B_MODEL_ID).sha
    if not resolved_revision:
        raise RuntimeError(
            f"Could not resolve an immutable revision for "
            f"{QWEN3_VL_32B_MODEL_ID}."
        )
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=QWEN3_VL_32B_MODEL_ID,
        revision=resolved_revision,
        local_dir=str(target),
        allow_patterns=[
            "*.json",
            "*.txt",
            "*.safetensors",
            "*.model",
        ],
    )
    revision_marker.write_text(
        resolved_revision + "\n", encoding="utf-8"
    )
    return target


def _download_official_source(vlmloc_root: Path) -> Path:
    target = vlmloc_root / DEFAULT_OFFICIAL_SOURCE_RELATIVE
    marker = target / ".vlmloc_source_commit"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == (
        OFFICIAL_SOURCE_COMMIT
    ):
        print(f"Pinned official source already exists: {target}")
        return target
    if target.exists():
        raise RuntimeError(
            f"Refusing to overwrite unverified official source directory: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            OFFICIAL_SOURCE_REPOSITORY,
            str(target),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(target), "sparse-checkout", "init", "--cone"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(target), "sparse-checkout", "set", "vlm-loc"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(target), "checkout", "--detach", OFFICIAL_SOURCE_COMMIT],
        check=True,
    )
    marker.write_text(OFFICIAL_SOURCE_COMMIT + "\n", encoding="utf-8")
    return target


def _audit(vlmloc_root: Path) -> dict[str, Any]:
    paths = default_vlmloc_paths(vlmloc_root)
    public_adapter = _find_public_qwen32_adapter(vlmloc_root)
    public_adapter_audit = (
        inspect_adapter(public_adapter) if public_adapter else None
    )
    public_adapter_hashes_match = bool(
        public_adapter_audit
        and all(
            public_adapter_audit.get(key) == expected
            for key, expected in PUBLIC_QWEN32_ADAPTER_SHA256.items()
        )
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "vlmloc_root": str(vlmloc_root.resolve()),
        "official_hf_dataset": OFFICIAL_HF_DATASET,
        "official_source_repository": OFFICIAL_SOURCE_REPOSITORY,
        "official_source_commit": OFFICIAL_SOURCE_COMMIT,
        "base_model_id": QWEN3_VL_32B_MODEL_ID,
        "public_release_files": PUBLIC_RELEASE_FILES,
        "public_qwen32_adapter": public_adapter_audit,
        "expected_public_qwen32_adapter_sha256": (
            PUBLIC_QWEN32_ADAPTER_SHA256
        ),
        "public_qwen32_adapter_hashes_match": (
            public_adapter_hashes_match
        ),
        "base_model": inspect_base_model(paths["base_model"]),
        "official_source": inspect_official_source(paths["official_source"]),
        "python_environment": inspect_python_environment(),
        "table8_adapter_expected_at": str(paths["table8_adapter"]),
        "table8_provenance_expected_at": str(paths["table8_provenance"]),
        "table8_adapter_present": (
            paths["table8_adapter"] / "adapter_model.safetensors"
        ).is_file(),
        "table8_provenance_present": paths["table8_provenance"].is_file(),
        "important_distinction": (
            "The downloadable Qwen3-VL-32B checkpoint-3300 is the public "
            "CityLoc-K 50 m adapter. It is not the separately retrained "
            "30 m KITTI360Pose "
            "Table-8 adapter."
        ),
    }
    report["public_assets_ready"] = bool(
        report["public_qwen32_adapter"]
        and report["public_qwen32_adapter_hashes_match"]
        and report["base_model"].get("architecture_matches_qwen3_vl")
        and report["base_model"].get("configured_as_bfloat16")
        and not report["base_model"].get("quantization_config_present", True)
        and not report["base_model"].get("missing_base_shards", ["unknown"])
        and report["official_source"].get("source_commit_matches")
        and not report["official_source"].get("missing_required_source_files")
    )
    report["table8_like_retraining_assets_ready"] = bool(
        report["base_model"].get("architecture_matches_qwen3_vl")
        and not report["base_model"].get(
            "missing_base_shards", ["unknown"]
        )
        and report["base_model"].get("revision_marker_exists")
        and report["official_source"].get("source_commit_matches")
        and not report["official_source"].get(
            "missing_required_source_files"
        )
    )
    report["requested_table8_assets_ready"] = bool(
        report["public_assets_ready"]
        and report["table8_adapter_present"]
        and report["table8_provenance_present"]
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download/audit the VLM-Loc Qwen3-VL-32B assets on the GPU "
            "machine without confusing CityLoc-K with KITTI360Pose Table 8."
        )
    )
    parser.add_argument(
        "--checkpoint-root",
        default=r"checkpoints\k360_30-10_scG_pd10_pc4_spY_all",
    )
    parser.add_argument(
        "--output-dir",
        default=r"evaluation_outputs\vlmloc_public_setup",
    )
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--skip-base-model", action="store_true")
    parser.add_argument("--skip-public-adapter", action="store_true")
    parser.add_argument("--skip-official-source", action="store_true")
    parser.add_argument(
        "--table8-like-retraining",
        action="store_true",
        help=(
            "Download/audit only the pinned source and immutable Qwen base "
            "needed to retrain on local KITTI360Pose 30 m data. The CityLoc "
            "public adapter is skipped."
        ),
    )
    parser.add_argument(
        "--public-qwen32-evaluation",
        action="store_true",
        help=(
            "Download/audit the public CityLoc-K Qwen3-VL-32B adapter, "
            "pinned source, and BF16 base weights for evaluation only. "
            "Success requires public_assets_ready, not an unpublished "
            "KITTI360Pose/Table-8 adapter."
        ),
    )
    return parser


def main() -> int:
    cli = build_parser().parse_args()
    checkpoint_root = Path(cli.checkpoint_root).resolve()
    vlmloc_root = checkpoint_root / "VLM-Loc"
    output_dir = Path(cli.output_dir).resolve()
    vlmloc_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    error: str | None = None
    try:
        if not cli.audit_only:
            if not cli.skip_official_source:
                _download_official_source(vlmloc_root)
            if not cli.skip_public_adapter and not cli.table8_like_retraining:
                _download_public_adapter(vlmloc_root)
            if not cli.skip_base_model:
                _download_base_model(vlmloc_root)
    except Exception as caught:
        error = f"{type(caught).__name__}: {caught}"

    report = _audit(vlmloc_root)
    report["setup_error"] = error
    report_path = output_dir / "vlmloc_public_setup_audit.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"VLM-Loc public setup audit: {report_path}")
    print(f"public_assets_ready={report['public_assets_ready']}")
    print(
        "requested_table8_assets_ready="
        f"{report['requested_table8_assets_ready']}"
    )
    print(
        "table8_like_retraining_assets_ready="
        f"{report['table8_like_retraining_assets_ready']}"
    )
    if error:
        print(error, file=sys.stderr)
        return 1
    if (
        cli.table8_like_retraining
        and report["table8_like_retraining_assets_ready"]
    ):
        return 0
    if cli.public_qwen32_evaluation and report["public_assets_ready"]:
        return 0
    if not report["requested_table8_assets_ready"]:
        print(
            "Public downloads completed/audited, but the requested Table-8 "
            "KITTI360Pose 30 m adapter/provenance is not in the public release.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
