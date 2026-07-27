"""Download and audit the official KITTI-360 files used by VLM-Loc BEV rendering.

The processed KITTI360Pose folder is never modified.  Archives are retained
under ``_downloads`` so interrupted downloads can be resumed and provenance
can be inspected later.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.vlmloc_kitti360pose import (
    SCENE_NAMES_TEST,
    SCENE_NAMES_TRAIN,
    SCENE_NAMES_VAL,
    _raw_semantic_scene_directory,
)


OFFICIAL_DOWNLOAD_PAGE = (
    "https://www.cvlibs.net/datasets/kitti-360/download.php"
)
ARCHIVES = {
    "data_3d_semantics.zip": {
        "url": (
            "https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/"
            "6489aabd632d115c4280b978b2dcf72cb0142ad9/"
            "data_3d_semantics.zip"
        ),
        "minimum_bytes": 10 * 1024**3,
    },
    "data_3d_semantics_test.zip": {
        "url": (
            "https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/"
            "6489aabd632d115c4280b978b2dcf72cb0142ad9/"
            "data_3d_semantics_test.zip"
        ),
        "minimum_bytes": 900 * 1024**2,
    },
    "data_poses.zip": {
        "url": (
            "https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/"
            "89a6bae3c8a6f789e12de4807fc1e8fdcf182cf4/"
            "data_poses.zip"
        ),
        "minimum_bytes": 5 * 1024**2,
    },
}
MINIMUM_FREE_BYTES = 35 * 1024**3


def _free_bytes(path: Path) -> int:
    existing = path.resolve()
    while not existing.exists():
        existing = existing.parent
    return shutil.disk_usage(existing).free


def _download_with_resume(url: str, target: Path) -> None:
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl is None:
        raise RuntimeError(
            "curl is required for resumable KITTI-360 downloads."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            curl,
            "--location",
            "--fail",
            "--retry",
            "5",
            "--continue-at",
            "-",
            "--output",
            str(target),
            url,
        ],
        check=True,
    )


def _validate_archive(path: Path, minimum_bytes: int) -> dict[str, Any]:
    exists = path.is_file()
    size = path.stat().st_size if exists else 0
    zip_ok = False
    zip_error = None
    if exists and size >= minimum_bytes:
        try:
            with zipfile.ZipFile(path) as archive:
                bad_member = archive.testzip()
            zip_ok = bad_member is None
            if bad_member:
                zip_error = f"CRC failure: {bad_member}"
        except Exception as error:
            zip_error = f"{type(error).__name__}: {error}"
    return {
        "path": str(path.resolve()),
        "exists": exists,
        "size_bytes": size,
        "minimum_bytes": minimum_bytes,
        "size_ok": size >= minimum_bytes,
        "zip_crc_ok": zip_ok,
        "zip_error": zip_error,
    }


def _safe_extract_zip(path: Path, destination: Path) -> None:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            try:
                target.relative_to(destination)
            except ValueError as error:
                raise RuntimeError(
                    f"Unsafe member path in {path}: {member.filename}"
                ) from error
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.is_file():
                if target.stat().st_size == member.file_size:
                    continue
                raise RuntimeError(
                    "Refusing to overwrite an extracted KITTI-360 file with "
                    f"a different size: {target}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)


def _audit(raw_root: Path, archive_dir: Path) -> dict[str, Any]:
    archive_audits = {
        name: _validate_archive(
            archive_dir / name, int(record["minimum_bytes"])
        )
        for name, record in ARCHIVES.items()
    }
    scene_audits = {}
    for scene in (
        *SCENE_NAMES_TRAIN,
        *SCENE_NAMES_VAL,
        *SCENE_NAMES_TEST,
    ):
        try:
            scene_dir = _raw_semantic_scene_directory(raw_root, scene)
            ply_files = sorted(scene_dir.glob("*.ply"))
            scene_audits[scene] = {
                "compatible": bool(ply_files),
                "directory": str(scene_dir),
                "ply_file_count": len(ply_files),
                "total_ply_bytes": sum(
                    path.stat().st_size for path in ply_files
                ),
            }
        except Exception as error:
            scene_audits[scene] = {
                "compatible": False,
                "error": f"{type(error).__name__}: {error}",
            }
    poses_candidates = (
        raw_root / "data_poses",
        raw_root / "data_poses" / "data_poses",
    )
    poses_root = next(
        (
            path
            for path in poses_candidates
            if path.is_dir()
        ),
        None,
    )
    return {
        "schema_version": 1,
        "official_download_page": OFFICIAL_DOWNLOAD_PAGE,
        "raw_root": str(raw_root.resolve()),
        "archives": archive_audits,
        "scenes": scene_audits,
        "poses_root": str(poses_root.resolve()) if poses_root else None,
        "archives_valid": all(
            record["size_ok"] and record["zip_crc_ok"]
            for record in archive_audits.values()
        ),
        "dense_renderer_inputs_ready": all(
            record["compatible"] for record in scene_audits.values()
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        default=r"data\KITTI-360-raw",
    )
    parser.add_argument(
        "--output-dir",
        default=r"evaluation_outputs\kitti360_raw_setup",
    )
    parser.add_argument("--audit-only", action="store_true")
    return parser


def main() -> int:
    cli = build_parser().parse_args()
    raw_root = Path(cli.raw_root).resolve()
    output_dir = Path(cli.output_dir).resolve()
    archive_dir = raw_root / "_downloads"
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_errors = []
    if not cli.audit_only:
        if _free_bytes(raw_root) < MINIMUM_FREE_BYTES:
            setup_errors.append(
                "Insufficient free space: keep at least 35 GiB free before "
                "downloading/extracting KITTI-360 semantic point clouds."
            )
        else:
            for name, record in ARCHIVES.items():
                archive_path = archive_dir / name
                try:
                    existing = _validate_archive(
                        archive_path, int(record["minimum_bytes"])
                    )
                    if not (
                        existing["size_ok"] and existing["zip_crc_ok"]
                    ):
                        _download_with_resume(
                            str(record["url"]), archive_path
                        )
                    validated = _validate_archive(
                        archive_path, int(record["minimum_bytes"])
                    )
                    if not (
                        validated["size_ok"]
                        and validated["zip_crc_ok"]
                    ):
                        raise RuntimeError(
                            f"Downloaded archive failed validation: "
                            f"{validated}"
                        )
                    _safe_extract_zip(archive_path, raw_root)
                except Exception as error:
                    setup_errors.append(
                        f"{name}: {type(error).__name__}: {error}"
                    )

    report = _audit(raw_root, archive_dir)
    report["setup_errors"] = setup_errors
    report_path = output_dir / "kitti360_raw_setup_audit.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"KITTI-360 raw setup audit: {report_path}")
    print(
        "dense_renderer_inputs_ready="
        f"{report['dense_renderer_inputs_ready']}"
    )
    if setup_errors:
        for error in setup_errors:
            print(error, file=sys.stderr)
        return 1
    return 0 if report["dense_renderer_inputs_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
