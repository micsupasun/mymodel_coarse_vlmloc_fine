"""Download and audit the publicly available assets for Table-8 step 1.

This script obtains the pinned CMMLoc/VLM-Loc sources and, if needed, the
official CMMLoc coarse checkpoint.  It deliberately does not download the
public CityLoc-K adapter or Qwen base model because the released adapter is not
the separately retrained KITTI360Pose 30 m adapter used by Table 8.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.table8_reproduction import (
    CMMLOC_COARSE_CHECKPOINT_SHA256,
    CMMLOC_SOURCE_COMMIT,
    CMMLOC_SOURCE_REPOSITORY,
    audit_cmmloc_coarse_checkpoint,
    audit_cmmloc_public_source,
)
from evaluation.vlmloc_release import (
    OFFICIAL_SOURCE_COMMIT as VLMLOC_SOURCE_COMMIT,
    default_vlmloc_paths,
    inspect_official_source,
)
from scripts.setup_vlmloc_gpu import _download_official_source


CMMLOC_COARSE_PUBLIC_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1htRvlORsyD4FcDD68lFkSaplXFQv-Dlq&export=download&confirm=t"
)
MIN_FREE_DOWNLOAD_BYTES = 512 * 1024**2


def _free_bytes(path: Path) -> int:
    existing = path.resolve()
    while not existing.exists():
        existing = existing.parent
    return shutil.disk_usage(existing).free


def _download_cmmloc_checkpoint(checkpoint_path: Path) -> None:
    checkpoint_path = checkpoint_path.resolve()
    if checkpoint_path.is_file():
        existing = audit_cmmloc_coarse_checkpoint(checkpoint_path)
        if existing["compatible_with_official_artifact"]:
            print(f"Official CMMLoc coarse checkpoint exists: {checkpoint_path}")
            return
        raise RuntimeError(
            "Refusing to overwrite a non-matching CMMLoc checkpoint: "
            f"{checkpoint_path}"
        )
    if _free_bytes(checkpoint_path) < MIN_FREE_DOWNLOAD_BYTES:
        raise RuntimeError(
            "At least 512 MiB free space is required to download/audit the "
            "CMMLoc coarse checkpoint."
        )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_suffix(".pth.download")
    if temporary.exists():
        raise RuntimeError(
            f"Refusing to overwrite an incomplete download: {temporary}"
        )
    try:
        with urllib.request.urlopen(CMMLOC_COARSE_PUBLIC_URL) as response:
            content_type = response.headers.get("Content-Type", "")
            if "text/html" in content_type.lower():
                raise RuntimeError(
                    "Google Drive returned HTML instead of coarse.pth."
                )
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        downloaded = audit_cmmloc_coarse_checkpoint(temporary)
        if not downloaded["compatible_with_official_artifact"]:
            raise RuntimeError(
                "Downloaded CMMLoc checkpoint failed hash/key audit: "
                f"{downloaded.get('official_artifact_mismatches')}"
            )
        os.replace(temporary, checkpoint_path)
    except Exception:
        if temporary.is_file():
            temporary.unlink()
        raise


def _download_cmmloc_source(source_root: Path) -> None:
    source_root = source_root.resolve()
    if source_root.is_dir():
        existing = audit_cmmloc_public_source(source_root)
        if existing.get("source_files_match_pinned_commit"):
            print(f"Pinned CMMLoc source exists: {source_root}")
            return
        raise RuntimeError(
            f"Refusing to overwrite unverified CMMLoc source: {source_root}"
        )
    source_root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            CMMLOC_SOURCE_REPOSITORY,
            str(source_root),
        ],
        check=True,
    )
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(source_root),
                "config",
                "core.autocrlf",
                "false",
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(source_root),
                "checkout",
                "--detach",
                CMMLOC_SOURCE_COMMIT,
            ],
            check=True,
        )
        audit = audit_cmmloc_public_source(source_root)
        if not audit.get("source_files_match_pinned_commit"):
            raise RuntimeError(
                "Downloaded CMMLoc source files do not match the pinned commit."
            )
    except Exception:
        # Keep the clone for forensic inspection; never silently replace it.
        raise


def _audit(
    *,
    checkpoint_root: Path,
    cmmloc_source_root: Path,
) -> dict[str, Any]:
    checkpoint_path = checkpoint_root / "CMMLoc" / "coarse.pth"
    vlmloc_root = checkpoint_root / "VLM-Loc"
    vlmloc_paths = default_vlmloc_paths(vlmloc_root)
    cmmloc_checkpoint = audit_cmmloc_coarse_checkpoint(checkpoint_path)
    cmmloc_source = audit_cmmloc_public_source(cmmloc_source_root)
    vlmloc_source = inspect_official_source(vlmloc_paths["official_source"])
    table8_adapter_present = all(
        (vlmloc_paths["table8_adapter"] / name).is_file()
        for name in (
            "adapter_config.json",
            "adapter_model.safetensors",
            "args.json",
        )
    )
    table8_provenance_present = vlmloc_paths["table8_provenance"].is_file()
    return {
        "schema_version": 1,
        "experiment": "CMMLoc coarse Top-1 + VLM-Loc fine",
        "cmmloc_coarse_checkpoint": cmmloc_checkpoint,
        "cmmloc_public_source": cmmloc_source,
        "vlmloc_public_source": vlmloc_source,
        "expected_vlmloc_source_commit": VLMLOC_SOURCE_COMMIT,
        "table8_adapter_expected_at": str(vlmloc_paths["table8_adapter"]),
        "table8_adapter_present": table8_adapter_present,
        "table8_provenance_expected_at": str(
            vlmloc_paths["table8_provenance"]
        ),
        "table8_provenance_present": table8_provenance_present,
        "cmmloc_checkpoint_official": cmmloc_checkpoint.get(
            "checkpoint_sha256"
        )
        == CMMLOC_COARSE_CHECKPOINT_SHA256,
        "available_public_assets_downloaded": bool(
            cmmloc_checkpoint.get("compatible_with_official_artifact")
            and cmmloc_source.get("source_files_match_pinned_commit")
            and vlmloc_source.get("source_commit_matches")
        ),
        "table8_step1_assets_ready": bool(
            cmmloc_checkpoint.get("compatible_with_public_source")
            and cmmloc_source.get("compatible")
            and table8_adapter_present
            and table8_provenance_present
        ),
        "important_distinction": (
            "The public VLM-Loc checkpoint is trained for CityLoc-K/50 m. It "
            "is not downloaded or substituted for the unpublished/retrained "
            "KITTI360Pose 30 m Table-8 adapter."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        default=r"checkpoints\k360_30-10_scG_pd10_pc4_spY_all",
    )
    parser.add_argument(
        "--output-dir",
        default=r"evaluation_outputs\table8_step1_setup",
    )
    parser.add_argument("--audit-only", action="store_true")
    return parser


def main() -> int:
    cli = build_parser().parse_args()
    checkpoint_root = Path(cli.checkpoint_root).resolve()
    output_dir = Path(cli.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cmmloc_source_root = (
        checkpoint_root
        / "CMMLoc"
        / "official_source"
        / f"CMMLoc-{CMMLOC_SOURCE_COMMIT[:12]}"
    )
    setup_errors: list[str] = []
    if not cli.audit_only:
        actions = (
            lambda: _download_cmmloc_checkpoint(
                checkpoint_root / "CMMLoc" / "coarse.pth"
            ),
            lambda: _download_cmmloc_source(cmmloc_source_root),
            lambda: _download_official_source(
                checkpoint_root / "VLM-Loc"
            ),
        )
        for action in actions:
            try:
                action()
            except Exception as error:
                setup_errors.append(f"{type(error).__name__}: {error}")

    report = _audit(
        checkpoint_root=checkpoint_root,
        cmmloc_source_root=cmmloc_source_root,
    )
    report["setup_errors"] = setup_errors
    report_path = output_dir / "table8_step1_setup_audit.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Table-8 step 1 setup audit: {report_path}")
    print(
        "available_public_assets_downloaded="
        f"{report['available_public_assets_downloaded']}"
    )
    print(f"table8_step1_assets_ready={report['table8_step1_assets_ready']}")
    if setup_errors:
        for error in setup_errors:
            print(error, file=sys.stderr)
        return 1
    if not report["table8_step1_assets_ready"]:
        print(
            "All obtainable official assets were audited, but exact Table-8 "
            "inference remains blocked; inspect the report.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
