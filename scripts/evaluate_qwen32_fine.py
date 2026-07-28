"""Lightweight evaluation-only CLI for the public VLM-Loc 32B adapter.

Unlike ``evaluation.coarse_to_fine``, this entry point does not import the
CMMLoc/PyTorch-Geometric runtime. It can therefore run in a separate native
Windows Conda environment containing the newer Qwen3-VL inference stack.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.vlmloc_kitti360pose import (
    audit_public_qwen32_runtime_preflight,
    evaluate_vlmloc_predictions,
)


def _preflight(cli: argparse.Namespace) -> int:
    checkpoint_root = Path(cli.checkpoint_root).resolve()
    vlmloc_root = checkpoint_root / "VLM-Loc"
    report_path = audit_public_qwen32_runtime_preflight(
        adapter_dir=Path(cli.adapter_dir),
        base_model_dir=(
            vlmloc_root / "base_models" / "Qwen3-VL-32B-Instruct"
        ),
        official_source_dir=(
            vlmloc_root
            / "official_source"
            / "nku-3d-vision-494a8b4e3fe9"
        ),
        data_dir=Path(cli.vlmloc_data_dir),
        smoke_predictions_path=Path(cli.smoke_predictions),
        output_dir=Path(cli.output_dir),
        require_dense_raw=cli.require_dense_raw,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for name, passed in report["checks"].items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print(f"VLM-Loc public Qwen32 runtime preflight: {report_path}")
    return 0 if report["compatible"] else 2


def _score(cli: argparse.Namespace) -> int:
    result_path = evaluate_vlmloc_predictions(
        predictions_path=Path(cli.predictions),
        test_index_path=Path(cli.test_index),
        output_dir=Path(cli.output_dir),
        evaluation_mode="public_qwen32_cityloc_zero_shot",
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    print(
        "CMMLoc Top-1 -> public Qwen3-VL-32B fine R@5/10/15m: "
        f"{result['recall']['5']:.4f}/"
        f"{result['recall']['10']:.4f}/"
        f"{result['recall']['15']:.4f}"
    )
    print(
        f"Valid={result['valid_prediction_count']}, "
        f"invalid-as-miss={result['invalid_prediction_count']}"
    )
    print(f"Fine metrics: {result_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit and score evaluation-only public VLM-Loc "
            "Qwen3-VL-32B inference."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--checkpoint-root", required=True)
    preflight.add_argument("--adapter-dir", required=True)
    preflight.add_argument("--vlmloc-data-dir", required=True)
    preflight.add_argument("--smoke-predictions", required=True)
    preflight.add_argument("--output-dir", required=True)
    preflight.add_argument("--require-dense-raw", action="store_true")
    preflight.set_defaults(handler=_preflight)

    score = subparsers.add_parser("score")
    score.add_argument("--predictions", required=True)
    score.add_argument("--test-index", required=True)
    score.add_argument("--output-dir", required=True)
    score.set_defaults(handler=_score)
    return parser


def main() -> int:
    cli = build_parser().parse_args()
    return int(cli.handler(cli))


if __name__ == "__main__":
    raise SystemExit(main())
