#!/usr/bin/env bash
set -euo pipefail

# Table-8-like local experiment:
#   CMMLoc Top-1 coarse -> VLM-Loc Qwen3-VL-32B fine -> R@5/10/15 m
#
# The KITTI360Pose source tree is read-only by protocol. All generated files
# are written under evaluation_outputs or the dedicated VLM-Loc checkpoint
# subtree.

DATA_ROOT="${DATA_ROOT:-data/k360_30-10_scG_pd10_pc4_spY_all}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/k360_30-10_scG_pd10_pc4_spY_all}"
OUTPUT_ROOT="${OUTPUT_ROOT:-evaluation_outputs/cmmloc_top1_qwen32}"
SETUP_DIR="${SETUP_DIR:-${OUTPUT_ROOT}/setup}"
COARSE_DIR="${COARSE_DIR:-${OUTPUT_ROOT}/stage1_cmmloc}"
DERIVED_DIR="${DERIVED_DIR:-${OUTPUT_ROOT}/vlmloc_data_30m}"
PREFLIGHT_DIR="${PREFLIGHT_DIR:-${OUTPUT_ROOT}/stage2_preflight}"
INFERENCE_DIR="${INFERENCE_DIR:-${OUTPUT_ROOT}/stage2_inference}"
VLMLOC_ROOT="${CHECKPOINT_ROOT}/VLM-Loc"
QWEN_BASE="${QWEN_BASE:-${VLMLOC_ROOT}/base_models/Qwen3-VL-32B-Instruct}"
OFFICIAL_SOURCE="${OFFICIAL_SOURCE:-${VLMLOC_ROOT}/official_source/nku-3d-vision-494a8b4e3fe9/vlm-loc}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-${OFFICIAL_SOURCE}/system_prompt.txt}"
TRAIN_OUTPUT="${TRAIN_OUTPUT:-${VLMLOC_ROOT}/table8_kitti360pose_30m/qwen3_vl_32b_runs}"
FINAL_ADAPTER="${FINAL_ADAPTER:-${VLMLOC_ROOT}/table8_kitti360pose_30m/qwen3_vl_32b}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
GLOBAL_BATCH_SIZE=4

IFS=',' read -r -a GPU_IDS <<< "${GPU_LIST}"
TRAIN_WORLD_SIZE="${TRAIN_WORLD_SIZE:-${#GPU_IDS[@]}}"
if [[ "${TRAIN_WORLD_SIZE}" -ne "${#GPU_IDS[@]}" ]]; then
  echo "TRAIN_WORLD_SIZE must equal the number of IDs in GPU_LIST." >&2
  exit 2
fi
if [[ "${TRAIN_WORLD_SIZE}" -ne 1 && "${TRAIN_WORLD_SIZE}" -ne 2 && "${TRAIN_WORLD_SIZE}" -ne 4 ]]; then
  echo "Use 1, 2, or 4 training processes to preserve global batch size 4." >&2
  exit 2
fi
GRAD_ACCUMULATION=$((GLOBAL_BATCH_SIZE / TRAIN_WORLD_SIZE))

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 2
  fi
}

refuse_existing_file() {
  if [[ -e "$1" ]]; then
    echo "Refusing to append to or overwrite existing output: $1" >&2
    exit 2
  fi
}

setup_assets() {
  python scripts/setup_table8_step1_gpu.py \
    --checkpoint-root "${CHECKPOINT_ROOT}" \
    --output-dir "${SETUP_DIR}/cmmloc"
  python scripts/setup_vlmloc_gpu.py \
    --checkpoint-root "${CHECKPOINT_ROOT}" \
    --output-dir "${SETUP_DIR}/qwen32" \
    --table8-like-retraining
}

run_coarse() {
  python -m evaluation.coarse_to_fine table8-like-stage1 \
    --data-root "${DATA_ROOT}" \
    --checkpoint-root "${CHECKPOINT_ROOT}" \
    --text-backbone t5-large \
    --batch-size "${COARSE_BATCH_SIZE:-16}" \
    --device "${COARSE_DEVICE:-cuda:0}" \
    --output-dir "${COARSE_DIR}"
}

prepare_vlm_data() {
  require_file "${COARSE_DIR}/cmmloc_top1_manifest.json"
  python -m evaluation.coarse_to_fine table8-like-vlmloc-prepare \
    --data-root "${DATA_ROOT}" \
    --checkpoint-root "${CHECKPOINT_ROOT}" \
    --manifest "${COARSE_DIR}/cmmloc_top1_manifest.json" \
    --output-dir "${DERIVED_DIR}"
}

train_qwen32() {
  require_file "${DERIVED_DIR}/vlmloc_training_data.json"
  require_file "${DERIVED_DIR}/vlmloc_validation_data.json"
  require_file "${SYSTEM_PROMPT}"
  require_file "${QWEN_BASE}/config.json"
  if [[ -d "${TRAIN_OUTPUT}" ]] && [[ -n "$(find "${TRAIN_OUTPUT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to reuse non-empty TRAIN_OUTPUT: ${TRAIN_OUTPUT}" >&2
    echo "Set TRAIN_OUTPUT to a fresh directory or resume manually." >&2
    exit 2
  fi
  mkdir -p "${TRAIN_OUTPUT}"
  echo "Training world size: ${TRAIN_WORLD_SIZE}"
  echo "Per-device batch: 1"
  echo "Gradient accumulation: ${GRAD_ACCUMULATION}"
  echo "Effective global batch: ${GLOBAL_BATCH_SIZE}"
  CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
  NPROC_PER_NODE="${TRAIN_WORLD_SIZE}" \
  NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}" \
  NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  swift sft \
    --system "${SYSTEM_PROMPT}" \
    --model "${QWEN_BASE}" \
    --attn_impl flash_attn \
    --dataset "${DERIVED_DIR}/vlmloc_training_data.json" \
    --val_dataset "${DERIVED_DIR}/vlmloc_validation_data.json" \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --target_modules all-linear \
    --freeze_vit false \
    --freeze_aligner false \
    --gradient_accumulation_steps "${GRAD_ACCUMULATION}" \
    --eval_strategy steps \
    --eval_steps 300 \
    --save_steps 300 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 1 \
    --save_total_limit 5 \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing true \
    --deepspeed zero3 \
    --seed 42 \
    --data_seed 42 \
    --add_version false \
    --output_dir "${TRAIN_OUTPUT}"
}

finalize_adapter() {
  if [[ -z "${SELECTED_ADAPTER:-}" ]]; then
    echo "Set SELECTED_ADAPTER to the checkpoint with the lowest validation loss." >&2
    exit 2
  fi
  require_file "${SELECTED_ADAPTER}/adapter_model.safetensors"
  python scripts/finalize_qwen32_adapter.py \
    --checkpoint-root "${CHECKPOINT_ROOT}" \
    --adapter-dir "${SELECTED_ADAPTER}" \
    --vlmloc-data-dir "${DERIVED_DIR}" \
    --training-world-size "${TRAIN_WORLD_SIZE}" \
    --destination "${FINAL_ADAPTER}"
}

smoke_infer() {
  require_file "${FINAL_ADAPTER}/adapter_model.safetensors"
  require_file "${DERIVED_DIR}/vlmloc_testing_smoke_1.json"
  require_file "${SYSTEM_PROMPT}"
  mkdir -p "${PREFLIGHT_DIR}"
  local result="${PREFLIGHT_DIR}/adapter_smoke_predictions.jsonl"
  refuse_existing_file "${result}"
  CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  swift infer \
    --model "${QWEN_BASE}" \
    --adapters "${FINAL_ADAPTER}" \
    --infer_backend pt \
    --device_map auto \
    --val_dataset "${DERIVED_DIR}/vlmloc_testing_smoke_1.json" \
    --val_dataset_sample 1 \
    --dataset_shuffle false \
    --temperature 0 \
    --max_new_tokens 512 \
    --system "${SYSTEM_PROMPT}" \
    --result_path "${result}"
}

preflight() {
  require_file "${PREFLIGHT_DIR}/adapter_smoke_predictions.jsonl"
  python -m evaluation.coarse_to_fine table8-like-vlmloc-preflight \
    --checkpoint-root "${CHECKPOINT_ROOT}" \
    --adapter-dir "${FINAL_ADAPTER}" \
    --vlmloc-data-dir "${DERIVED_DIR}" \
    --smoke-predictions "${PREFLIGHT_DIR}/adapter_smoke_predictions.jsonl" \
    --output-dir "${PREFLIGHT_DIR}"
}

full_infer() {
  require_file "${PREFLIGHT_DIR}/vlmloc_runtime_preflight.json"
  python - "${PREFLIGHT_DIR}/vlmloc_runtime_preflight.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("compatible") is not True:
    raise SystemExit("Runtime preflight is not PASS; full inference is blocked.")
PY
  require_file "${DERIVED_DIR}/vlmloc_testing_data.json"
  mkdir -p "${INFERENCE_DIR}"
  local result="${INFERENCE_DIR}/predictions.jsonl"
  refuse_existing_file "${result}"
  CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  swift infer \
    --model "${QWEN_BASE}" \
    --adapters "${FINAL_ADAPTER}" \
    --infer_backend pt \
    --device_map auto \
    --val_dataset "${DERIVED_DIR}/vlmloc_testing_data.json" \
    --dataset_shuffle false \
    --max_batch_size 1 \
    --temperature 0 \
    --max_new_tokens 512 \
    --system "${SYSTEM_PROMPT}" \
    --result_path "${result}"
}

score() {
  require_file "${INFERENCE_DIR}/predictions.jsonl"
  require_file "${DERIVED_DIR}/vlmloc_testing_index.json"
  python -m evaluation.coarse_to_fine table8-like-vlmloc-evaluate \
    --predictions "${INFERENCE_DIR}/predictions.jsonl" \
    --test-index "${DERIVED_DIR}/vlmloc_testing_index.json" \
    --output-dir "${INFERENCE_DIR}"
}

usage() {
  cat <<'EOF'
Usage: scripts/run_cmmloc_qwen32_gpu.sh STAGE

Stages:
  setup       Download/audit pinned CMMLoc source and Qwen3-VL-32B assets
  coarse      Run CMMLoc baseline Top-1 and write the immutable hand-off
  prepare     Derive 30 m BEV/scene-graph JSON from original dataset cells
  train       Train Qwen3-VL-32B LoRA for 2 epochs (DeepSpeed ZeRO-3)
  finalize    Copy selected adapter and write provenance (set SELECTED_ADAPTER)
  smoke       Run deterministic one-query inference
  preflight   Audit source data, hashes, adapter, base model, and smoke output
  infer       Run fine inference for every original ordered test query
  score       Report world-coordinate R@5/10/15 m

Useful environment variables:
  GPU_LIST=0,1,2,3
  DATA_ROOT=...
  CHECKPOINT_ROOT=...
  SELECTED_ADAPTER=.../checkpoint-N
EOF
}

case "${1:-}" in
  setup) setup_assets ;;
  coarse) run_coarse ;;
  prepare) prepare_vlm_data ;;
  train) train_qwen32 ;;
  finalize) finalize_adapter ;;
  smoke) smoke_infer ;;
  preflight) preflight ;;
  infer) full_infer ;;
  score) score ;;
  *) usage; exit 2 ;;
esac
