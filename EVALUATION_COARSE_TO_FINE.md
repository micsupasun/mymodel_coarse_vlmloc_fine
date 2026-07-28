# CMMLoc Top-1 -> public VLM-Loc Qwen3-VL-32B

This is an evaluation-only native Windows workflow:

1. CMMLoc baseline retrieves exactly one KITTI360Pose cell per test query.
2. The released VLM-Loc Qwen3-VL-32B adapter predicts a pixel in that cell.
3. The pixel is converted through the original 30 m candidate bounding box.
4. The final report contains world-coordinate R@5/10/15 m.

There is no training stage.

## Scientific scope

The downloadable Qwen3-VL-32B VLM-Loc adapter
(`output/v0-20251109-125010-qwen3_32b/checkpoint-3300`) was trained on
CityLoc-K/50 m. The unpublished Qwen3-VL-32B adapter retrained by the authors
for KITTI360Pose Table 8 is not in the public release.

Therefore this workflow reports:

```text
CMMLoc Top-1 on KITTI360Pose/30m
-> public CityLoc-K/50m VLM-Loc Qwen3-VL-32B adapter
-> cross-dataset zero-shot R@5/10/15m
```

It does not claim to reproduce Table 8. Table 8 is kept only as a protocol
reference (11,404 queries and 40.36/51.69/54.74). The local release contains
11,505 ordered test queries, and all 11,505 are evaluated.

## Dataset immutability

`data/k360_30-10_scG_pd10_pc4_spY_all` remains source-only:

- no pickle, split, pose, cell ID, bounding box, or query order is changed;
- no CityLoc sample is copied into KITTI360Pose;
- only test BEV images and inference JSON are generated;
- derived files live under
  `evaluation_outputs/cmmloc_top1_qwen32_public_zero_shot/vlmloc_data_30m`;
- preparation snapshots the source tree before and after and fails if it
  changes.

## Two native Windows Conda environments

Keep the existing `cmmloc_mncl` environment for CMMLoc. Its PyTorch 2.0
packages may be tied to the installed PyTorch-Geometric build.

Create a separate inference environment for Qwen3-VL-32B:

```bat
conda create -n vlmloc_qwen32 python=3.10 -y
conda activate vlmloc_qwen32
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install "ms-swift==3.12.6" "transformers==4.57.6" "qwen-vl-utils==0.0.14" decord accelerate peft pillow huggingface_hub psutil
```

DeepSpeed and FlashAttention are not required because this workflow performs
inference only. It uses PyTorch SDPA and `device_map=auto`.

The RTX 4000 Ada has insufficient VRAM to hold all 32B BF16 weights. The
launcher defaults to:

```bat
set VLM_MAX_MEMORY={0: "15GiB", "cpu": "49GiB"}
```

This places part of the real BF16 model in system RAM. It is expected to be
much slower than a multi-GPU run, but it does not quantize the weights.

## Run order in Windows Command Prompt

After pulling the latest repository changes, first use the dedicated inference
environment to validate it and download/audit all required assets:

```bat
conda activate vlmloc_qwen32
scripts\run_cmmloc_qwen32_gpu.cmd check
scripts\run_cmmloc_qwen32_gpu.cmd setup
```

Then use the existing CMMLoc environment for the coarse stage and input
preparation:

```bat
conda activate cmmloc_mncl
scripts\run_cmmloc_qwen32_gpu.cmd check-coarse
scripts\run_cmmloc_qwen32_gpu.cmd coarse
scripts\run_cmmloc_qwen32_gpu.cmd prepare
```

`prepare` creates test/smoke inputs only. It does not create train or
validation inputs.

Then switch to the dedicated 32B inference environment:

```bat
conda activate vlmloc_qwen32
scripts\run_cmmloc_qwen32_gpu.cmd smoke
scripts\run_cmmloc_qwen32_gpu.cmd preflight
scripts\run_cmmloc_qwen32_gpu.cmd infer
scripts\run_cmmloc_qwen32_gpu.cmd score
```

`setup` downloads and hashes the official CMMLoc checkpoint, pinned VLM-Loc
source, exact public Qwen32 adapter, and full Qwen3-VL-32B BF16 base weights.

Full inference is blocked unless the smoke prediction loads successfully and
the following pass:

- derived test hashes and KITTI360Pose immutability;
- exact released Qwen32 adapter hashes and internal LoRA shapes;
- complete Qwen3-VL-32B base shards and pinned revision;
- pinned VLM-Loc source and compatible inference packages.

## Outputs

- Coarse Top-1 handoff:
  `evaluation_outputs/cmmloc_top1_qwen32_public_zero_shot/stage1_cmmloc/cmmloc_top1_manifest.json`
- Data immutability audit:
  `evaluation_outputs/cmmloc_top1_qwen32_public_zero_shot/vlmloc_data_30m/vlmloc_data_preparation_audit.json`
- Runtime/model audit:
  `evaluation_outputs/cmmloc_top1_qwen32_public_zero_shot/stage2_preflight/vlmloc_runtime_preflight.json`
- Final R@5/10/15 m:
  `evaluation_outputs/cmmloc_top1_qwen32_public_zero_shot/stage2_inference/vlmloc_fine_metrics.json`

Malformed or out-of-range Qwen outputs count as misses. The fine stage never
retrieves a replacement cell or discards a query.
