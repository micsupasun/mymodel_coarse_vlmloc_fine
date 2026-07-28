# CMMLoc Top-1 -> VLM-Loc Qwen3-VL-32B

This repository evaluates one fixed protocol:

1. CMMLoc baseline retrieves exactly one KITTI360Pose cell per test query.
2. VLM-Loc with Qwen3-VL-32B predicts a pixel in that exact candidate.
3. The pixel is converted through the candidate cell's original 30 m world
   bounding box.
4. The final report contains localization R@5/10/15 m.

Table 8 in VLM-Loc is the protocol reference. It uses the same CMMLoc Top-1
handoff, reports 11,404 samples, and reports 40.36/51.69/54.74 for VLM-Loc.
The local KITTI360Pose release contains 11,505 ordered test queries. This
experiment evaluates all local queries and never filters 101 samples merely to
match the paper count.

## Dataset immutability

`data/k360_30-10_scG_pd10_pc4_spY_all` is source-only:

- no pickle is rewritten;
- no split, pose, cell ID, bounding box, or query order is changed;
- no CityLoc-K 50 m sample or raw KITTI-360 point is mixed into the data;
- generated 224 px BEV images and ms-swift JSON live only under
  `evaluation_outputs/cmmloc_top1_qwen32/vlmloc_data_30m`;
- preparation records metadata snapshots before and after generation and
  fails if the source tree changes.

The released Qwen3-VL-32B `checkpoint-3300` is retained only as a CityLoc-K
sanity reference. It is not used as the KITTI360Pose fine checkpoint. The
32B adapter is retrained for two epochs on derived representations of the
unchanged 30 m KITTI360Pose cells.

## GPU environment

The supported user entry point is native Windows Command Prompt in the
activated `cmmloc_mncl` Conda environment. The `.cmd` launcher does not invoke
WSL, PowerShell, Slurm, or a Linux shell.

Install ms-swift and the Python dependencies into the active Windows
environment:

```bat
python -m pip install -U "ms-swift[llm]" pillow huggingface_hub
```

DeepSpeed must be installed using its official native Windows build procedure,
including the Visual C++ build tools. FlashAttention must be built or installed
for the exact Python, PyTorch, CUDA, and GPU combination in this Conda
environment. Do not use an unrelated wheel.

Verify the complete native Windows environment before downloading or training:

```bat
conda activate cmmloc_mncl
scripts\run_cmmloc_qwen32_gpu.cmd check
```

The check fails closed unless Windows can load NVIDIA, PyTorch CUDA/BF16,
ms-swift, DeepSpeed, and FlashAttention.

Qwen3-VL-32B BF16 is large. In CMD, select the GPUs:

```bat
set GPU_LIST=0,1,2,3
```

One, two, or four training processes are accepted. The launcher sets
`NPROC_PER_NODE` and adjusts gradient accumulation to preserve the paper's
global batch size of 4.

## Run order

Download and audit the pinned public sources and Qwen3-VL-32B base model:

```bat
scripts\run_cmmloc_qwen32_gpu.cmd setup
```

Run CMMLoc Top-1 on every ordered local test query:

```bat
scripts\run_cmmloc_qwen32_gpu.cmd coarse
```

The only handoff to the fine stage is:

```text
evaluation_outputs/cmmloc_top1_qwen32/stage1_cmmloc/cmmloc_top1_manifest.json
```

Create BEV images, scene graphs, training/validation JSON, test JSON, and the
world-coordinate test index without changing KITTI360Pose:

```bat
scripts\run_cmmloc_qwen32_gpu.cmd prepare
```

The preparation audit must print:

```text
Renderer: processed_cell_downsampled_points_v1
KITTI360Pose source modified: False
new KITTI360Pose samples added: 0
```

Train Qwen3-VL-32B LoRA:

```bat
scripts\run_cmmloc_qwen32_gpu.cmd train
```

Select the saved `checkpoint-N` with the lowest validation loss. Do not reuse
the CityLoc-K checkpoint number. Finalize the selected adapter and prove
global batch size 4, model identity, input hashes, source commit, and dataset
immutability:

```bat
set SELECTED_ADAPTER=checkpoints/k360_30-10_scG_pd10_pc4_spY_all/VLM-Loc/table8_kitti360pose_30m/qwen3_vl_32b_runs/checkpoint-N
scripts\run_cmmloc_qwen32_gpu.cmd finalize
```

Run one deterministic query and then the fail-closed preflight:

```bat
scripts\run_cmmloc_qwen32_gpu.cmd smoke
scripts\run_cmmloc_qwen32_gpu.cmd preflight
```

Full inference is blocked unless every preflight check passes:

```bat
scripts\run_cmmloc_qwen32_gpu.cmd infer
```

Score predictions in the world frame of each original CMMLoc candidate:

```bat
scripts\run_cmmloc_qwen32_gpu.cmd score
```

The final result is:

```text
evaluation_outputs/cmmloc_top1_qwen32/stage2_inference/vlmloc_fine_metrics.json
```

Malformed/out-of-range Qwen outputs count as misses. The metric command does
not retrieve again, replace candidates, or discard queries.

## Important output separation

- Coarse diagnostics:
  `stage1_cmmloc/cmmloc_coarse_metrics.json`
- Immutable Top-1 handoff:
  `stage1_cmmloc/cmmloc_top1_manifest.json`
- Derived VLM input audit:
  `vlmloc_data_30m/vlmloc_data_preparation_audit.json`
- Finalized 32B adapter provenance:
  `checkpoints/.../VLM-Loc/table8_kitti360pose_30m/qwen3_vl_32b/vlmloc_table8_provenance.json`
- Fine localization result:
  `stage2_inference/vlmloc_fine_metrics.json`

Only the last file is the requested CMMLoc Top-1 -> Qwen3-VL-32B
R@5/10/15 m result.
