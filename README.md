# CMMLoc coarse + VLM-Loc fine

The active evaluation is:

```text
CMMLoc baseline Top-1 -> VLM-Loc Qwen3-VL-32B -> R@5/10/15 m
```

KITTI360Pose is treated as immutable source data. See
[EVALUATION_COARSE_TO_FINE.md](EVALUATION_COARSE_TO_FINE.md) for the audited
GPU workflow. On a Windows GPU computer, run every stage from Command Prompt
with `scripts\run_cmmloc_qwen32_gpu.cmd STAGE`. The launcher is native Windows
and does not use WSL.
