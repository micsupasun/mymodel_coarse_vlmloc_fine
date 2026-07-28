# CMMLoc coarse + VLM-Loc fine

The active evaluation-only protocol is:

```text
CMMLoc baseline Top-1 -> public VLM-Loc Qwen3-VL-32B -> R@5/10/15 m
```

KITTI360Pose is treated as immutable source data. See
[EVALUATION_COARSE_TO_FINE.md](EVALUATION_COARSE_TO_FINE.md) for the audited
GPU workflow. On a Windows GPU computer, run every stage from Command Prompt
with `scripts\run_cmmloc_qwen32_gpu.cmd STAGE`. The launcher is native Windows
and does not use WSL or train a model. The released 32B adapter was trained on
CityLoc-K/50 m, so its unchanged KITTI360Pose/30 m result is labelled as a
cross-dataset zero-shot evaluation rather than an exact Table-8 reproduction.
