# Coarse-to-fine KITTI360Pose evaluation

## Experiment order and current scope

The experiments must be run in this order:

1. **Current scope:** CMMLoc baseline coarse Top-1 + VLM-Loc fine, reproducing
   VLM-Loc Table 8.
2. `my_model` coarse + `my_model` fine.
3. `my_model` coarse + VLM-Loc fine.
4. `my_model` coarse + CMMLoc/MNCL fine.

Step 1 is a different protocol from the shared-`my_model` pipeline documented
later in this file. It must use:

- split: KITTI360Pose **test**;
- retrieval: CMMLoc baseline, **Top-1 only**;
- localization: VLM-Loc fine;
- metric: localization `R@5/10/15 m`;
- reported Table-8 sample count: **11,404**;
- reported VLM-Loc reference: **40.36/51.69/54.74**.

It must not construct or load `my_model`, and it must not report Top-3/5/10
retrieval results as Table-8 localization results.

### Step-1 setup and preflight

On the GPU machine, download/audit only the obtainable official assets:

```cmd
python scripts\setup_table8_step1_gpu.py --checkpoint-root "checkpoints\k360_30-10_scG_pd10_pc4_spY_all" --output-dir "evaluation_outputs\table8_step1_setup"
```

Then run the dedicated fail-closed preflight:

```cmd
python -m evaluation.coarse_to_fine table8-preflight --data-root "data\k360_30-10_scG_pd10_pc4_spY_all" --checkpoint-root "checkpoints\k360_30-10_scG_pd10_pc4_spY_all" --text-backbone "t5-large" --device "cuda:0" --output-dir "evaluation_outputs\table8_step1_preflight"
```

Both commands currently return exit code 2 after writing their JSON audit.
That result is expected from the public artifacts currently available; it is
not an instruction to bypass the checks. Inference is not exposed until every
required check passes.

Confirmed blockers:

1. The local `CMMLoc/coarse.pth` is byte-for-byte the official 118,308,613-byte
   file (SHA-256
   `5e14e158c3de1fc046d9b970ef1d06c6d4a98d55a1cfdd09f6d26dfc23076f85`).
   However, the pinned official constructor has no `cell_encoder2`,
   `obj_inter_module`, or `modular_vector_mapping`, totaling 155 checkpoint
   keys. The official evaluation script hides these as unexpected keys with
   `strict=False`; this pipeline reports them and refuses to guess the training
   architecture.
2. The local ordered test split has 11,505 queries, while Table 8 explicitly
   reports 11,404. The ordered query/cell fingerprints are written to the
   audit.
3. The downloadable VLM-Loc adapter/test assets are CityLoc-K/50 m artifacts.
   Table 8 states that VLM-Loc was separately retrained on KITTI360Pose, but
   that exact 30 m adapter, training provenance, and 11,404-item prepared test
   set are not in the public release.
4. The CMMLoc checkpoint tensors support a 1,024-dimensional T5-large family
   and the source exposes NLTK sentence splitting plus `FixedPoints(256)` and
   PointNet defaults. The release nevertheless records only `PATH_TO_T5` and
   `PATH_TO_POINTNET`, not the exact text-model revision or PointNet artifact
   provenance. The preflight reports these fields as unresolved.

The setup command downloads the pinned CMMLoc/VLM-Loc source and the official
CMMLoc coarse checkpoint if missing. It deliberately does not spend roughly
18+ GB downloading the CityLoc adapter/base-model combination, because those
files cannot satisfy the Table-8 preflight.

The two output directories above are separate. No stage-1 retrieval manifest
or fine predictions are written while step 1 is blocked.

## Modified shared-`my_model` evaluation

This repository now has a fail-closed two-stage evaluation entry point:

```text
python -m evaluation.coarse_to_fine
```

It follows the fairness principle used by VLM-Loc Table 8, but implements the
requested modified protocol:

- `my_model/coarse.pth` is the only retrieval model.
- Stage 1 writes exactly 10 ordered candidate cell IDs per ordered test query.
- Stage 2 reads those IDs from a checksummed manifest; it cannot retrieve again.
- All fine backends use seed 42 and the same `R@5/10/15 m` implementation.
- Results are reported at top-k `1, 3, 5, 10`.

Stage 1 reports two deliberately separate diagnostics:

- **Exact-cell Retrieval Recall@K**: the ground-truth cell ID occurs in the
  first K retrieved IDs. This is the coarse metric used by the local ablation
  table.
- **Cell-center localization R@5/10/15 m**: every candidate predicts its cell
  center. This is a distance-based baseline and must not be compared with
  exact-cell Retrieval Recall@K.

The coarse checkpoint contains a learned reranker head, but the historical
ablation table evaluates more than one inference policy with that checkpoint:

- `--coarse-rerank-mode learned_reranker` reproduces
  `v1.1_mncl_structured_rerank`.
- `--coarse-rerank-mode structured_rerank` reproduces
  `v1.4_structured_rerank`: rerank the base top-50 using
  `base_score + 0.6 * label_coverage + 1.0 * color_label_coverage`, without
  using the learned reranker head.
- `--coarse-rerank-mode none` keeps the base retrieval ranking.

The selected policy, top-N, learned-head flag, and all fixed weights are stored
inside the retrieval manifest and coarse metrics. When reporting v1.4, label
the coarse method as `my_model embeddings + fixed structured rerank (v1.4
policy)`, not as the learned `mncl_structured_rerank` inference result.

The paper's Table 8 is not this exact experiment. Table 8 uses shared **Top-1
CMMLoc retrieval**, reports 11,404 KITTI360Pose test samples, and reports only
`R@5/10/15`. The modified experiment therefore must not be labeled as a direct
reproduction of the Table 8 numbers.

## Evidence found before implementation

### Local dataset

The localized pickle compatibility loader maps only the historical
`datapreparation.kitti360.*` module name to
`datapreparation.kitti360pose.*`. It does not install a global alias or change
serialized values.

The checked local test ordering and counts are:

| Ordered scene | Queries | Cells |
| --- | ---: | ---: |
| `2013_05_28_drive_0003_sync` | 949 | 447 |
| `2013_05_28_drive_0005_sync` | 3,079 | 1,199 |
| `2013_05_28_drive_0009_sync` | 7,477 | 2,662 |
| Total | **11,505** | **4,308** |

This differs from Table 8's 11,404 samples. The pipeline records the local
ordered-query and ordered-cell SHA-256 values in every manifest and refuses a
stage-2 run if they differ.

### Checkpoint architecture evidence

Static inspection of tensor keys/shapes produced:

| Checkpoint | Tensor keys | Relevant top-level structure |
| --- | ---: | --- |
| `my_model/coarse.pth` | 307 | custom projections and 4-key trainable reranker |
| `CMMLoc/coarse.pth` | 446 | second cell encoder, object interaction module |
| `my_model/fine.pth` | 267 | CMMLoc-style `CrossMatch` |
| `CMMLoc/fine.pth` | 267 | same key/shape signature as local `CrossMatch` |
| `MNCL/fine.pth` | 260 | different object/language/cross-attention signature |

The CMMLoc source inspected at commit
`d49458963d4caeddf7e9169e0e6384cb8223e22c` specifies fixed T5-large,
fine dimension 128, 4 decoder heads, 2 decoder layers, and 256 PointNet points;
the runtime preflight records and verifies those values.

`CMMLoc/fine.pth` versus `MNCL/fine.pth` has 62 keys only in CMMLoc and 55
only in MNCL. MNCL's checkpoint uses
`language_encoder.attention.*`, `language_encoder.gpool.*`, and
`language_encoder.toare.*`; it does not use the public source's
`language_encoder.MSG.*` namespace. The public MNCL source inspected at commit
`11ea10e1658b38e53b2127f4ee55f9d4236d9f50` constructs `self.MSG =
LanguageMsgEncoder(...)`.

The VLM-Loc files are PEFT/LoRA `CAUSAL_LM` adapters, not PyTorch
`CrossMatch` weights. Their local test JSON has 6,109 items and its image paths
reference `k360_50-10_gridCells...`, whereas this evaluation uses the current
11,505-query, 30 m KITTI360Pose test set. The public VLM-Loc source inspected at
commit `494a8b4e3fe9226849697e11d85e70a98e071283` also configures the released
CityLoc-K generator with `BEV_RANGE = 50.0` and `IMAGE_SIZE = 224`.

The official Hugging Face release was queried before adding any downloader.
Its complete file list is:

| Public file | Bytes | Use in this evaluation |
| --- | ---: | --- |
| `checkpoints.tar.gz` | 901,513,025 | Contains the public CityLoc adapters; useful only as reference |
| `CityLoc-C.tar.gz` | 16,092,589,771 | Not used; different dataset |
| `CityLoc-K.tar.gz` | 13,213,169,068 | Not used; 50 m CityLoc-K protocol |
| `dataset_items.tar.gz` | 2,494,464 | Not used as the 30 m ordered test set |

`Qwen/Qwen3-VL-8B-Instruct` is a separate roughly 17.5 GB download. The local
public Qwen3-VL-8B adapter has 736 BF16 tensors, LoRA rank 8, alpha 16, and
SHA-256
`e4a6379f29ec9ffb4456fcff0472e2aace4addcad2a84baea0835de2fa5d315c`.
Its `args.json` records CityLoc data paths. These facts are read directly from
the safetensors header and checkpoint metadata; the adapter is not selected
merely because its directory contains `k360`.
The adapter model card records PEFT 0.11.1 and its model metadata requests
`transformers>=4.57.0.dev`, `qwen_vl_utils>=0.0.14`, and `decord`, but the
release does not record an exact ms-swift/Transformers/Torch environment.
The setup audit therefore reports installed versions and does not silently
install a guessed compatibility stack.

## Backend status

| Backend | Status | Reason |
| --- | --- | --- |
| my_model coarse | Implemented; GPU preflight required | Own architecture, audited own checkpoint |
| CMMLoc fine | Implemented; GPU preflight required | Separate `CrossMatch` backend; exact keys/shapes audited before load |
| MNCL fine | Blocked, fail-closed | Supplied checkpoint and public released source have incompatible language namespaces |
| VLM-Loc fine | Blocked, fail-closed | Supplied adapters/data are CityLoc-style assets; exact Table-8 KITTI360Pose fine source, rendering pipeline, matching base model, and dataset are absent |

MNCL and VLM-Loc are intentionally **not** loaded into CMMLoc's architecture.
Stage 2 performs all selected preflights before any fine inference. If even one
selected backend fails, none of them runs.

For my_model and CMMLoc, key/shape/config compatibility is verifiable, but the
`.pth` files do not embed a training Git commit. Structural compatibility alone
cannot prove that an unrecorded forward-semantic source change never occurred;
the GPU smoke/numerical checks remain required and their result must be kept
with the preflight reports.

To unblock the two backends, supply:

1. The exact MNCL training source revision that produced the flat
   `attention/gpool/toare` checkpoint namespaces, including its preprocessing,
   tokenizer/backbone, and PointNet configuration.
2. The exact VLM-Loc KITTI360Pose Table-8 adapter/checkpoint, corresponding base
   VLM, 30 m BEV/scene-graph generator, prompt/parser, and ordered 11,404-sample
   evaluation split—or a documented mapping to the local 11,505-query split.

## Public VLM-Loc setup on the GPU machine

The setup command downloads only artifacts relevant to a Qwen3-VL-8B audit:
the pinned official source, the public Qwen3-VL-8B adapter if it is absent, and
the Qwen3-VL-8B base model. It deliberately does not download the 29 GB
CityLoc-C/K data archives because they cannot be used as the requested 30 m
KITTI360Pose evaluation data.

Run this one-line command from Windows CMD on the GPU machine. At least 20 GiB
of free space is required:

```cmd
python scripts\setup_vlmloc_gpu.py --checkpoint-root "checkpoints\k360_30-10_scG_pd10_pc4_spY_all" --output-dir "evaluation_outputs\vlmloc_public_setup"
```

The command writes
`evaluation_outputs\vlmloc_public_setup\vlmloc_public_setup_audit.json`.
Exit code 2 is expected while the unpublished/retrained Table-8 adapter and its
provenance are absent. This is a truthful blocker, not a download failure.

The exact Table-8 artifacts, when obtained from the authors or reproduced with
documented training, must be placed separately at:

```text
checkpoints/k360_30-10_scG_pd10_pc4_spY_all/VLM-Loc/
└── table8_kitti360pose_30m/qwen3_vl_8b/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── args.json
    └── vlmloc_table8_provenance.json
```

The public CityLoc adapter remains under `output/.../checkpoint-3600` and is
never silently promoted into that directory.

The accepted v1.4 coarse result already produced on the GPU machine is:

```text
evaluation_outputs/stage1_structured_rerank_v1_4_batch1_probe/retrieval_manifest.json
```

Its exact-cell Recall@K is `0.407301 / 0.652586 / 0.732638 / 0.802173` for
K=`1/3/5/10`. Do not rerun or replace that manifest while resolving the
VLM-Loc fine backend.

## Checkpoint loading policy

Before loading, the runtime writes every:

- allowed missing key;
- forbidden missing key;
- unexpected key;
- tensor shape mismatch; and
- model/checkpoint prefix count.

The released coarse/fine checkpoints omit the frozen T5 model parameters.
Those omissions are explicitly enumerated under
`language_encoder.llm_model.*`. Only after the audit finds no forbidden
missing keys, no unexpected keys, and no shape mismatch does loading proceed.
The loader then asserts that the runtime return exactly matches the enumerated
frozen-backbone omissions. No incompatibility is silently ignored.

Every structurally compatible backend then runs a one-query forward smoke test
on the selected device. It verifies finite values, output shapes, query order,
and candidate-cell order. A failed smoke test changes the backend preflight to
FAIL. Full stage-2 inference starts only if all selected backends pass.
The coarse smoke test additionally runs the selected rerank policy over 50
real dataset cells. For `structured_rerank`, it compares the production
reranker output against an independent fixed-score reference and uses a model
sentinel that raises if the learned head is called. The audit records the
input/output IDs, whether ordering changed, whether the independent reference
matched, and whether the learned head was exercised.
The smoke tests save and restore Python, NumPy, CPU Torch, and CUDA RNG states,
so they do not change the stochastic `FixedPoints(256)` subsets used by the
real evaluation. Stage 1 otherwise preserves the original source protocol:
seed 42 is set once before dataset/model construction and is not reset
immediately before coarse inference.

Stage 2 rejects older manifests that do not contain the explicit coarse
policy, top-N, learned-head flag, and structured weights. This prevents a
v1.1 retrieval manifest from being silently used for a v1.4 comparison.

## Windows CMD commands on the GPU machine

Run from:

```text
C:\Users\zh932237\working\mymodel_coarse_vlmloc_fine
```

Preflight all requested backends (this should currently return exit code 2 and
write evidence for the two known blockers):

```cmd
python -m evaluation.coarse_to_fine preflight --data-root "data\k360_30-10_scG_pd10_pc4_spY_all" --checkpoint-root "checkpoints\k360_30-10_scG_pd10_pc4_spY_all" --text-backbone "t5-large" --output-dir "evaluation_outputs\preflight_all" --backends vlmloc cmmloc mncl
```

Policy-specific preflight for the currently runnable v1.4 + CMMLoc path:

```cmd
python -m evaluation.coarse_to_fine preflight --data-root "data\k360_30-10_scG_pd10_pc4_spY_all" --checkpoint-root "checkpoints\k360_30-10_scG_pd10_pc4_spY_all" --text-backbone "t5-large" --batch-size 16 --device "cuda:0" --coarse-rerank-mode structured_rerank --output-dir "evaluation_outputs\preflight_structured_rerank_v1_4_v2" --backends cmmloc
```

Stage 1 using the requested v1.4 fixed structured-rerank policy, after
`my_model` coarse preflight passes:

```cmd
python -m evaluation.coarse_to_fine stage1 --data-root "data\k360_30-10_scG_pd10_pc4_spY_all" --checkpoint-root "checkpoints\k360_30-10_scG_pd10_pc4_spY_all" --text-backbone "t5-large" --batch-size 16 --device "cuda:0" --coarse-rerank-mode structured_rerank --output-dir "evaluation_outputs\stage1_structured_rerank_v1_4"
```

Stage 2 for the currently verifiable CMMLoc backend:

```cmd
python -m evaluation.coarse_to_fine stage2 --data-root "data\k360_30-10_scG_pd10_pc4_spY_all" --checkpoint-root "checkpoints\k360_30-10_scG_pd10_pc4_spY_all" --text-backbone "t5-large" --manifest "evaluation_outputs\stage1_structured_rerank_v1_4\retrieval_manifest.json" --output-dir "evaluation_outputs\stage2_cmmloc_from_v1_4" --backends cmmloc
```

The preflight directory, stage-1 directory, and stage-2 directory are separate.
Within stage 2, each backend also gets a separate subdirectory.

## Validation performed on the CPU workspace

1. All modified/new Python files pass `py_compile` using the bundled Python
   3.12 runtime.
2. Legacy pickle remapping loads all three test scenes and reproduces the
   11,505-query/4,308-cell count.
3. Ordered query/cell fingerprints were generated and scene ordering was
   asserted.
4. Retrieval row construction asserted 10 unique known cell IDs per query; a
   synthetic row modification changed the retrieval SHA-256.
5. Independent checkpoint metadata inspection verified key counts, prefix
   signatures, cross-checkpoint key differences, and shared-key shapes.
6. VLM adapter configs/test JSON were checked for task type, external base-model
   paths, test count, and dataset path token.
7. Source search confirmed that unsafe direct `strict=False` calls in the old
   evaluation entry point were replaced by the audited loader.
8. One-query coarse/fine forward smoke tests are wired into GPU preflight and
   recorded in each backend report (execution remains pending on the GPU).

Not yet proven on this CPU machine:

- real PyTorch architecture construction;
- tokenizer/T5 weight resolution;
- CUDA inference;
- GPU memory/runtime behavior;
- numerical coarse or fine recall results.

Those checks are deliberately left to the GPU preflight and evaluation runs.
