@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Native Windows CMD launcher:
rem   CMMLoc Top-1 coarse -> VLM-Loc Qwen3-VL-32B fine -> R@5/10/15 m
rem
rem Run this file from an activated Windows Conda environment. No WSL,
rem PowerShell, Slurm, or Linux shell is used.

if "%~1"=="" goto usage
set "STAGE=%~1"

pushd "%~dp0.."
if errorlevel 1 (
  echo ERROR: Cannot enter the repository root.
  exit /b 2
)

if not defined DATA_ROOT set "DATA_ROOT=data\k360_30-10_scG_pd10_pc4_spY_all"
if not defined CHECKPOINT_ROOT set "CHECKPOINT_ROOT=checkpoints\k360_30-10_scG_pd10_pc4_spY_all"
if not defined OUTPUT_ROOT set "OUTPUT_ROOT=evaluation_outputs\cmmloc_top1_qwen32"
if not defined SETUP_DIR set "SETUP_DIR=%OUTPUT_ROOT%\setup"
if not defined COARSE_DIR set "COARSE_DIR=%OUTPUT_ROOT%\stage1_cmmloc"
if not defined DERIVED_DIR set "DERIVED_DIR=%OUTPUT_ROOT%\vlmloc_data_30m"
if not defined PREFLIGHT_DIR set "PREFLIGHT_DIR=%OUTPUT_ROOT%\stage2_preflight"
if not defined INFERENCE_DIR set "INFERENCE_DIR=%OUTPUT_ROOT%\stage2_inference"
if not defined VLMLOC_ROOT set "VLMLOC_ROOT=%CHECKPOINT_ROOT%\VLM-Loc"
if not defined QWEN_BASE set "QWEN_BASE=%VLMLOC_ROOT%\base_models\Qwen3-VL-32B-Instruct"
if not defined OFFICIAL_SOURCE set "OFFICIAL_SOURCE=%VLMLOC_ROOT%\official_source\nku-3d-vision-494a8b4e3fe9\vlm-loc"
if not defined SYSTEM_PROMPT set "SYSTEM_PROMPT=%OFFICIAL_SOURCE%\system_prompt.txt"
if not defined TRAIN_OUTPUT set "TRAIN_OUTPUT=%VLMLOC_ROOT%\table8_kitti360pose_30m\qwen3_vl_32b_runs"
if not defined FINAL_ADAPTER set "FINAL_ADAPTER=%VLMLOC_ROOT%\table8_kitti360pose_30m\qwen3_vl_32b"
if not defined GPU_LIST set "GPU_LIST=0,1,2,3"
if not defined COARSE_BATCH_SIZE set "COARSE_BATCH_SIZE=16"
if not defined COARSE_DEVICE set "COARSE_DEVICE=cuda:0"
set "GLOBAL_BATCH_SIZE=4"

set "GPU_COUNT=0"
for %%G in (!GPU_LIST:,= !) do set /a GPU_COUNT+=1
if "!GPU_COUNT!"=="0" (
  echo ERROR: GPU_LIST is empty.
  popd
  exit /b 2
)
if not defined TRAIN_WORLD_SIZE set "TRAIN_WORLD_SIZE=!GPU_COUNT!"
if not "!TRAIN_WORLD_SIZE!"=="!GPU_COUNT!" (
  echo ERROR: TRAIN_WORLD_SIZE must equal the number of IDs in GPU_LIST.
  popd
  exit /b 2
)
if not "!TRAIN_WORLD_SIZE!"=="1" if not "!TRAIN_WORLD_SIZE!"=="2" if not "!TRAIN_WORLD_SIZE!"=="4" (
  echo ERROR: Use 1, 2, or 4 GPUs to preserve global batch size 4.
  popd
  exit /b 2
)
set /a GRAD_ACCUMULATION=GLOBAL_BATCH_SIZE/TRAIN_WORLD_SIZE

if /I "%STAGE%"=="check" goto check
if /I "%STAGE%"=="setup" goto setup
if /I "%STAGE%"=="coarse" goto coarse
if /I "%STAGE%"=="prepare" goto prepare
if /I "%STAGE%"=="train" goto train
if /I "%STAGE%"=="finalize" goto finalize
if /I "%STAGE%"=="smoke" goto smoke
if /I "%STAGE%"=="preflight" goto preflight
if /I "%STAGE%"=="infer" goto infer
if /I "%STAGE%"=="score" goto score
goto usage_from_root

:check
echo Checking native Windows GPU environment...
where nvidia-smi.exe >nul 2>nul
if errorlevel 1 (
  echo ERROR: nvidia-smi.exe is not on PATH.
  popd
  exit /b 2
)
nvidia-smi.exe -L
if errorlevel 1 (
  echo ERROR: The NVIDIA driver cannot enumerate GPUs.
  popd
  exit /b 2
)
where python.exe >nul 2>nul
if errorlevel 1 (
  echo ERROR: python.exe is not available in the active Conda environment.
  popd
  exit /b 2
)
where swift.exe >nul 2>nul
if errorlevel 1 (
  echo ERROR: swift.exe from ms-swift is not installed in this environment.
  popd
  exit /b 2
)
python -c "import torch; from packaging.version import Version; print('torch=', torch.__version__); print('torch_cuda=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available()); print('gpu_count=', torch.cuda.device_count()); assert Version(torch.__version__.split('+')[0]) >= Version('2.1.0'), 'PyTorch >= 2.1 is required'; print('bf16_supported=', torch.cuda.is_bf16_supported()); assert torch.cuda.is_available(), 'PyTorch CUDA is unavailable'; assert torch.cuda.is_bf16_supported(), 'BF16 is unsupported by this GPU/PyTorch build'"
if errorlevel 1 (
  echo ERROR: PyTorch CUDA/BF16 preflight failed.
  popd
  exit /b 2
)
python -c "import deepspeed; print('deepspeed=', deepspeed.__version__)"
if errorlevel 1 (
  echo ERROR: DeepSpeed is missing or its native Windows build cannot load.
  echo Build/install the official Windows wheel in this Conda environment.
  popd
  exit /b 2
)
python -c "import flash_attn; print('flash_attn=', flash_attn.__version__)"
if errorlevel 1 (
  echo ERROR: FlashAttention is missing or incompatible with this Windows CUDA/PyTorch build.
  popd
  exit /b 2
)
echo Native Windows CMD environment check passed.
echo GPU_LIST=!GPU_LIST!
echo TRAIN_WORLD_SIZE=!TRAIN_WORLD_SIZE!
echo Effective global batch=!GLOBAL_BATCH_SIZE!
popd
exit /b 0

:setup
python scripts\setup_table8_step1_gpu.py ^
  --checkpoint-root "%CHECKPOINT_ROOT%" ^
  --output-dir "%SETUP_DIR%\cmmloc"
if errorlevel 1 goto command_failed
python scripts\setup_vlmloc_gpu.py ^
  --checkpoint-root "%CHECKPOINT_ROOT%" ^
  --output-dir "%SETUP_DIR%\qwen32" ^
  --table8-like-retraining
if errorlevel 1 goto command_failed
popd
exit /b 0

:coarse
python -m evaluation.coarse_to_fine table8-like-stage1 ^
  --data-root "%DATA_ROOT%" ^
  --checkpoint-root "%CHECKPOINT_ROOT%" ^
  --text-backbone t5-large ^
  --batch-size "%COARSE_BATCH_SIZE%" ^
  --device "%COARSE_DEVICE%" ^
  --output-dir "%COARSE_DIR%"
if errorlevel 1 goto command_failed
popd
exit /b 0

:prepare
if not exist "%COARSE_DIR%\cmmloc_top1_manifest.json" (
  echo ERROR: Missing %COARSE_DIR%\cmmloc_top1_manifest.json
  popd
  exit /b 2
)
python -m evaluation.coarse_to_fine table8-like-vlmloc-prepare ^
  --data-root "%DATA_ROOT%" ^
  --checkpoint-root "%CHECKPOINT_ROOT%" ^
  --manifest "%COARSE_DIR%\cmmloc_top1_manifest.json" ^
  --output-dir "%DERIVED_DIR%"
if errorlevel 1 goto command_failed
popd
exit /b 0

:train
if not exist "%DERIVED_DIR%\vlmloc_training_data.json" (
  echo ERROR: Missing %DERIVED_DIR%\vlmloc_training_data.json
  popd
  exit /b 2
)
if not exist "%DERIVED_DIR%\vlmloc_validation_data.json" (
  echo ERROR: Missing %DERIVED_DIR%\vlmloc_validation_data.json
  popd
  exit /b 2
)
if not exist "%SYSTEM_PROMPT%" (
  echo ERROR: Missing %SYSTEM_PROMPT%
  popd
  exit /b 2
)
if not exist "%QWEN_BASE%\config.json" (
  echo ERROR: Missing %QWEN_BASE%\config.json
  popd
  exit /b 2
)
if exist "%TRAIN_OUTPUT%\" (
  for /f "delims=" %%F in ('dir /b /a "%TRAIN_OUTPUT%" 2^>nul') do (
    echo ERROR: Refusing to reuse non-empty TRAIN_OUTPUT: %TRAIN_OUTPUT%
    echo Set TRAIN_OUTPUT to a fresh directory or resume manually.
    popd
    exit /b 2
  )
)
if not exist "%TRAIN_OUTPUT%\" mkdir "%TRAIN_OUTPUT%"
if errorlevel 1 goto command_failed

set "CUDA_VISIBLE_DEVICES=%GPU_LIST%"
set "NPROC_PER_NODE=%TRAIN_WORLD_SIZE%"
if not defined PYTORCH_CUDA_ALLOC_CONF set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo Training world size: %TRAIN_WORLD_SIZE%
echo Per-device batch: 1
echo Gradient accumulation: %GRAD_ACCUMULATION%
echo Effective global batch: %GLOBAL_BATCH_SIZE%

swift sft ^
  --system "%SYSTEM_PROMPT%" ^
  --model "%QWEN_BASE%" ^
  --attn_impl flash_attn ^
  --dataset "%DERIVED_DIR%\vlmloc_training_data.json" ^
  --val_dataset "%DERIVED_DIR%\vlmloc_validation_data.json" ^
  --train_type lora ^
  --torch_dtype bfloat16 ^
  --num_train_epochs 2 ^
  --per_device_train_batch_size 1 ^
  --per_device_eval_batch_size 1 ^
  --learning_rate 1e-4 ^
  --lora_rank 8 ^
  --lora_alpha 16 ^
  --target_modules all-linear ^
  --freeze_vit false ^
  --freeze_aligner false ^
  --gradient_accumulation_steps "%GRAD_ACCUMULATION%" ^
  --eval_strategy steps ^
  --eval_steps 300 ^
  --save_steps 300 ^
  --warmup_ratio 0.05 ^
  --dataloader_num_workers 4 ^
  --dataset_num_proc 1 ^
  --save_total_limit 5 ^
  --gradient_checkpointing true ^
  --vit_gradient_checkpointing true ^
  --deepspeed zero3 ^
  --seed 42 ^
  --data_seed 42 ^
  --add_version false ^
  --output_dir "%TRAIN_OUTPUT%"
if errorlevel 1 goto command_failed
popd
exit /b 0

:finalize
if not defined SELECTED_ADAPTER (
  echo ERROR: Set SELECTED_ADAPTER to the checkpoint with the lowest validation loss.
  popd
  exit /b 2
)
if not exist "%SELECTED_ADAPTER%\adapter_model.safetensors" (
  echo ERROR: Missing %SELECTED_ADAPTER%\adapter_model.safetensors
  popd
  exit /b 2
)
python scripts\finalize_qwen32_adapter.py ^
  --checkpoint-root "%CHECKPOINT_ROOT%" ^
  --adapter-dir "%SELECTED_ADAPTER%" ^
  --vlmloc-data-dir "%DERIVED_DIR%" ^
  --training-world-size "%TRAIN_WORLD_SIZE%" ^
  --destination "%FINAL_ADAPTER%"
if errorlevel 1 goto command_failed
popd
exit /b 0

:smoke
if not exist "%FINAL_ADAPTER%\adapter_model.safetensors" (
  echo ERROR: Missing finalized adapter: %FINAL_ADAPTER%
  popd
  exit /b 2
)
if not exist "%DERIVED_DIR%\vlmloc_testing_smoke_1.json" (
  echo ERROR: Missing smoke dataset.
  popd
  exit /b 2
)
if not exist "%PREFLIGHT_DIR%\" mkdir "%PREFLIGHT_DIR%"
if exist "%PREFLIGHT_DIR%\adapter_smoke_predictions.jsonl" (
  echo ERROR: Refusing to overwrite existing smoke predictions.
  popd
  exit /b 2
)
set "CUDA_VISIBLE_DEVICES=%GPU_LIST%"
if not defined PYTORCH_CUDA_ALLOC_CONF set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
swift infer ^
  --model "%QWEN_BASE%" ^
  --adapters "%FINAL_ADAPTER%" ^
  --infer_backend pt ^
  --device_map auto ^
  --val_dataset "%DERIVED_DIR%\vlmloc_testing_smoke_1.json" ^
  --val_dataset_sample 1 ^
  --dataset_shuffle false ^
  --temperature 0 ^
  --max_new_tokens 512 ^
  --system "%SYSTEM_PROMPT%" ^
  --result_path "%PREFLIGHT_DIR%\adapter_smoke_predictions.jsonl"
if errorlevel 1 goto command_failed
popd
exit /b 0

:preflight
if not exist "%PREFLIGHT_DIR%\adapter_smoke_predictions.jsonl" (
  echo ERROR: Missing smoke predictions.
  popd
  exit /b 2
)
python -m evaluation.coarse_to_fine table8-like-vlmloc-preflight ^
  --checkpoint-root "%CHECKPOINT_ROOT%" ^
  --adapter-dir "%FINAL_ADAPTER%" ^
  --vlmloc-data-dir "%DERIVED_DIR%" ^
  --smoke-predictions "%PREFLIGHT_DIR%\adapter_smoke_predictions.jsonl" ^
  --output-dir "%PREFLIGHT_DIR%"
if errorlevel 1 goto command_failed
popd
exit /b 0

:infer
if not exist "%PREFLIGHT_DIR%\vlmloc_runtime_preflight.json" (
  echo ERROR: Missing runtime preflight report.
  popd
  exit /b 2
)
python -c "import json,sys; report=json.load(open(sys.argv[1], encoding='utf-8')); assert report.get('compatible') is True, 'Runtime preflight is not PASS; full inference is blocked.'" "%PREFLIGHT_DIR%\vlmloc_runtime_preflight.json"
if errorlevel 1 (
  popd
  exit /b 2
)
if not exist "%INFERENCE_DIR%\" mkdir "%INFERENCE_DIR%"
if exist "%INFERENCE_DIR%\predictions.jsonl" (
  echo ERROR: Refusing to overwrite existing full predictions.
  popd
  exit /b 2
)
set "CUDA_VISIBLE_DEVICES=%GPU_LIST%"
if not defined PYTORCH_CUDA_ALLOC_CONF set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
swift infer ^
  --model "%QWEN_BASE%" ^
  --adapters "%FINAL_ADAPTER%" ^
  --infer_backend pt ^
  --device_map auto ^
  --val_dataset "%DERIVED_DIR%\vlmloc_testing_data.json" ^
  --dataset_shuffle false ^
  --max_batch_size 1 ^
  --temperature 0 ^
  --max_new_tokens 512 ^
  --system "%SYSTEM_PROMPT%" ^
  --result_path "%INFERENCE_DIR%\predictions.jsonl"
if errorlevel 1 goto command_failed
popd
exit /b 0

:score
if not exist "%INFERENCE_DIR%\predictions.jsonl" (
  echo ERROR: Missing %INFERENCE_DIR%\predictions.jsonl
  popd
  exit /b 2
)
if not exist "%DERIVED_DIR%\vlmloc_testing_index.json" (
  echo ERROR: Missing %DERIVED_DIR%\vlmloc_testing_index.json
  popd
  exit /b 2
)
python -m evaluation.coarse_to_fine table8-like-vlmloc-evaluate ^
  --predictions "%INFERENCE_DIR%\predictions.jsonl" ^
  --test-index "%DERIVED_DIR%\vlmloc_testing_index.json" ^
  --output-dir "%INFERENCE_DIR%"
if errorlevel 1 goto command_failed
popd
exit /b 0

:command_failed
set "RUN_EXIT_CODE=!ERRORLEVEL!"
if "!RUN_EXIT_CODE!"=="0" set "RUN_EXIT_CODE=1"
echo ERROR: Stage %STAGE% failed with exit code !RUN_EXIT_CODE!.
popd
exit /b !RUN_EXIT_CODE!

:usage_from_root
popd

:usage
echo Usage: scripts\run_cmmloc_qwen32_gpu.cmd STAGE
echo.
echo Native Windows CMD stages:
echo   check       Verify NVIDIA, PyTorch CUDA/BF16, ms-swift, DeepSpeed, FlashAttention
echo   setup       Download/audit CMMLoc, VLM-Loc source, and Qwen3-VL-32B
echo   coarse      Run CMMLoc baseline Top-1
echo   prepare     Create derived VLM inputs without modifying KITTI360Pose
echo   train       Train the Qwen3-VL-32B LoRA with DeepSpeed ZeRO-3
echo   finalize    Finalize SELECTED_ADAPTER and write provenance
echo   smoke       Run deterministic one-query fine inference
echo   preflight   Audit dataset, hashes, model, adapter, and smoke output
echo   infer       Run fine inference over all ordered test queries
echo   score       Write R@5/10/15 m
echo.
echo Example:
echo   conda activate cmmloc_mncl
echo   set GPU_LIST=0,1,2,3
echo   scripts\run_cmmloc_qwen32_gpu.cmd check
exit /b 2
