@echo off
setlocal EnableExtensions

rem Native Windows CMD, evaluation only:
rem   CMMLoc baseline Top-1 -> public VLM-Loc Qwen3-VL-32B -> R@5/10/15 m
rem
rem No WSL, training, DeepSpeed, FlashAttention, or dataset mutation is used.

if "%~1"=="" goto usage
set "STAGE=%~1"

pushd "%~dp0.."
if errorlevel 1 (
  echo ERROR: Cannot enter the repository root.
  exit /b 2
)

if not defined DATA_ROOT set "DATA_ROOT=data\k360_30-10_scG_pd10_pc4_spY_all"
if not defined CHECKPOINT_ROOT set "CHECKPOINT_ROOT=checkpoints\k360_30-10_scG_pd10_pc4_spY_all"
if not defined OUTPUT_ROOT set "OUTPUT_ROOT=evaluation_outputs\cmmloc_top1_qwen32_public_zero_shot"
if not defined SETUP_DIR set "SETUP_DIR=%OUTPUT_ROOT%\setup"
if not defined COARSE_DIR set "COARSE_DIR=%OUTPUT_ROOT%\stage1_cmmloc"
if not defined DERIVED_DIR set "DERIVED_DIR=%OUTPUT_ROOT%\vlmloc_data_30m"
if not defined PREFLIGHT_DIR set "PREFLIGHT_DIR=%OUTPUT_ROOT%\stage2_preflight"
if not defined INFERENCE_DIR set "INFERENCE_DIR=%OUTPUT_ROOT%\stage2_inference"
if not defined VLMLOC_ROOT set "VLMLOC_ROOT=%CHECKPOINT_ROOT%\VLM-Loc"
if not defined QWEN_BASE set "QWEN_BASE=%VLMLOC_ROOT%\base_models\Qwen3-VL-32B-Instruct"
if not defined PUBLIC_ADAPTER set "PUBLIC_ADAPTER=%VLMLOC_ROOT%\output\v0-20251109-125010-qwen3_32b\checkpoint-3300"
if not defined OFFICIAL_SOURCE set "OFFICIAL_SOURCE=%VLMLOC_ROOT%\official_source\nku-3d-vision-494a8b4e3fe9\vlm-loc"
if not defined SYSTEM_PROMPT set "SYSTEM_PROMPT=%OFFICIAL_SOURCE%\system_prompt.txt"
if not defined GPU_LIST set "GPU_LIST=0"
if not defined COARSE_BATCH_SIZE set "COARSE_BATCH_SIZE=16"
if not defined COARSE_DEVICE set "COARSE_DEVICE=cuda:0"
if not defined VLM_MAX_MEMORY set VLM_MAX_MEMORY={0: "15GiB", "cpu": "49GiB"}

if /I "%STAGE%"=="check-coarse" goto check_coarse
if /I "%STAGE%"=="check" goto check_vlm
if /I "%STAGE%"=="setup" goto setup
if /I "%STAGE%"=="coarse" goto coarse
if /I "%STAGE%"=="prepare" goto prepare
if /I "%STAGE%"=="smoke" goto smoke
if /I "%STAGE%"=="preflight" goto preflight
if /I "%STAGE%"=="infer" goto infer
if /I "%STAGE%"=="score" goto score
goto usage_from_root

:check_coarse
echo Checking the CMMLoc coarse environment...
where nvidia-smi.exe >nul 2>nul
if errorlevel 1 goto missing_nvidia
where python.exe >nul 2>nul
if errorlevel 1 goto missing_python
nvidia-smi.exe -L
if errorlevel 1 goto command_failed
python -c "import torch,torch_geometric; print('torch=',torch.__version__); print('torch_geometric=',torch_geometric.__version__); print('cuda_available=',torch.cuda.is_available()); assert torch.cuda.is_available()"
if errorlevel 1 (
  echo ERROR: CMMLoc PyTorch/PyG CUDA check failed.
  popd
  exit /b 2
)
echo CMMLoc coarse environment check passed.
popd
exit /b 0

:check_vlm
echo Checking the Qwen3-VL-32B inference environment...
where nvidia-smi.exe >nul 2>nul
if errorlevel 1 goto missing_nvidia
where python.exe >nul 2>nul
if errorlevel 1 goto missing_python
where swift.exe >nul 2>nul
if errorlevel 1 (
  echo ERROR: swift.exe from ms-swift is not installed in this environment.
  popd
  exit /b 2
)
nvidia-smi.exe -L
if errorlevel 1 goto command_failed
python -c "import torch,transformers,peft,accelerate,swift,psutil,shutil; from packaging.version import Version; print('torch=',torch.__version__); print('transformers=',transformers.__version__); print('peft=',peft.__version__); print('accelerate=',accelerate.__version__); print('torch_cuda=',torch.version.cuda); print('cuda_available=',torch.cuda.is_available()); print('bf16_supported=',torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False); r=psutil.virtual_memory(); d=shutil.disk_usage('.'); print('RAM_available_GiB=',round(r.available/2**30,1)); print('Disk_free_GiB=',round(d.free/2**30,1)); assert Version(torch.__version__.split('+')[0]) >= Version('2.1.0'), 'PyTorch >=2.1 required'; assert Version(transformers.__version__) >= Version('4.57.0'), 'transformers >=4.57 required'; assert torch.cuda.is_available(), 'PyTorch CUDA unavailable'; assert torch.cuda.is_bf16_supported(), 'BF16 unsupported'"
if errorlevel 1 (
  echo ERROR: Qwen3-VL-32B inference check failed.
  echo Use the separate vlmloc_qwen32 Conda environment documented in EVALUATION_COARSE_TO_FINE.md.
  popd
  exit /b 2
)
echo Qwen3-VL-32B BF16 inference environment check passed.
echo GPU_LIST=%GPU_LIST%
echo VLM_MAX_MEMORY=%VLM_MAX_MEMORY%
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
  --public-qwen32-evaluation
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
  --output-dir "%DERIVED_DIR%" ^
  --evaluation-only
if errorlevel 1 goto command_failed
popd
exit /b 0

:smoke
if not exist "%PUBLIC_ADAPTER%\adapter_model.safetensors" (
  echo ERROR: Missing public Qwen32 adapter: %PUBLIC_ADAPTER%
  popd
  exit /b 2
)
if not exist "%QWEN_BASE%\model.safetensors.index.json" (
  echo ERROR: Missing Qwen3-VL-32B BF16 base weights: %QWEN_BASE%
  popd
  exit /b 2
)
if not exist "%SYSTEM_PROMPT%" (
  echo ERROR: Missing VLM-Loc system prompt: %SYSTEM_PROMPT%
  popd
  exit /b 2
)
if not exist "%DERIVED_DIR%\vlmloc_testing_smoke_1.json" (
  echo ERROR: Missing evaluation-only smoke dataset.
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
set "TOKENIZERS_PARALLELISM=false"
if not defined PYTORCH_CUDA_ALLOC_CONF set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
swift infer ^
  --model "%QWEN_BASE%" ^
  --adapters "%PUBLIC_ADAPTER%" ^
  --load_args false ^
  --infer_backend pt ^
  --template qwen3_vl ^
  --torch_dtype bfloat16 ^
  --attn_impl sdpa ^
  --device_map auto ^
  --max_memory "%VLM_MAX_MEMORY%" ^
  --val_dataset "%DERIVED_DIR%\vlmloc_testing_smoke_1.json" ^
  --val_dataset_sample 1 ^
  --dataset_shuffle false ^
  --max_batch_size 1 ^
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
python scripts\evaluate_qwen32_fine.py preflight ^
  --checkpoint-root "%CHECKPOINT_ROOT%" ^
  --adapter-dir "%PUBLIC_ADAPTER%" ^
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
python -c "import json,sys; report=json.load(open(sys.argv[1],encoding='utf-8')); assert report.get('compatible') is True; assert report.get('training_performed') is False; assert report.get('table8_reproduction') is False" "%PREFLIGHT_DIR%\vlmloc_runtime_preflight.json"
if errorlevel 1 (
  echo ERROR: Runtime preflight is not PASS; full inference is blocked.
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
set "TOKENIZERS_PARALLELISM=false"
if not defined PYTORCH_CUDA_ALLOC_CONF set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
swift infer ^
  --model "%QWEN_BASE%" ^
  --adapters "%PUBLIC_ADAPTER%" ^
  --load_args false ^
  --infer_backend pt ^
  --template qwen3_vl ^
  --torch_dtype bfloat16 ^
  --attn_impl sdpa ^
  --device_map auto ^
  --max_memory "%VLM_MAX_MEMORY%" ^
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
python scripts\evaluate_qwen32_fine.py score ^
  --predictions "%INFERENCE_DIR%\predictions.jsonl" ^
  --test-index "%DERIVED_DIR%\vlmloc_testing_index.json" ^
  --output-dir "%INFERENCE_DIR%"
if errorlevel 1 goto command_failed
popd
exit /b 0

:missing_nvidia
echo ERROR: nvidia-smi.exe is not on PATH.
popd
exit /b 2

:missing_python
echo ERROR: python.exe is not available in the active Conda environment.
popd
exit /b 2

:command_failed
set "RUN_EXIT_CODE=%ERRORLEVEL%"
if "%RUN_EXIT_CODE%"=="0" set "RUN_EXIT_CODE=1"
echo ERROR: Stage %STAGE% failed with exit code %RUN_EXIT_CODE%.
popd
exit /b %RUN_EXIT_CODE%

:usage_from_root
popd

:usage
echo Usage: scripts\run_cmmloc_qwen32_gpu.cmd STAGE
echo.
echo Native Windows CMD evaluation-only stages:
echo   check-coarse  Check the existing CMMLoc/PyG environment
echo   check         Check the separate Qwen3-VL-32B BF16 inference environment
echo   setup         Download/audit CMMLoc, public 32B adapter, source, and BF16 base
echo   coarse        Run CMMLoc baseline Top-1
echo   prepare       Generate test-only derived VLM inputs; never modify KITTI360Pose
echo   smoke         Run one public Qwen3-VL-32B zero-shot fine prediction
echo   preflight     Audit hashes, model identity, data immutability, and smoke output
echo   infer         Run all ordered KITTI360Pose test queries
echo   score         Write world-coordinate R@5/10/15 m
echo.
echo This is a CityLoc-K/50m adapter evaluated zero-shot on KITTI360Pose/30m.
echo It performs no training and is not an exact Table-8 reproduction.
exit /b 2
