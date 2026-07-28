@echo off
setlocal EnableExtensions

rem Windows CMD entry point for:
rem   CMMLoc Top-1 coarse -> VLM-Loc Qwen3-VL-32B fine -> R@5/10/15 m
rem
rem The user invokes only this .cmd file. The Linux CUDA workload is executed
rem inside the default WSL distribution because DeepSpeed ZeRO-3 and
rem FlashAttention are not supported reliably by the native Windows stack.

if "%~1"=="" goto usage
set "STAGE=%~1"

where wsl.exe >nul 2>nul
if errorlevel 1 (
  echo ERROR: WSL is not installed or wsl.exe is not on PATH.
  echo Install WSL with an Ubuntu distribution and enable NVIDIA CUDA for WSL.
  exit /b 2
)

pushd "%~dp0.."
if errorlevel 1 (
  echo ERROR: Cannot enter the repository root.
  exit /b 2
)

for /f "delims=" %%P in ('wsl.exe wslpath -a "%CD%" 2^>nul') do set "WSL_REPO=%%P"
if not defined WSL_REPO (
  echo ERROR: The default WSL distribution cannot translate the repository path.
  popd
  exit /b 2
)

if not defined GPU_LIST set "GPU_LIST=0,1,2,3"

rem WSLENV forwards optional CMD variables when they are defined. Relative
rem paths should use forward slashes, for example data/k360_....
set "WSLENV=GPU_LIST:TRAIN_WORLD_SIZE:DATA_ROOT:CHECKPOINT_ROOT:OUTPUT_ROOT:SETUP_DIR:COARSE_DIR:DERIVED_DIR:PREFLIGHT_DIR:INFERENCE_DIR:QWEN_BASE:OFFICIAL_SOURCE:SYSTEM_PROMPT:TRAIN_OUTPUT:FINAL_ADAPTER:SELECTED_ADAPTER:COARSE_BATCH_SIZE:COARSE_DEVICE:NCCL_P2P_DISABLE:NCCL_IB_DISABLE:PYTORCH_CUDA_ALLOC_CONF:%WSLENV%"

if /I "%STAGE%"=="check" (
  echo Checking NVIDIA CUDA in WSL...
  wsl.exe nvidia-smi
  if errorlevel 1 (
    echo ERROR: NVIDIA CUDA is not visible inside the default WSL distribution.
    popd
    exit /b 2
  )
  echo Checking Python and ms-swift in WSL...
  wsl.exe bash -lc "command -v python && python --version && command -v swift"
  if errorlevel 1 (
    echo ERROR: Python or the swift command is missing inside WSL.
    popd
    exit /b 2
  )
  echo CMD/WSL GPU environment check passed.
  popd
  exit /b 0
)

set "NEEDS_GPU=0"
for %%S in (coarse train smoke infer) do (
  if /I "%STAGE%"=="%%S" set "NEEDS_GPU=1"
)
if "%NEEDS_GPU%"=="1" (
  wsl.exe nvidia-smi -L >nul 2>nul
  if errorlevel 1 (
    echo ERROR: NVIDIA CUDA is not visible inside the default WSL distribution.
    echo Run "wsl nvidia-smi" from CMD and fix CUDA for WSL before this stage.
    popd
    exit /b 2
  )
)

echo Repository: %CD%
echo WSL path:  %WSL_REPO%
echo Stage:     %STAGE%
echo GPUs:      %GPU_LIST%

rem tr removes CR characters if Git checked the Bash launcher out with Windows
rem line endings. No tracked file is modified.
wsl.exe bash -lc "cd '%WSL_REPO%' && tr -d '\r' < scripts/run_cmmloc_qwen32_gpu.sh | bash -s -- '%STAGE%'"
set "RUN_EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %RUN_EXIT_CODE%

:usage
echo Usage: scripts\run_cmmloc_qwen32_gpu.cmd STAGE
echo.
echo Stages:
echo   check       Verify WSL CUDA, Python, and the ms-swift command
echo   setup       Download/audit CMMLoc, VLM-Loc source, and Qwen3-VL-32B
echo   coarse      Run CMMLoc baseline Top-1
echo   prepare     Create derived VLM inputs without modifying KITTI360Pose
echo   train       Train the Qwen3-VL-32B LoRA
echo   finalize    Finalize SELECTED_ADAPTER and write provenance
echo   smoke       Run deterministic one-query fine inference
echo   preflight   Audit dataset, hashes, model, adapter, and smoke output
echo   infer       Run fine inference over all ordered test queries
echo   score       Write R@5/10/15 m
echo.
echo Example:
echo   set GPU_LIST=0,1,2,3
echo   scripts\run_cmmloc_qwen32_gpu.cmd setup
exit /b 2
