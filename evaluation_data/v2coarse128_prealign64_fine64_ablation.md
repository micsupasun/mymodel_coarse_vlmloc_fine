<!-- evaluation -->
1. MNCL + structured + rerank
1.1. powershell.exe -ExecutionPolicy Bypass -File "scripts\v2coarse128_prealign64_fine64_ablation.ps1" -Mode mncl_structured_rerank -Split val -RepoRoot "C:\Users\zh932237\working\CMMLoc_MNCLv4"

1.2. powershell.exe -ExecutionPolicy Bypass -File "scripts\v2coarse128_prealign64_fine64_ablation.ps1" -Mode mncl_structured_rerank -Split test -RepoRoot "C:\Users\zh932237\working\CMMLoc_MNCLv4"



2. MNCL + structured
1.1. powershell.exe -ExecutionPolicy Bypass -File "scripts\v2coarse128_prealign64_fine64_ablation.ps1" -Mode mncl_structured -Split val -RepoRoot "C:\Users\zh932237\working\CMMLoc_MNCLv4"
1.2. powershell.exe -ExecutionPolicy Bypass -File "scripts\v2coarse128_prealign64_fine64_ablation.ps1" -Mode mncl_structured -Split test -RepoRoot "C:\Users\zh932237\working\CMMLoc_MNCLv4"



3. MNCL + rerank
1.1. powershell.exe -ExecutionPolicy Bypass -File "scripts\v2coarse128_prealign64_fine64_ablation.ps1" -Mode mncl_rerank -Split val -RepoRoot "C:\Users\zh932237\working\CMMLoc_MNCLv4"
1.2. powershell.exe -ExecutionPolicy Bypass -File "scripts\v2coarse128_prealign64_fine64_ablation.ps1" -Mode mncl_rerank -Split test -RepoRoot "C:\Users\zh932237\working\CMMLoc_MNCLv4"

4. structured + rerank
1.1. powershell.exe -ExecutionPolicy Bypass -File "scripts\v2coarse128_prealign64_fine64_ablation.ps1" -Mode structured_rerank -Split val -RepoRoot "C:\Users\zh932237\working\CMMLoc_MNCLv4"
1.2. powershell.exe -ExecutionPolicy Bypass -File "scripts\v2coarse128_prealign64_fine64_ablation.ps1" -Mode structured_rerank -Split test -RepoRoot "C:\Users\zh932237\working\CMMLoc_MNCLv4"

5. MNCL
1.1. powershell.exe -ExecutionPolicy Bypass -File "scripts\v2coarse128_prealign64_fine64_ablation.ps1" -Mode mncl -Split val -RepoRoot "C:\Users\zh932237\working\CMMLoc_MNCLv4"
1.2. powershell.exe -ExecutionPolicy Bypass -File "scripts\v2coarse128_prealign64_fine64_ablation.ps1" -Mode mncl -Split test -RepoRoot "C:\Users\zh932237\working\CMMLoc_MNCLv4"

6. structured agreement
1.1. powershell.exe -ExecutionPolicy Bypass -File "scripts\v2coarse128_prealign64_fine64_ablation.ps1" -Mode structured -Split val -RepoRoot "C:\Users\zh932237\working\CMMLoc_MNCLv4"
1.2. powershell.exe -ExecutionPolicy Bypass -File "scripts\v2coarse128_prealign64_fine64_ablation.ps1" -Mode structured -Split test -RepoRoot "C:\Users\zh932237\working\CMMLoc_MNCLv4"

7. rerank
1.1. powershell.exe -ExecutionPolicy Bypass -File "scripts\v2coarse128_prealign64_fine64_ablation.ps1" -Mode rerank -Split val -RepoRoot "C:\Users\zh932237\working\CMMLoc_MNCLv4"
1.2. powershell.exe -ExecutionPolicy Bypass -File "scripts\v2coarse128_prealign64_fine64_ablation.ps1" -Mode rerank -Split test -RepoRoot "C:\Users\zh932237\working\CMMLoc_MNCLv4"







