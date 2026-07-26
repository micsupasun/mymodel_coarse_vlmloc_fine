param(
    [Parameter(Mandatory=$true)]
    [ValidateSet(
        "mncl_structured_rerank",
        "mncl_structured",
        "mncl_rerank",
        "structured_rerank",
        "mncl",
        "structured",
        "rerank"
    )]
    [string]$Mode,

    [ValidateSet("val", "test")]
    [string]$Split = "val",

    [switch]$CoarseOnly,

    [int[]]$Threshs = @(5, 10, 15, 20, 25),

    # [string]$RepoRoot = "I:\Github storage\CMMLocPP"
    [string]$RepoRoot = "C:\Users\zh932237\working\CMMLoc_MNCLv4" # update:1 repo
    
)

$ErrorActionPreference = "Stop"
Set-Location $RepoRoot

$dataset = Join-Path $RepoRoot "data\k360_30-10_scG_pd10_pc4_spY_all" # update:2 dataset
$root = Join-Path $RepoRoot "checkpoints\k360_30-10_scG_pd10_pc4_spY_all" # update:3
$pointnet = Join-Path $RepoRoot "checkpoints\pointnet_acc0.86_lr1_p256.pth" # update:4

$cmmlocCoarse = Join-Path $root "coarsev1_batch64_epoch20_ablation_study\coarse_contN_epoch6_acc0.788_ecl0_eco0_p256_npa1_loss-contrastive_f-class-color-position-num.pth" # update coarse(don't change just update name of folder with cmmlocpp_mncl_v2) # update:5
$mnclCoarse = Join-Path $root "coarsev1_batch64_epoch20_ablation_study\coarse_contN_epoch6_acc0.788_ecl0_eco0_p256_npa1_loss-contrastive_f-class-color-position-num.pth" # update checkpoint name of coarse # update:6
$rerankCoarse = Join-Path $root "coarsev1_batch64_epoch20_ablation_study\coarse_contN_epoch6_acc0.788_ecl0_eco0_p256_npa1_loss-contrastive_f-class-color-position-num.pth" # update:7
$fine = Join-Path $root "finev2_batch32_epoch45_ablation_study\fine_contN_epoch15_offset0.100_lr0.0003_obj-6-16_ecl0_eco0_p256_npa1_f-class-color-position-num.pth"

$coarsePath = $cmmlocCoarse
$extra = @()
$description = ""

switch ($Mode) {
    "mncl_structured_rerank" {
        $coarsePath = $rerankCoarse
        $extra = @("--use_model_reranker", "--rerank_topn", "50")
        $description = "MNCL + structured agreement + trainable model rerank"
    }
    "mncl_structured" {
        $coarsePath = $mnclCoarse
        $extra = @("--rerank_topn", "50", "--rerank_base_weight", "0.0", "--rerank_label_weight", "0.6", "--rerank_color_weight", "1.0")
        $description = "MNCL candidate pool + structured agreement score"
    }
    "mncl_rerank" {
        $coarsePath = $mnclCoarse
        $extra = @("--rerank_topn", "50", "--rerank_base_weight", "1.0", "--rerank_label_weight", "0.0", "--rerank_color_weight", "0.0")
        $description = "MNCL candidate pool + rerank path with base score only"
    }
    "structured_rerank" {
        $coarsePath = $cmmlocCoarse
        $extra = @("--rerank_topn", "50", "--rerank_base_weight", "1.0", "--rerank_label_weight", "0.6", "--rerank_color_weight", "1.0")
        $description = "CMMLoc-style candidate pool + structured agreement + rerank"
    }
    "mncl" {
        $coarsePath = $mnclCoarse
        $extra = @()
        $description = "MNCL checkpoint only, no rerank"
    }
    "structured" {
        $coarsePath = $cmmlocCoarse
        $extra = @("--rerank_topn", "50", "--rerank_base_weight", "0.0", "--rerank_label_weight", "0.6", "--rerank_color_weight", "1.0")
        $description = "Structured agreement score only inside CMMLoc-style top-50 candidate pool"
    }
    "rerank" {
        $coarsePath = $cmmlocCoarse
        $extra = @("--rerank_topn", "50", "--rerank_base_weight", "1.0", "--rerank_label_weight", "0.0", "--rerank_color_weight", "0.0")
        $description = "Rerank path enabled with base score only"
    }
}

$cmd = @(
    "-m", "evaluation.pipeline",
    "--base_path", $dataset,
    "--use_features", "class", "color", "position", "num",
    "--no_pc_augment",
    "--no_pc_augment_fine",
    "--hungging_model", "t5-large",
    "--fixed_embedding",
    "--text_max_length", "128",
    "--pointnet_path", $pointnet,
    "--threshs"
) + $Threshs + @(
    "--path_coarse", $coarsePath,
    "--path_fine", $fine
) + $extra

if ($Split -eq "test") {
    $cmd += "--use_test_set"
}
if ($CoarseOnly) {
    $cmd += "--coarse_only"
}

Write-Host "Phase 1 ablation mode: $Mode"
Write-Host "Description: $description"
Write-Host "Split: $Split"
Write-Host "Thresholds: $($Threshs -join '/')"
Write-Host "Coarse checkpoint: $coarsePath"
Write-Host "Fine checkpoint: $fine"
Write-Host ""
Write-Host "Running: python $($cmd -join ' ')"
Write-Host ""

& python @cmd
