<#
.SYNOPSIS
Run shared warmup, the five IB/ITM ablations, and paired validation.
.EXAMPLE
.\scripts\run_ib_itm.ps1
.EXAMPLE
.\scripts\run_ib_itm.ps1 -Steps 6400 -RunDir artifacts/experiments/ib_itm_long_6400
#>
[CmdletBinding()]
param(
    [string]$Config = 'artifacts/experiments/query_ib_v3/fresh_config.yaml',
    [string]$StageCCheckpoint = 'artifacts/checkpoints/connected.pt',
    [string]$RunDir = 'artifacts/experiments/ib_itm_long',
    [ValidateRange(1, 2147483647)]
    [int]$WarmupSteps = 1500,
    [ValidateRange(1, 2147483647)]
    [int]$Steps = 3200,
    [ValidateNotNullOrEmpty()]
    [ValidateRange(0, 2147483647)]
    [int[]]$Seeds = @(42, 43),
    [ValidateRange(1, 2147483647)]
    [int]$BatchSize = 32,
    [ValidateRange(1, 2147483647)]
    [int]$EvalBatchSize = 64,
    [ValidateRange(1, 2147483647)]
    [int]$EvalSteps = 40
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$modes = @('baseline', 'ib', 'itm', 'ib_itm', 'dynamic_itm')

Push-Location -LiteralPath $repoRoot
try {
    $warmupDir = Join-Path $RunDir 'warm'
    python -m training.train_shared --config $Config --base-checkpoint $StageCCheckpoint --steps $WarmupSteps --batch-size $BatchSize --seed 1337 --run-dir $warmupDir
    if ($LASTEXITCODE -ne 0) { throw 'Shared warmup failed' }

    $baseCheckpoint = Join-Path $warmupDir 'shared.pt'
    foreach ($seed in $Seeds) {
        foreach ($mode in $modes) {
            $modeDir = Join-Path $RunDir "seed_$seed/$mode"
            python -m training.train_shared --config $Config --base-checkpoint $baseCheckpoint --steps $Steps --batch-size $BatchSize --seed $seed --objective $mode --run-dir $modeDir
            if ($LASTEXITCODE -ne 0) { throw "Training failed for $mode, seed $seed" }
        }

        # Validate all scenes. Here --steps controls ODE sampling.
        $evaluation = @('--config', $Config, '--split', 'val',
                        '--batch-size', "$EvalBatchSize", '--steps', "$EvalSteps", '--record-joint',
                        '--out', (Join-Path $RunDir "validation_seed_$seed.json"))
        foreach ($mode in $modes) {
            $checkpoint = Join-Path $RunDir "seed_$seed/$mode/shared.pt"
            $evaluation += @('--checkpoint', $mode, $checkpoint)
        }
        python -m evaluation.evaluate @evaluation
        if ($LASTEXITCODE -ne 0) { throw "Evaluation failed for seed $seed" }
    }
}
finally {
    Pop-Location
}
