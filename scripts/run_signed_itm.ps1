<#
.SYNOPSIS
Train the signed beta(A, C) objective from the shared IB/ITM warm checkpoint and validate it with the baseline.
.EXAMPLE
.\scripts\run_signed_itm.ps1
#>
[CmdletBinding()]
param(
    [string]$Config = 'artifacts/experiments/query_ib_v3/fresh_config.yaml',
    [string]$BaseCheckpoint = 'artifacts/experiments/ib_itm_long/warm/shared.pt',
    [string]$BaselineRoot = 'artifacts/experiments/ib_itm_long',
    [string]$RunDir = 'artifacts/experiments/signed_itm_long',
    [ValidateRange(1, 2147483647)]
    [int]$Steps = 3200,
    [ValidateNotNullOrEmpty()]
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

Push-Location -LiteralPath $repoRoot
try {
    foreach ($seed in $Seeds) {
        $modeDir = Join-Path $RunDir "seed_$seed/signed_itm"
        python -m training.train_shared --config $Config --base-checkpoint $BaseCheckpoint --steps $Steps --batch-size $BatchSize --seed $seed --objective signed_itm --run-dir $modeDir
        if ($LASTEXITCODE -ne 0) { throw "signed_itm training failed for seed $seed" }

        # The baseline is re-scored to confirm pairing with the stored per-scene hits.
        python -m evaluation.evaluate --config $Config --split val --batch-size $EvalBatchSize --steps $EvalSteps --record-joint `
            --out (Join-Path $RunDir "validation_seed_$seed.json") `
            --checkpoint baseline (Join-Path $BaselineRoot "seed_$seed/baseline/shared.pt") `
            --checkpoint signed_itm (Join-Path $modeDir 'shared.pt')
        if ($LASTEXITCODE -ne 0) { throw "Evaluation failed for seed $seed" }
    }
}
finally {
    Pop-Location
}
