# Usage: Right-click → Run with PowerShell 
# or cd/dev & run: powershell -ExecutionPolicy Bypass -File run_experiments.ps1

param(
  [string]$Company = "CBA.AX",
  [string]$StartDate = "2023-10-01",
  [string]$EndDate = "2025-09-30",
  [int]$LagDays = 60,
  [int]$Epochs = 200,
  [int]$BatchSize = 32,
  [string]$ModelName = "lstm",
  [int[]]$ReturnSteps = @(1,5,10),
  [int[]]$PriceSteps = @(1,5)
)

$ErrorActionPreference = 'Stop'

# Resolve python from local venv if present
$python = Join-Path $PSScriptRoot '..\venv\Scripts\python.exe'
if (!(Test-Path $python)) { $python = 'python' }

Write-Host "=== Running RETURN targets for steps: $ReturnSteps ===" -ForegroundColor Cyan
foreach ($k in $ReturnSteps) {
  & $python "$PSScriptRoot\main.py" `
    --company $Company `
    --start_date $StartDate `
    --end_date $EndDate `
    --lookup_steps $k `
    --lag_days $LagDays `
    --model_name $ModelName `
    --layers 50 50 50 `
    --dropout_rate 0.2 `
    --epochs $Epochs `
    --batch_size $BatchSize `
    --scale `
    --split_method date `
    --target_ret
}

Write-Host "=== Running PRICE targets for steps: $PriceSteps ===" -ForegroundColor Cyan
foreach ($k in $PriceSteps) {
  & $python "$PSScriptRoot\main.py" `
    --company $Company `
    --start_date $StartDate `
    --end_date $EndDate `
    --lookup_steps $k `
    --lag_days $LagDays `
    --model_name $ModelName `
    --layers 50 50 50 `
    --dropout_rate 0.2 `
    --epochs $Epochs `
    --batch_size $BatchSize `
    --scale `
    --split_method date
}

Write-Host "=== Done ===" -ForegroundColor Green

