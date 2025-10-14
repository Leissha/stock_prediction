<#
Usage:
  cd dev
  powershell -ExecutionPolicy Bypass -File run_experiments.ps1

This script runs batch experiments for Task C.6 across TF models and the ensemble (SARIMA + TF).
#>

param(
  [string]$Company = "CBA.AX",
  [string]$StartDate = "2023-10-11",
  [string]$EndDate = "2025-10-10",
  [string]$TargetFeature = "Close",
  [int[]]$Lookbacks = @(60),
  [int[]]$Horizons = @(1, 5),
  [string[]]$TfModels = @('lstm','gru','rnn','bilstm'),
  [string[]]$Layers = @('50 50 50','64 32'),
  [double[]]$Dropouts = @(0.2),
  [int[]]$Epochs = @(100),
  [int[]]$BatchSizes = @(32),
  [string[]]$Optimizers = @('adam'),
  [double[]]$LearningRates = @(0.0005),
  [double[]]$SarimaWeights = @(0.3, 0.5, 0.7)
)

$ErrorActionPreference = 'Stop'

# Resolve python from local venv if present
$python = Join-Path $PSScriptRoot '..\venv\Scripts\python.exe'
if (!(Test-Path $python)) { $python = 'python' }

function Invoke-OneExperiment {
  param(
    [string]$ModelName,
    [string]$Ensemble2 = '',
    [int]$L,
    [int]$K,
    [string]$LayerStr,
    [double]$Drop,
    [int]$Ep,
    [int]$Bs,
    [string]$Opt,
    [double]$Lr,
    [double]$SarimaWeight = 0.5,
    [switch]$LogRet
  )

  # Calculate model_2_weight (complementary to SARIMA weight)
  $Model2Weight = 1.0 - $SarimaWeight

  # Build meta path to optionally skip if results already exist
  $retSuffix = if ($LogRet) { '_log_ret' } else { '' }
  $base = "$Company" + "_${StartDate}_to_${EndDate}" + "_${TargetFeature}${retSuffix}_step_${K}"
  $layerItems = ($LayerStr -split '\s+')
  $layersTag = '[' + ($layerItems -join ', ') + ']'
  $metaTag = if ($ModelName -eq 'ensemble') { 
    "ensemble+sarima+${Ensemble2}_w${SarimaWeight}"
  } else { 
    $ModelName 
  }
  $metaPath = "${base}_seq-${L}_${metaTag}_layers${layersTag}_dropout${Drop}_epochs${Ep}_bs${Bs}"
  $resultCsv = Join-Path (Join-Path $PSScriptRoot 'results') ("${metaPath}.csv")
  if (Test-Path $resultCsv) {
    Write-Host "Skip existing results: $resultCsv" -ForegroundColor Yellow
    return
  }

  $common = @(
    '--company', $Company,
    '--start_date', $StartDate,
    '--end_date', $EndDate,
    '--lookback', $L,
    '--horizon', $K,
    '--layers'
  )
  # split '50 50 50' -> '50','50','50'
  $common += ($LayerStr -split '\s+')
  $common += @(
    '--dropout_rate', $Drop,
    '--epochs', $Ep,
    '--batch_size', $Bs,
    '--optimizer', $Opt,
    '--learning_rate', $Lr,
    '--scale',
    '--val_size', 0.1,
    '--sarima_weight', $SarimaWeight,
    '--model_2_weight', $Model2Weight
  )

  if ($LogRet) { $common += '--log_ret' }

  if ($ModelName -eq 'ensemble') {
    Write-Host "Running ensemble: SARIMA($SarimaWeight) + $Ensemble2($Model2Weight)" -ForegroundColor Yellow
    & $python "$PSScriptRoot\main.py" @common --model_name ensemble --ensemble_2 $Ensemble2 | Write-Output
  } else {
    & $python "$PSScriptRoot\main.py" @common --model_name $ModelName | Write-Output
  }
}

Write-Host "=== Running All Models (TF Baselines + Ensembles) ===" -ForegroundColor Cyan
Write-Host "SARIMA Weights: $($SarimaWeights -join ', ')" -ForegroundColor Green

foreach ($m in $TfModels) {
  foreach ($L in $Lookbacks) {
    foreach ($K in $Horizons) {
      foreach ($layer in $Layers) {
        foreach ($opt in $Optimizers) {
          foreach ($lr in $LearningRates) {
            # Run TF baseline
            Write-Host "Running TF baseline: $m" -ForegroundColor Yellow
            Invoke-OneExperiment -ModelName $m -L $L -K $K -LayerStr $layer -Drop $Dropouts[0] -Ep $Epochs[0] -Bs $BatchSizes[0] -Opt $opt -Lr $lr -LogRet
            
            # Run ensemble with this TF model for each SARIMA weight
            foreach ($sarimaWeight in $SarimaWeights) {
              Invoke-OneExperiment -ModelName 'ensemble' -Ensemble2 $m -L $L -K $K -LayerStr $layer -Drop $Dropouts[0] -Ep $Epochs[0] -Bs $BatchSizes[0] -Opt $opt -Lr $lr -SarimaWeight $sarimaWeight -LogRet
            }
          }
        }
      }
    }
  }
}

Write-Host "=== Done ===" -ForegroundColor Green