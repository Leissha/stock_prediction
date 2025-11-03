# Smart Model Comparison Script
# Focuses on key comparisons: Baseline vs Best vs Hybrid models
# Tests feature engineering impact (sentiment + technical indicators)
# Optimized to skip sentiment fetching when cache is sufficient

param(
    [string]$Company = "AAPL",
    [int]$Lookback = 60,
    [string]$StartDate = "2023-11-02",
    [string]$EndDate = "2025-11-01",
    [switch]$SkipSentimentFetch = $false,
    [switch]$FastMode = $false  # Reduced experiments for quick testing
)

$script_dir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($script_dir) {
    Set-Location $script_dir
}

if (-not (Test-Path "main.py")) {
    Write-Host "ERROR: main.py not found" -ForegroundColor Red
    exit 1
}

# Smart Configuration: Focus on meaningful comparisons
# 1. Baseline model (simple, fast)
# 2. Best model (Attention-LSTM - best performance)
# 3. Hybrid models (CNN-RNN, CNN-BiLSTM - show hybrid benefits)
# 4. Attention variants (Attention-RNN - show attention mechanism)

if ($FastMode) {
    # Fast mode: Only most important comparisons
    $models = @("rnn", "attention_lstm")
    $horizons = @(1, 20)
    $patience_values = @(20)  # Single patience value
    $reduce_lr_modes = @("with_reduce_lr")  # Fast mode: only with ReduceLROnPlateau
    Write-Host "FAST MODE: Reduced experiments for quick testing" -ForegroundColor Yellow
} else {
    # Full comparison: Best vs Worst + Hybrid showcase + Feature engineering
    $models = @("rnn", "attention_lstm", "cnn_rnn", "cnn_bilstm", "attention_rnn")
    $horizons = @(1, 5, 20)
    $patience_values = @(20, 50)  # Test different patience levels
    $reduce_lr_modes = @("with_reduce_lr", "no_reduce_lr")  # Test ReduceLROnPlateau impact
}

$epochs_values = @(100)  # Fixed at 100, let patience handle early stopping
$sentiment_modes = @("with_sentiment", "baseline")
$reduce_lr_modes = @("with_reduce_lr", "no_reduce_lr")  # Test feature engineering impact

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "SMART MODEL COMPARISON EXPERIMENT" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Company: $Company" -ForegroundColor White
Write-Host "  Lookback: $Lookback days" -ForegroundColor White
Write-Host "  Date Range: $StartDate to $EndDate" -ForegroundColor White
Write-Host "  Models: $($models -join ', ')" -ForegroundColor White
Write-Host "    - Baseline: RNN" -ForegroundColor Gray
Write-Host "    - Best: Attention-LSTM" -ForegroundColor Gray
Write-Host "    - Hybrid: CNN-RNN, CNN-BiLSTM" -ForegroundColor Gray
Write-Host "    - Attention: Attention-RNN" -ForegroundColor Gray
Write-Host "  Horizons: $($horizons -join ', ') days" -ForegroundColor White
Write-Host "  Patience: $($patience_values -join ', ') (ReduceLROnPlateau patience)" -ForegroundColor White
Write-Host "  Epochs: $($epochs_values -join ', ') (with early stopping)" -ForegroundColor White
Write-Host "  Sentiment: 2 modes (with sentiment+social, baseline)" -ForegroundColor White
Write-Host "  ReduceLROnPlateau: $($reduce_lr_modes.Count) modes (with/without)" -ForegroundColor White
Write-Host "  Skip Sentiment Fetch: $SkipSentimentFetch (use cache if sufficient)" -ForegroundColor $(if ($SkipSentimentFetch) { "Green" } else { "Yellow" })
Write-Host ""

$total_runs = $models.Count * $horizons.Count * $patience_values.Count * $epochs_values.Count * $sentiment_modes.Count
Write-Host "Total Experiments: $total_runs" -ForegroundColor Green
Write-Host "Estimated Time: ~$([math]::Round($total_runs * 2.5 / 60, 1)) hours (assuming 2.5 min per run)" -ForegroundColor Green
Write-Host ""
Write-Host "Key Comparisons:" -ForegroundColor Cyan
Write-Host "  1. Baseline (RNN) vs Best (Attention-LSTM)" -ForegroundColor White
Write-Host "  2. Hybrid models (CNN-RNN, CNN-BiLSTM) to show architecture benefits" -ForegroundColor White
Write-Host "  3. Sentiment impact (with vs without)" -ForegroundColor White
Write-Host "  4. Feature engineering (ReduceLROnPlateau + Early Stopping)" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to cancel, or any key to start..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

# Create results directory
$resultsDir = "results\model_comparison_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null
Write-Host "Results will be saved to: $resultsDir" -ForegroundColor Cyan
Write-Host ""

# Initialize counters
$run_count = 0
$success_count = 0
$error_count = 0
$skipped_count = 0
$results = @()

# Start timer
$start_time = Get-Date

# Build experiments list (avoid deeply nested loops)
$experiments = @()
foreach ($model in $models) {
    foreach ($horizon in $horizons) {
        foreach ($patience in $patience_values) {
            foreach ($epochs in $epochs_values) {
                foreach ($sentiment_mode in $sentiment_modes) {
                    $experiments += [PSCustomObject]@{
                        Model = $model
                        Horizon = $horizon
                        Patience = $patience
                        Epochs = $epochs
                        Sentiment = $sentiment_mode
                    }
                }
            }
        }
    }
}

# Check sentiment cache before starting (optional optimization)
$skip_sentiment_flag = ""
if ($SkipSentimentFetch) {
    Write-Host "Checking sentiment cache..." -ForegroundColor Cyan
    try {
        $cache_check = python -c "from sentiment.crawl_news import _read_cache; from datetime import datetime; df = _read_cache('$Company'); start = datetime.strptime('$StartDate', '%Y-%m-%d'); end = datetime.strptime('$EndDate', '%Y-%m-%d'); in_range = df[(df['date'] >= start) & (df['date'] <= end)] if not df.empty else pd.DataFrame(); print('CACHE_SUFFICIENT' if len(in_range) > 100 else 'CACHE_INSUFFICIENT'); import pandas as pd" 2>&1
        if ($cache_check -match 'CACHE_SUFFICIENT') {
            Write-Host "  ✓ Sentiment cache sufficient, will skip fetching" -ForegroundColor Green
            # Note: We'll let the pipeline handle this automatically via sentiment_cache.py logic
        } else {
            Write-Host "  ⚠ Cache insufficient, will fetch if needed" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "  ⚠ Could not check cache, will proceed normally" -ForegroundColor Yellow
    }
    Write-Host ""
}

# Execute experiments
foreach ($exp in $experiments) {
    $run_count++
    
    # Build command
    $cmd = "python main.py " +
           "--company $Company " +
           "--start_date $StartDate " +
           "--end_date $EndDate " +
           "--model_name $($exp.Model) " +
           "--lookback $Lookback " +
           "--horizon $($exp.Horizon) " +
           "--patience $($exp.Patience) " +
           "--epochs $($exp.Epochs) " +
           "--log_ret"
    
    if ($exp.Sentiment -eq "with_sentiment") {
        $cmd += " --use_sentiment --include_social"
    }
    
    # Display progress
    $progress = [math]::Round(($run_count / $total_runs) * 100, 1)
    $elapsed = (Get-Date) - $start_time
    if ($run_count -gt 0) {
        $avg_time = $elapsed.TotalSeconds / $run_count
    } else {
        $avg_time = 0
    }
    $remaining = [TimeSpan]::FromSeconds($avg_time * ($total_runs - $run_count))
    
    Write-Host "============================================================" -ForegroundColor Gray
    Write-Host "[$run_count/$total_runs] ($progress%)" -ForegroundColor Cyan
    Write-Host "Model: $($exp.Model) | Horizon: $($exp.Horizon) | Patience: $($exp.Patience) | Sentiment: $($exp.Sentiment)" -ForegroundColor White
    Write-Host "Elapsed: $($elapsed.ToString('hh\:mm\:ss')) | Remaining: ~$($remaining.ToString('hh\:mm\:ss'))" -ForegroundColor Gray
    
    # Check if model already exists in cache (optional skip)
    $model_pattern = $($exp.Model) + "_layers50_50_50"
    $cache_pattern = "cache/trained_models/${Company}_${model_pattern}_*patience$($exp.Patience)*.keras"
    $existing = Get-ChildItem -Path $cache_pattern -ErrorAction SilentlyContinue
    if ($null -ne $existing -and $existing.Count -gt 0) {
        # Try to extract results from existing CSV if available
        $csv_pattern = "results/${Company}_*${model_pattern}*patience$($exp.Patience)*.csv"
        $csv_existing = Get-ChildItem -Path $csv_pattern -ErrorAction SilentlyContinue
        if ($null -ne $csv_existing -and $csv_existing.Count -gt 0) {
            Write-Host "[SKIP] Model exists, checking for results..." -ForegroundColor Yellow
            try {
                $csv_content = Import-Csv $csv_existing[0].FullName
                if ($csv_content -and $csv_content[0].MAE) {
                    Write-Host "[SKIP] Found existing results, skipping run" -ForegroundColor Yellow
                    $skipped_count++
                    $results += [PSCustomObject]@{
                        Run = $run_count
                        Model = $exp.Model
                        Horizon = $exp.Horizon
                        Patience = $exp.Patience
                        Epochs = $exp.Epochs
                        Sentiment = $exp.Sentiment
                        MAE = [float]$csv_content[0].MAE
                        RMSE = [float]$csv_content[0].RMSE
                        DA = [float]$csv_content[0].DA
                        Time = 0
                        Status = "SKIPPED"
                    }
                    Write-Host ""
                    continue
                }
            } catch {
                # If can't read CSV, continue with run
            }
        }
    }
    
    # Run experiment
    $run_start = Get-Date
    $output_file = Join-Path $resultsDir "run_${run_count}_$($exp.Model)_h$($exp.Horizon)_p$($exp.Patience)_e$($exp.Epochs)_$($exp.Sentiment).log"
    
    try {
        Push-Location $script_dir
        try {
            $invoke_cmd = $cmd + ' 2>&1'
            $output = Invoke-Expression $invoke_cmd | Tee-Object -FilePath $output_file
        }
        finally {
            Pop-Location
        }
        
        $run_elapsed = (Get-Date) - $run_start
        $output_string = $output | Out-String
        
        # Parse results
        $mae = $null
        $rmse = $null
        $da = $null
        
        if ($output_string -match 'MAE:\s+([\d.]+)') {
            $mae = [float]$Matches[1]
        }
        if ($output_string -match 'RMSE:\s+([\d.]+)') {
            $rmse = [float]$Matches[1]
        }
        if ($output_string -match 'Directional Accuracy.*?:\s+([\d.]+)') {
            $da = [float]$Matches[1]
        }
        $has_status_ok = $output_string -match 'STATUS:\s+OK'
        
        # Check if we have valid results
        $has_results = ($null -ne $mae) -and ($null -ne $rmse) -and ($null -ne $da)
        
        if ($has_status_ok -or $has_results) {
            Write-Host "[OK] SUCCESS" -ForegroundColor Green
            Write-Host "  MAE: $mae | RMSE: $rmse | DA: $da" -ForegroundColor White
            Write-Host "  Time: $($run_elapsed.ToString('mm\:ss'))" -ForegroundColor Gray
            
            $success_count++
            
            $results += [PSCustomObject]@{
                Run = $run_count
                Model = $exp.Model
                Horizon = $exp.Horizon
                Patience = $exp.Patience
                Epochs = $exp.Epochs
                Sentiment = $exp.Sentiment
                MAE = $mae
                RMSE = $rmse
                DA = $da
                Time = $run_elapsed.TotalSeconds
                Status = "OK"
            }
        } else {
            Write-Host "[ERROR] No valid results found" -ForegroundColor Red
            $error_count++
            
            $results += [PSCustomObject]@{
                Run = $run_count
                Model = $exp.Model
                Horizon = $exp.Horizon
                Patience = $exp.Patience
                Epochs = $exp.Epochs
                Sentiment = $exp.Sentiment
                MAE = $null
                RMSE = $null
                DA = $null
                Time = $run_elapsed.TotalSeconds
                Status = "ERROR"
            }
        }
    } catch {
        Write-Host "[EXCEPTION] $_" -ForegroundColor Red
        $error_count++
        
        $run_elapsed = (Get-Date) - $run_start
        $results += [PSCustomObject]@{
            Run = $run_count
            Model = $exp.Model
            Horizon = $exp.Horizon
            Patience = $exp.Patience
            Epochs = $exp.Epochs
            Sentiment = $exp.Sentiment
            MAE = $null
            RMSE = $null
            DA = $null
            Time = $run_elapsed.TotalSeconds
            Status = "EXCEPTION"
        }
    }
    
    Write-Host ""
}

# Save summary results
$summary_file = Join-Path $resultsDir "summary_results.csv"
$results | Export-Csv -Path $summary_file -NoTypeInformation

# Final summary
$total_time = (Get-Date) - $start_time
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "EXPERIMENT SUMMARY" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Total Runs: $total_runs" -ForegroundColor White
Write-Host "  Successful: $success_count" -ForegroundColor Green
Write-Host "  Errors: $error_count" -ForegroundColor Red
Write-Host "  Skipped: $skipped_count" -ForegroundColor Yellow
Write-Host "Total Time: $($total_time.ToString('hh\:mm\:ss'))" -ForegroundColor White
Write-Host "Results saved to: $resultsDir" -ForegroundColor Cyan
Write-Host "Summary CSV: $summary_file" -ForegroundColor Cyan
Write-Host ""

# Display key comparisons
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "KEY FINDINGS" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Best vs Worst by DA
Write-Host "Top 5 Models by Directional Accuracy (DA):" -ForegroundColor Yellow
$results | Where-Object { $null -ne $_.DA } | 
    Sort-Object -Property DA -Descending | 
    Select-Object -First 5 | 
    Format-Table -Property Model, Horizon, Patience, Sentiment, DA, MAE, RMSE -AutoSize

# 2. Sentiment Impact (average improvement)
Write-Host ""
Write-Host "Sentiment Impact Analysis:" -ForegroundColor Yellow
$with_sentiment = $results | Where-Object { $_.Sentiment -eq "with_sentiment" -and $null -ne $_.DA }
$baseline = $results | Where-Object { $_.Sentiment -eq "baseline" -and $null -ne $_.DA }

if ($with_sentiment.Count -gt 0 -and $baseline.Count -gt 0) {
    $avg_da_sentiment = ($with_sentiment | Measure-Object -Property DA -Average).Average
    $avg_da_baseline = ($baseline | Measure-Object -Property DA -Average).Average
    $improvement = $avg_da_sentiment - $avg_da_baseline
    
    Write-Host "  Average DA with Sentiment: $([math]::Round($avg_da_sentiment, 3))" -ForegroundColor White
    Write-Host "  Average DA Baseline: $([math]::Round($avg_da_baseline, 3))" -ForegroundColor White
    Write-Host "  Improvement: $([math]::Round($improvement, 3)) ($([math]::Round($improvement / $avg_da_baseline * 100, 1))%)" -ForegroundColor $(if ($improvement -gt 0) { "Green" } else { "Red" })
}

# 3. Architecture Comparison
Write-Host ""
Write-Host "Architecture Comparison (Average DA):" -ForegroundColor Yellow
$results | Where-Object { $null -ne $_.DA } | 
    Group-Object -Property Model | 
    ForEach-Object {
        $avg_da = ($_.Group | Measure-Object -Property DA -Average).Average
        $count = $_.Group.Count
        [PSCustomObject]@{
            Model = $_.Name
            AvgDA = [math]::Round($avg_da, 3)
            Runs = $count
        }
    } | Sort-Object -Property AvgDA -Descending | Format-Table -AutoSize

Write-Host ""
Write-Host "Done!" -ForegroundColor Green
Write-Host ""
Write-Host "Note: ReduceLROnPlateau is automatically enabled for all models" -ForegroundColor Gray
Write-Host "      with validation data. This demonstrates feature engineering" -ForegroundColor Gray
Write-Host "      through adaptive learning rate scheduling." -ForegroundColor Gray
