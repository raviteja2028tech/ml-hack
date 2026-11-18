# =====================================================================
# PathSnap Demo Diagnostics & Pre-flight Checker
# =====================================================================
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location -Path $ProjectRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "            PATHSNAP DEMO PRE-FLIGHT DIAGNOSTICS            " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Model Check
$ModelPath = Join-Path $ProjectRoot "artifacts\models\lightgbm.joblib"
if (Test-Path $ModelPath) {
    $sizeMb = [math]::Round((Get-Item $ModelPath).Length / 1MB, 2)
    Write-Host "  [PASS] Model Artifact: lightgbm.joblib ($sizeMb MB)" -ForegroundColor Green
} else {
    Write-Host "  [FAIL] Model Artifact: Missing at $ModelPath" -ForegroundColor Red
}

# 2. Dataset Check
$TrainDir = Join-Path $ProjectRoot "data\raw\Student folder\train"
$TestDir = Join-Path $ProjectRoot "data\raw\Student folder\student_test"
$TrainCount = (Get-ChildItem -Path $TrainDir -Directory -ErrorAction SilentlyContinue).Count
$TestCount = (Get-ChildItem -Path $TestDir -Directory -ErrorAction SilentlyContinue).Count

if ($TrainCount -ge 80 -and $TestCount -ge 10) {
    Write-Host "  [PASS] Datasets Available: $TrainCount train trips, $TestCount student-test trips" -ForegroundColor Green
} else {
    Write-Host "  [FAIL] Datasets: Found $TrainCount train and $TestCount test trips" -ForegroundColor Red
}

# 3. Process Check & Verification Script
Write-Host "`nRunning real in-process end-to-end verification..." -ForegroundColor Cyan
$verifyOutput = & $VenvPython "scripts\verify_real_trips.py"
Write-Host $verifyOutput

Write-Host "`nDiagnostics Complete." -ForegroundColor Green
