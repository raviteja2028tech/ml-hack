# =====================================================================
# PathSnap One-Click Demo Launcher (Single & Two-Laptop Deployment)
# =====================================================================
$ErrorActionPreference = "Stop"

# 1. Project Directory
$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location -Path $ProjectRoot

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "       PATHSNAP — MAP-MATCHING & ROUTE RECONSTRUCTION       " -ForegroundColor Cyan
Write-Host "                   ONE-CLICK DEMO LAUNCHER                  " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Workspace: $ProjectRoot" -ForegroundColor Gray

# 2. Check Virtual Environment
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "[ERROR] Virtual environment not found at .venv\Scripts\python.exe!" -ForegroundColor Red
    Write-Host "Please create a venv: python -m venv .venv and install requirements.txt" -ForegroundColor Yellow
    exit 1
}
Write-Host "[OK] Python runtime: $VenvPython" -ForegroundColor Green

# 3. Check Model Artifacts
$ModelPath = Join-Path $ProjectRoot "artifacts\models\lightgbm.joblib"
if (-not (Test-Path $ModelPath)) {
    Write-Host "[ERROR] LightGBM model artifact missing at $ModelPath!" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Model artifact verified: lightgbm.joblib" -ForegroundColor Green

# 4. Check Local IP Address
$LocalIP = "127.0.0.1"
try {
    $ipDetect = & $VenvPython -c "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('8.8.8.8', 80)); print(s.getsockname()[0]); s.close()"
    if ($ipDetect) { $LocalIP = $ipDetect.Trim() }
} catch {
    $LocalIP = "127.0.0.1"
}

# 5. Check Port 8000 Availability
$PortOccupied = $false
try {
    $con = Test-NetConnection -ComputerName "127.0.0.1" -Port 8000 -InformationLevel Quiet -WarningAction SilentlyContinue
    if ($con) { $PortOccupied = $true }
} catch {
    $PortOccupied = $false
}

if ($PortOccupied) {
    Write-Host "[WARN] Port 8000 is already in use." -ForegroundColor Yellow
    Write-Host "Testing if existing PathSnap server is responding on http://127.0.0.1:8000/health..." -ForegroundColor Gray
    try {
        $h = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get -TimeoutSec 3 -ErrorAction Stop
        if ($h.status -eq "ok" -or $h.service -match "PathSnap") {
            Write-Host "[OK] Existing PathSnap API instance is active and healthy!" -ForegroundColor Green
        }
    } catch {
        Write-Host "[ERROR] Port 8000 is occupied by another process. Please free port 8000 or terminate conflicting process." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[INFO] Starting FastAPI server on 0.0.0.0:8000..." -ForegroundColor Cyan
    $Process = Start-Process -FilePath $VenvPython -ArgumentList "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000" -PassThru -WindowStyle Normal
    Write-Host "[INFO] Process launched (PID: $($Process.Id)). Waiting for /health..." -ForegroundColor Gray

    $Ready = $false
    for ($i = 0; $i -lt 15; $i++) {
        Start-Sleep -Seconds 1
        try {
            $res = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get -TimeoutSec 2 -ErrorAction Stop
            if ($res.status -eq "ok" -or $res.model_loaded -eq $true) {
                $Ready = $true
                break
            }
        } catch {
            # continue waiting
        }
    }

    if (-not $Ready) {
        Write-Host "[ERROR] Server failed to respond within 15 seconds." -ForegroundColor Red
        exit 1
    }
}

# 6. Display URLs & Launch Browser
Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "           PATHSNAP DEMO SERVER IS READY & ONLINE!          " -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "Local UI (This Machine)  : http://localhost:8000/ui/" -ForegroundColor Yellow
Write-Host "LAN UI (Laptop 2 Client) : http://${LocalIP}:8000/ui/" -ForegroundColor Yellow
Write-Host "API Documentation        : http://localhost:8000/docs" -ForegroundColor Gray
Write-Host "Health Check             : http://localhost:8000/health" -ForegroundColor Gray
Write-Host "============================================================`n" -ForegroundColor Green

Write-Host "Opening dashboard in your default browser..." -ForegroundColor Cyan
Start-Process "http://localhost:8000/ui/"

Write-Host "`nTo stop the server, close the uvicorn terminal window or run: Stop-Process -Name python" -ForegroundColor Gray
