# start.ps1 — Launch the complete FinSentAI system
# Run from project root: .\start.ps1

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   FinSentAI — Financial Intelligence     " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# --- 1. Install FastAPI backend deps (into venv if present) ---
Write-Host "[1/3] Installing backend dependencies..." -ForegroundColor Yellow

$venvPython = ""
if (Test-Path ".venv\Scripts\python.exe") {
    $venvPython = ".venv\Scripts\python.exe"
} elseif (Test-Path "venv\Scripts\python.exe") {
    $venvPython = "venv\Scripts\python.exe"
} else {
    $venvPython = "python"
}

Write-Host "Using Python: $venvPython" -ForegroundColor Gray
& $venvPython -m pip install fastapi uvicorn pydantic vaderSentiment --quiet

# --- 2. Start backend in background ---
Write-Host ""
Write-Host "[2/3] Starting FastAPI backend on http://localhost:8000 ..." -ForegroundColor Yellow
$backend = Start-Process -FilePath $venvPython -ArgumentList "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload" -PassThru -NoNewWindow
Write-Host "Backend PID: $($backend.Id)" -ForegroundColor Gray
Start-Sleep -Seconds 3

# --- 3. Start React dashboard ---
Write-Host ""
Write-Host "[3/3] Starting React dashboard on http://localhost:5173 ..." -ForegroundColor Yellow
Write-Host ""
Write-Host "✅ Dashboard will open at: http://localhost:5173" -ForegroundColor Green
Write-Host "✅ API health check:       http://localhost:8000/health" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop the dashboard (backend runs in separate process)." -ForegroundColor Gray
Write-Host ""

Set-Location dashboard
npm run dev
