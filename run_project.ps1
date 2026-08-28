$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$ports = @(8000, 8501)
foreach ($port in $ports) {
    $procIds = @(Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique)
    if ($procIds.Count -gt 0) {
        foreach ($procId in $procIds) {
            try {
                taskkill /F /PID $procId 2>$null | Out-Null
                Write-Host "Stopped stale process on port $port (PID $procId)"
            } catch {
                Write-Host "Could not stop PID $procId on port $port"
            }
        }
    }
}

$backendLog = Join-Path $projectRoot 'backend_run.log'
$backendErr = Join-Path $projectRoot 'backend_run.err'
$streamlitLog = Join-Path $projectRoot 'streamlit_run.log'
$streamlitErr = Join-Path $projectRoot 'streamlit_run.err'

foreach ($logFile in @($backendLog, $backendErr, $streamlitLog, $streamlitErr)) {
    if (Test-Path $logFile) {
        Remove-Item $logFile -Force
    }
}

Write-Host 'Starting backend...'
$backend = Start-Process -FilePath 'python' -ArgumentList '-m', 'uvicorn', 'backend.main:app', '--host', '0.0.0.0', '--port', '8000' -WorkingDirectory $projectRoot -PassThru -RedirectStandardOutput $backendLog -RedirectStandardError $backendErr

$backendReady = $false
for ($attempt = 0; $attempt -lt 60; $attempt++) {
    try {
        $resp = Invoke-WebRequest -Uri 'http://localhost:8000/health' -UseBasicParsing -TimeoutSec 3
        if ($resp.StatusCode -eq 200) {
            $backendReady = $true
            Write-Host "Backend ready: $($resp.StatusCode)"
            Write-Host $resp.Content
            break
        }
    } catch {
        Start-Sleep -Seconds 1
    }
}

if (-not $backendReady) {
    throw 'Backend did not become ready on port 8000.'
}

Write-Host 'Starting Streamlit...'
$streamlit = Start-Process -FilePath 'python' -ArgumentList '-m', 'streamlit', 'run', 'frontend/app.py', '--server.address', '0.0.0.0', '--server.port', '8501' -WorkingDirectory $projectRoot -PassThru -RedirectStandardOutput $streamlitLog -RedirectStandardError $streamlitErr

$frontendReady = $false
for ($attempt = 0; $attempt -lt 60; $attempt++) {
    try {
        $resp = Invoke-WebRequest -Uri 'http://localhost:8501/' -UseBasicParsing -TimeoutSec 3
        if ($resp.StatusCode -eq 200) {
            $frontendReady = $true
            Write-Host "Frontend ready: $($resp.StatusCode)"
            Write-Host $resp.Content.Substring(0, 120)
            break
        }
    } catch {
        Start-Sleep -Seconds 1
    }
}

if (-not $frontendReady) {
    throw 'Frontend did not become ready on port 8501.'
}

Write-Host 'Both services are running.'
Write-Host "Backend PID: $($backend.Id)"
Write-Host "Streamlit PID: $($streamlit.Id)"
Write-Host "Open: http://localhost:8501/"
