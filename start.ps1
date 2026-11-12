<#
.SYNOPSIS
    Starts the Mars Terrain Intelligence backend (port 8000) and frontend (port 5173) in the background.

.DESCRIPTION
    Both processes run hidden. Their output goes to .run\backend.log, .run\backend.err.log,
    .run\frontend.log and .run\frontend.err.log, and their process ids to .run\backend.pid and
    .run\frontend.pid. Use .\stop.ps1 to stop them.

.PARAMETER Open
    Open the web app in the default browser once it is ready.

.PARAMETER TimeoutSeconds
    How long to wait for each server to answer (default 60).
#>
[CmdletBinding()]
param(
    [switch]$Open,
    [int]$TimeoutSeconds = 60
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$BackendPort = 8000
$FrontendPort = 5173
$RunDir = Join-Path $PSScriptRoot '.run'
$VenvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$FrontendDir = Join-Path $PSScriptRoot 'frontend'
$HealthUrl = "http://127.0.0.1:$BackendPort/api/health"
$FrontendUrl = "http://127.0.0.1:$FrontendPort"

function Write-Step([string]$Message) { Write-Host "==> $Message" -ForegroundColor Cyan }
function Write-Ok([string]$Message) { Write-Host "    [ok] $Message" -ForegroundColor Green }
function Write-Warn([string]$Message) { Write-Host "    [warning] $Message" -ForegroundColor Yellow }
function Fail([string]$Message) {
    Write-Host ''
    Write-Host "[start failed] $Message" -ForegroundColor Red
    exit 1
}

# True when something is listening on the port (any address). Uses Get-NetTCPConnection when available
# and always also tries a direct connection, which works where that cmdlet is missing or restricted.
function Test-PortInUse([int]$Port) {
    if (Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue) {
        $listening = $null
        try { $listening = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue } catch { $listening = $null }
        if ($listening) { return $true }
    }
    foreach ($address in @('127.0.0.1', '::1')) {
        $client = $null
        try {
            $client = New-Object System.Net.Sockets.TcpClient
            $async = $client.BeginConnect($address, $Port, $null, $null)
            if ($async.AsyncWaitHandle.WaitOne(400)) {
                try { $client.EndConnect($async); return $true } catch { }
            }
        } catch {
        } finally {
            if ($client) { $client.Close() }
        }
    }
    return $false
}

function Test-ProcessAlive([int]$ProcessId) {
    return [bool](Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)
}

function Stop-Tree([int]$ProcessId) {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & taskkill.exe /PID $ProcessId /T /F 2>&1 | Out-Null } catch { } finally { $ErrorActionPreference = $previous }
}

function Wait-Http([string]$Url, [System.Diagnostics.Process]$Process, [int]$Seconds) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        if ($Process.HasExited) { return $false }
        try {
            $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 400) { return $true }
        } catch { }
        Start-Sleep -Milliseconds 750
    }
    return $false
}

function Show-LogTail([string]$Path) {
    if (Test-Path -LiteralPath $Path) {
        $lines = Get-Content -LiteralPath $Path -Tail 25 -ErrorAction SilentlyContinue
        if ($lines) {
            Write-Host "---- last lines of $Path ----" -ForegroundColor DarkGray
            $lines | ForEach-Object { Write-Host "    $_" }
        }
    }
}

# ----------------------------------------------------------------------------- prerequisites
if (-not (Test-Path -LiteralPath $VenvPython)) {
    Fail 'The Python environment (.venv) is missing. Run .\setup.ps1 first.'
}
if (-not (Test-Path -LiteralPath (Join-Path $FrontendDir 'node_modules'))) {
    Fail 'Frontend packages (frontend\node_modules) are missing. Run .\setup.ps1 first.'
}
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

# Refuse to start twice: a live PID file means the app is already running.
foreach ($name in @('backend', 'frontend')) {
    $pidFile = Join-Path $RunDir "$name.pid"
    if (Test-Path -LiteralPath $pidFile) {
        $oldPid = 0
        [void][int]::TryParse("$(Get-Content -LiteralPath $pidFile -Raw)".Trim(), [ref]$oldPid)
        if ($oldPid -gt 0 -and (Test-ProcessAlive $oldPid)) {
            Fail "The $name appears to be running already (PID $oldPid). Run .\stop.ps1 first."
        }
        Remove-Item -LiteralPath $pidFile -Force   # stale file from an earlier session
    }
}

foreach ($port in @($BackendPort, $FrontendPort)) {
    if (Test-PortInUse $port) {
        Fail ("Port $port is already in use by another program. Run .\stop.ps1, or close the program " +
              "using it (find it with: Get-NetTCPConnection -LocalPort $port -State Listen).")
    }
}

$backendLog = Join-Path $RunDir 'backend.log'
$backendErr = Join-Path $RunDir 'backend.err.log'
$frontendLog = Join-Path $RunDir 'frontend.log'
$frontendErr = Join-Path $RunDir 'frontend.err.log'

# ----------------------------------------------------------------------------- backend
Write-Step "Starting backend on port $BackendPort"
$backend = Start-Process -FilePath $VenvPython `
    -ArgumentList @('-m', 'uvicorn', 'backend.app.main:app', '--host', '127.0.0.1', '--port', "$BackendPort") `
    -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $backendLog -RedirectStandardError $backendErr
Set-Content -LiteralPath (Join-Path $RunDir 'backend.pid') -Value $backend.Id -Encoding Ascii

Write-Host "    waiting for $HealthUrl (up to $TimeoutSeconds s)"
if (-not (Wait-Http -Url $HealthUrl -Process $backend -Seconds $TimeoutSeconds)) {
    Show-LogTail $backendErr
    Stop-Tree $backend.Id
    Remove-Item -LiteralPath (Join-Path $RunDir 'backend.pid') -Force -ErrorAction SilentlyContinue
    Fail "The backend did not become ready. Full logs: $backendLog and $backendErr"
}
Write-Ok "backend ready (PID $($backend.Id))"

# ----------------------------------------------------------------------------- frontend
# npm is a .cmd script on Windows, so it is started through cmd.exe. stop.ps1 kills the whole
# process tree (cmd -> npm -> node/vite).
Write-Step "Starting frontend on port $FrontendPort"
$frontend = Start-Process -FilePath $env:ComSpec `
    -ArgumentList @('/c', 'npm', 'run', 'dev', '--', '--host', '127.0.0.1', '--port', "$FrontendPort", '--strictPort') `
    -WorkingDirectory $FrontendDir -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $frontendLog -RedirectStandardError $frontendErr
Set-Content -LiteralPath (Join-Path $RunDir 'frontend.pid') -Value $frontend.Id -Encoding Ascii

Write-Host "    waiting for $FrontendUrl (up to $TimeoutSeconds s)"
if (-not (Wait-Http -Url $FrontendUrl -Process $frontend -Seconds $TimeoutSeconds)) {
    Show-LogTail $frontendErr
    Show-LogTail $frontendLog
    Stop-Tree $frontend.Id
    Remove-Item -LiteralPath (Join-Path $RunDir 'frontend.pid') -Force -ErrorAction SilentlyContinue
    Write-Warn 'The backend is still running; stop it with .\stop.ps1.'
    Fail "The frontend did not become ready. Full logs: $frontendLog and $frontendErr"
}
Write-Ok "frontend ready (PID $($frontend.Id))"

# ----------------------------------------------------------------------------- done
Write-Host ''
Write-Host 'Mars Terrain Intelligence is running.' -ForegroundColor Green
Write-Host "  Web app:   http://localhost:$FrontendPort   (or $FrontendUrl)"
Write-Host "  API docs:  http://127.0.0.1:$BackendPort/docs"
Write-Host "  Logs:      $RunDir"
Write-Host '  Stop with: .\stop.ps1'

if ($Open) { Start-Process "http://localhost:$FrontendPort" }
