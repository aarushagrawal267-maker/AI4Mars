<#
.SYNOPSIS
    Stops the Mars Terrain Intelligence backend and frontend started by start.ps1.

.DESCRIPTION
    1. If a training run is active, asks the backend to stop it gracefully (it stops after the current
       batch) and waits briefly.
    2. Kills the process trees recorded in .run\backend.pid and .run\frontend.pid
       (taskkill /T /F: npm/cmd spawn node, and the backend may have spawned a training process).
    3. Frees anything still listening on ports 8000/5173 that is a python or node process.
       Other programs on those ports are reported, never touched.
    4. Removes the PID files.

    Note: taskkill /F is forceful. A training process killed this way cannot finish its current epoch,
    but checkpoints from completed epochs (checkpoints\<id>\best.pt and last.pt) are kept regardless,
    because they are written atomically (temporary file, then rename). The next backend start marks
    such a run as "interrupted".

.PARAMETER TrainingWaitSeconds
    How long to wait for an active training run to stop gracefully (default 30). 0 skips this step.
#>
[CmdletBinding()]
param(
    [int]$TrainingWaitSeconds = 30
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$Ports = @(8000, 5173)
$RunDir = Join-Path $PSScriptRoot '.run'
$ApiBase = 'http://127.0.0.1:8000/api'
$Stopped = New-Object System.Collections.Generic.List[string]

function Write-Step([string]$Message) { Write-Host "==> $Message" -ForegroundColor Cyan }
function Write-Ok([string]$Message) { Write-Host "    [ok] $Message" -ForegroundColor Green }
function Write-Info([string]$Message) { Write-Host "    $Message" -ForegroundColor Gray }
function Write-Warn([string]$Message) { Write-Host "    [warning] $Message" -ForegroundColor Yellow }

function Stop-Tree([int]$ProcessId) {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & taskkill.exe /PID $ProcessId /T /F 2>&1 | Out-Null } catch { } finally { $ErrorActionPreference = $previous }
}

# Process ids listening on a TCP port. Uses Get-NetTCPConnection, or parses netstat where it is unavailable.
function Get-ListeningProcessIds([int]$Port) {
    $ids = @()
    if (Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue) {
        try {
            $ids = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue |
                     Select-Object -ExpandProperty OwningProcess)
        } catch { $ids = @() }
    } else {
        $previous = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            # Listening sockets have a foreign address of 0.0.0.0:0 or [::]:0 (this avoids the
            # localised "LISTENING" state text).
            $lines = @(& netstat.exe -ano -p TCP) + @(& netstat.exe -ano -p TCPv6)
            foreach ($line in $lines) {
                if ($line -match '^\s*TCP\s+\S+:(\d+)\s+(0\.0\.0\.0:0|\[::\]:0)\s+\S+\s+(\d+)\s*$') {
                    if ([int]$Matches[1] -eq $Port) { $ids += [int]$Matches[3] }
                }
            }
        } catch { } finally { $ErrorActionPreference = $previous }
    }
    return @($ids | Where-Object { $_ -gt 0 } | Select-Object -Unique)
}

# ----------------------------------------------------------------------------- 1. graceful training stop
if ($TrainingWaitSeconds -gt 0) {
    $status = $null
    try { $status = Invoke-RestMethod -Uri "$ApiBase/training/status" -TimeoutSec 3 } catch { $status = $null }
    if ($status -and $status.active) {
        Write-Step "A training run is active ($($status.experiment_id)); asking it to stop after the current batch"
        try {
            Invoke-RestMethod -Method Post -Uri "$ApiBase/training/stop" -TimeoutSec 10 | Out-Null
            $deadline = (Get-Date).AddSeconds($TrainingWaitSeconds)
            do {
                Start-Sleep -Seconds 1
                try { $status = Invoke-RestMethod -Uri "$ApiBase/training/status" -TimeoutSec 3 } catch { $status = $null }
            } while ($status -and $status.active -and (Get-Date) -lt $deadline)
            if ($status -and $status.active) {
                Write-Warn 'The run did not stop in time; it will be killed. Completed-epoch checkpoints are kept.'
            } else {
                Write-Ok 'Training run stopped'
            }
        } catch {
            Write-Warn "Could not ask the run to stop ($($_.Exception.Message)); it will be killed."
        }
    }
}

# ----------------------------------------------------------------------------- 2. PID files
Write-Step 'Stopping processes started by start.ps1'
foreach ($name in @('backend', 'frontend')) {
    $pidFile = Join-Path $RunDir "$name.pid"
    if (-not (Test-Path -LiteralPath $pidFile)) {
        Write-Info "$name - no PID file"
        continue
    }
    $procId = 0
    [void][int]::TryParse("$(Get-Content -LiteralPath $pidFile -Raw)".Trim(), [ref]$procId)
    $proc = $null
    if ($procId -gt 0) { $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue }
    if ($proc -and $proc.ProcessName -notmatch '^(python|pythonw|cmd|node)$') {
        # A stale PID file whose number now belongs to an unrelated program.
        Write-Warn "$name - PID $procId now belongs to '$($proc.ProcessName)', not to this app; left running."
    } elseif ($proc) {
        Stop-Tree $procId
        $Stopped.Add("$name (PID $procId, with child processes)")
        Write-Ok "$name stopped (PID $procId)"
    } else {
        Write-Info "$name - PID $procId was not running"
    }
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
}

# ----------------------------------------------------------------------------- 3. leftover listeners
Write-Step 'Checking ports 8000 and 5173'
Start-Sleep -Milliseconds 500
foreach ($port in $Ports) {
    $ids = Get-ListeningProcessIds $port
    if (-not $ids -or $ids.Count -eq 0) {
        Write-Ok "port $port is free"
        continue
    }
    foreach ($id in $ids) {
        $proc = Get-Process -Id $id -ErrorAction SilentlyContinue
        if (-not $proc) { continue }
        if ($proc.ProcessName -match '^(python|pythonw|node)$') {
            Stop-Tree $id
            $Stopped.Add("$($proc.ProcessName) on port $port (PID $id)")
            Write-Ok "stopped $($proc.ProcessName) (PID $id) listening on port $port"
        } else {
            Write-Warn "port $port is used by '$($proc.ProcessName)' (PID $id), which was not started by this app; left running."
        }
    }
}

# ----------------------------------------------------------------------------- summary
Write-Host ''
if ($Stopped.Count -gt 0) {
    Write-Host 'Stopped:' -ForegroundColor Green
    foreach ($s in $Stopped) { Write-Host "  - $s" }
} else {
    Write-Host 'Nothing was running.' -ForegroundColor Green
}
