<#
.SYNOPSIS
    One-time setup of Mars Terrain Intelligence on Windows (Windows PowerShell 5.1+ or PowerShell 7).

.DESCRIPTION
    Checks Python (>= 3.10) and Node.js (>= 18), creates .venv, installs PyTorch (CUDA 12.4 build when an
    NVIDIA GPU is detected, otherwise the CPU build), installs the Python requirements and the frontend
    packages, creates the local data folders, copies .env.example to .env, initialises the SQLite database
    and prints a hardware summary. Safe to run again: existing pieces are reused.

    If script execution is blocked, run it as:
        powershell -ExecutionPolicy Bypass -File .\setup.ps1

.PARAMETER Cuda
    Install the CUDA 12.4 PyTorch build even if nvidia-smi is not found.

.PARAMETER CpuOnly
    Install the CPU-only PyTorch build even if an NVIDIA GPU is present.

.PARAMETER WithE2E
    Also download the Playwright Chromium browser needed by the end-to-end tests (npm run e2e).
#>
[CmdletBinding()]
param(
    [switch]$Cuda,
    [switch]$CpuOnly,
    [switch]$WithE2E
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$TorchCudaIndex = 'https://download.pytorch.org/whl/cu124'
$TorchCpuIndex = 'https://download.pytorch.org/whl/cpu'
$VenvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$FrontendDir = Join-Path $PSScriptRoot 'frontend'

# ----------------------------------------------------------------------------- output helpers
function Write-Step([string]$Message) { Write-Host ''; Write-Host "==> $Message" -ForegroundColor Cyan }
function Write-Ok([string]$Message) { Write-Host "    [ok] $Message" -ForegroundColor Green }
function Write-Info([string]$Message) { Write-Host "    $Message" -ForegroundColor Gray }
function Write-Warn([string]$Message) { Write-Host "    [warning] $Message" -ForegroundColor Yellow }
function Fail([string]$Message) {
    Write-Host ''
    Write-Host "[setup failed] $Message" -ForegroundColor Red
    exit 1
}

# Runs a native program, lets its output through, and stops with a clear message on a non-zero exit code.
# The error preference is relaxed during the call because Windows PowerShell 5.1 can turn harmless stderr
# output (pip/npm warnings) into terminating errors when it is set to Stop.
function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$ArgumentList = @(),
        [Parameter(Mandatory = $true)][string]$FailMessage
    )
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $FilePath @ArgumentList
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($code -ne 0) { Fail "$FailMessage (exit code $code)" }
}

# Runs a native program and returns its trimmed stdout, or $null if it fails. Never throws.
function Get-NativeOutput {
    param([Parameter(Mandatory = $true)][string]$FilePath, [string[]]$ArgumentList = @())
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $out = & $FilePath @ArgumentList 2>$null
        if ($LASTEXITCODE -ne 0) { return $null }
        if ($null -eq $out) { return $null }
        return (($out | Out-String).Trim())
    } catch {
        return $null
    } finally {
        $ErrorActionPreference = $previous
    }
}

function Resolve-Tool([string[]]$Names) {
    foreach ($n in $Names) {
        $cmd = Get-Command $n -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($cmd) { return $cmd.Source }
    }
    return $null
}

if ($Cuda -and $CpuOnly) { Fail 'Use either -Cuda or -CpuOnly, not both.' }

Write-Host 'Mars Terrain Intelligence - setup' -ForegroundColor White
Write-Host 'Independent student research project. Not a NASA system and not connected to any rover.' -ForegroundColor DarkGray

# ----------------------------------------------------------------------------- 1. Python
Write-Step 'Checking Python (3.10 or newer required)'
$pythonCmd = $null
$candidates = @(
    [pscustomobject]@{ Exe = 'py'; Args = @('-3') },
    [pscustomobject]@{ Exe = 'python'; Args = @() }
)
foreach ($c in $candidates) {
    if (-not (Get-Command $c.Exe -ErrorAction SilentlyContinue)) { continue }
    $ver = Get-NativeOutput -FilePath $c.Exe -ArgumentList ($c.Args + @('-c', "import sys; print('%d.%d.%d' % sys.version_info[:3])"))
    if (-not $ver -or $ver -notmatch '^\d+\.\d+\.\d+$') { continue }   # e.g. the Microsoft Store alias
    $parsed = [version]$ver
    if ($parsed -ge [version]'3.10.0') {
        $pythonCmd = $c
        Write-Ok ("Python $ver (" + (($c.Exe + ' ' + ($c.Args -join ' ')).Trim()) + ')')
        break
    }
    Write-Warn "'$($c.Exe)' is Python $ver, which is too old."
}
if (-not $pythonCmd) {
    Fail ('Python 3.10 or newer was not found. Install it from https://www.python.org/downloads/ ' +
          '(tick "Add python.exe to PATH" and keep the "py" launcher), open a new terminal and run setup.ps1 again.')
}

# ----------------------------------------------------------------------------- 2. Node.js and npm
Write-Step 'Checking Node.js (18 or newer) and npm'
$node = Resolve-Tool @('node.exe', 'node')
if (-not $node) { Fail 'Node.js was not found. Install the LTS version from https://nodejs.org/ and open a new terminal.' }
$nodeVer = Get-NativeOutput -FilePath $node -ArgumentList @('--version')
if (-not ($nodeVer -match '^v(\d+)\.')) { Fail "Could not read the Node.js version from '$node'." }
if ([int]$Matches[1] -lt 18) { Fail "Node.js $nodeVer is too old. Install Node.js 18 or newer (LTS) from https://nodejs.org/." }
Write-Ok "Node.js $nodeVer"
# Prefer npm.cmd: it works even where the execution policy blocks npm.ps1.
$npm = Resolve-Tool @('npm.cmd', 'npm')
if (-not $npm) { Fail 'npm was not found. It is installed together with Node.js; reinstall Node.js LTS.' }
$npmVer = Get-NativeOutput -FilePath $npm -ArgumentList @('--version')
if (-not $npmVer) { Fail "npm at '$npm' did not run." }
Write-Ok "npm $npmVer"
$npx = Resolve-Tool @('npx.cmd', 'npx')

# ----------------------------------------------------------------------------- 3. virtual environment
Write-Step 'Creating the Python virtual environment (.venv)'
if (Test-Path -LiteralPath $VenvPython) {
    Write-Ok '.venv already exists; reusing it'
} else {
    Invoke-Checked -FilePath $pythonCmd.Exe -ArgumentList ($pythonCmd.Args + @('-m', 'venv', '.venv')) `
        -FailMessage 'Could not create the virtual environment'
    if (-not (Test-Path -LiteralPath $VenvPython)) { Fail ".venv was created but $VenvPython is missing." }
    Write-Ok '.venv created'
}

Write-Step 'Upgrading pip'
Invoke-Checked -FilePath $VenvPython -ArgumentList @('-m', 'pip', 'install', '--upgrade', 'pip') `
    -FailMessage 'pip upgrade failed. Check your internet connection or proxy settings'
Write-Ok 'pip is up to date'

# ----------------------------------------------------------------------------- 4. PyTorch
Write-Step 'Selecting the PyTorch build'
$gpuName = $null
$smi = Resolve-Tool @('nvidia-smi.exe', 'nvidia-smi')
if ($smi) {
    $gpuName = Get-NativeOutput -FilePath $smi -ArgumentList @('--query-gpu=name,driver_version', '--format=csv,noheader')
}
if ($CpuOnly) {
    $want = 'cpu'; Write-Info '-CpuOnly given: installing the CPU build.'
} elseif ($Cuda) {
    $want = 'cuda'; Write-Info '-Cuda given: installing the CUDA 12.4 build.'
} elseif ($gpuName) {
    $want = 'cuda'; Write-Info "NVIDIA GPU detected: $gpuName"
} else {
    $want = 'cpu'; Write-Info 'No NVIDIA GPU detected (nvidia-smi not found or not working): installing the CPU build.'
}
$index = if ($want -eq 'cuda') { $TorchCudaIndex } else { $TorchCpuIndex }

# Reuse an existing torch if it is already the right variant; otherwise replace it.
$current = Get-NativeOutput -FilePath $VenvPython -ArgumentList @('-c', "import torch; print('cuda' if torch.version.cuda else 'cpu')")
if ($current -eq $want) {
    Write-Ok "PyTorch ($want build) is already installed"
} else {
    if ($current) {
        Write-Info "Replacing the installed $current build of PyTorch with the $want build."
        Invoke-Checked -FilePath $VenvPython -ArgumentList @('-m', 'pip', 'uninstall', '-y', 'torch', 'torchvision') `
            -FailMessage 'Could not uninstall the existing PyTorch build'
    }
    Write-Info "Installing torch + torchvision from $index (this can take several minutes)"
    Invoke-Checked -FilePath $VenvPython -ArgumentList @('-m', 'pip', 'install', 'torch', 'torchvision', '--index-url', $index) `
        -FailMessage "PyTorch installation from $index failed. Check your connection; on an older GPU driver try -CpuOnly"
    Write-Ok "PyTorch ($want build) installed"
}

# ----------------------------------------------------------------------------- 5. Python requirements
Write-Step 'Installing Python requirements (requirements.txt)'
Invoke-Checked -FilePath $VenvPython -ArgumentList @('-m', 'pip', 'install', '-r', 'requirements.txt') `
    -FailMessage 'pip install -r requirements.txt failed'
Write-Ok 'Python requirements installed'

# ----------------------------------------------------------------------------- 6. frontend packages
Write-Step 'Installing frontend packages'
if (-not (Test-Path -LiteralPath (Join-Path $FrontendDir 'package.json'))) { Fail "frontend\package.json was not found in $FrontendDir." }
Push-Location -LiteralPath $FrontendDir
try {
    if (Test-Path -LiteralPath 'package-lock.json') {
        Invoke-Checked -FilePath $npm -ArgumentList @('ci') -FailMessage 'npm ci failed in frontend\'
    } else {
        Invoke-Checked -FilePath $npm -ArgumentList @('install') -FailMessage 'npm install failed in frontend\'
    }
} finally {
    Pop-Location
}
Write-Ok 'Frontend packages installed'

# ----------------------------------------------------------------------------- 7. folders and .env
Write-Step 'Creating local data folders'
foreach ($d in @('data\raw\imports', 'data\processed', 'data\analyses', 'checkpoints', 'experiments')) {
    New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot $d) | Out-Null
    Write-Ok $d
}

Write-Step 'Configuration file (.env)'
if (Test-Path -LiteralPath '.env') {
    Write-Ok '.env already exists; left unchanged'
} elseif (Test-Path -LiteralPath '.env.example') {
    Copy-Item -LiteralPath '.env.example' -Destination '.env'
    Write-Ok '.env created from .env.example'
} else {
    Write-Warn '.env.example not found; built-in defaults will be used.'
}

# ----------------------------------------------------------------------------- 8. database
Write-Step 'Initialising the SQLite database'
Invoke-Checked -FilePath $VenvPython -ArgumentList @('-c',
    "from backend.app.core.config import get_settings; from backend.app.database.db import Database; s=get_settings(); s.ensure_dirs(); Database(s.db_url); print('database ready')") `
    -FailMessage 'Database initialisation failed'

# ----------------------------------------------------------------------------- 9. hardware diagnostics
Write-Step 'Hardware diagnostics'
Invoke-Checked -FilePath $VenvPython -ArgumentList @('-c',
    "import os, torch, psutil; c=torch.cuda.is_available(); print('    torch', torch.__version__, '| CUDA build:', torch.version.cuda); print('    CUDA available:', c); print('    GPU:', torch.cuda.get_device_name(0) if c else 'none'); print('    CPU cores (logical):', os.cpu_count()); print('    RAM (GB):', round(psutil.virtual_memory().total / 1e9, 1))") `
    -FailMessage 'Could not import torch/psutil from .venv'
if ($want -eq 'cuda') {
    $cudaOk = Get-NativeOutput -FilePath $VenvPython -ArgumentList @('-c', 'import torch; print(torch.cuda.is_available())')
    if ($cudaOk -ne 'True') {
        Write-Warn 'The CUDA build is installed but PyTorch cannot use the GPU. Update the NVIDIA driver, or run setup.ps1 -CpuOnly.'
    }
}

# ----------------------------------------------------------------------------- 10. optional e2e browser
if ($WithE2E) {
    Write-Step 'Installing the Playwright Chromium browser (for npm run e2e)'
    if (-not $npx) { Fail 'npx was not found. It is installed together with Node.js.' }
    Push-Location -LiteralPath $FrontendDir
    try {
        Invoke-Checked -FilePath $npx -ArgumentList @('playwright', 'install', 'chromium') -FailMessage 'Playwright browser installation failed'
    } finally {
        Pop-Location
    }
    Write-Ok 'Chromium for Playwright installed'
}

# ----------------------------------------------------------------------------- done
Write-Host ''
Write-Host 'Setup complete.' -ForegroundColor Green
Write-Host 'Next steps:' -ForegroundColor White
Write-Host '  1. Start the app:       .\start.ps1        (add -Open to open the browser)'
Write-Host '  2. Open the web app:    http://localhost:5173'
Write-Host '  3. API documentation:   http://127.0.0.1:8000/docs'
Write-Host '  4. Stop everything:     .\stop.ps1'
Write-Host '  Tests: .venv\Scripts\python -m pytest   (see docs\testing.md)'
