# bundle-sidecar-win.ps1
# Bundles the Python analysis engine into a Windows executable using PyInstaller.
# The output is placed in the Tauri sidecar directory for bundling.

param(
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$AnalysisDir = Join-Path $ProjectRoot "packages" "analysis"
$TauriDir = Join-Path $ProjectRoot "apps" "desktop" "src-tauri"

if (-not $OutputDir) {
    $OutputDir = Join-Path $TauriDir "sidecars"
}

Write-Host "=== AI DJ Assist - Windows Sidecar Bundler ===" -ForegroundColor Cyan
Write-Host "Analysis dir: $AnalysisDir"
Write-Host "Output dir:   $OutputDir"

# Ensure output directory exists
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

# Check Python
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "[ERROR] Python not found. Install Python >= 3.11" -ForegroundColor Red
    exit 1
}

$pyVersion = & python --version 2>&1
Write-Host "[OK] $pyVersion" -ForegroundColor Green

# Create/activate virtual environment if needed
$VenvDir = Join-Path $AnalysisDir ".venv-win"
if (-not (Test-Path $VenvDir)) {
    Write-Host "--- Creating virtual environment ---"
    & python -m venv $VenvDir
}

$ActivateScript = Join-Path $VenvDir "Scripts" "Activate.ps1"
. $ActivateScript

# Install dependencies
Write-Host "--- Installing dependencies ---"
& pip install -e "$AnalysisDir[audio]" --quiet
& pip install pyinstaller --quiet

# Run PyInstaller
Write-Host "--- Building sidecar executable ---"
$SpecFile = Join-Path $AnalysisDir "ai-dj-analysis.spec"

Push-Location $AnalysisDir
try {
    & pyinstaller $SpecFile `
        --clean `
        --distpath $OutputDir `
        --workpath (Join-Path $env:TEMP "ai-dj-analysis-build") `
        --noconfirm

    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] PyInstaller failed" -ForegroundColor Red
        exit 1
    }
} finally {
    Pop-Location
}

# Verify output
$ExePath = Join-Path $OutputDir "ai-dj-analysis.exe"
if (Test-Path $ExePath) {
    $size = (Get-Item $ExePath).Length / 1MB
    Write-Host ""
    Write-Host "[OK] Sidecar built: $ExePath ($([math]::Round($size, 1)) MB)" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Expected output not found: $ExePath" -ForegroundColor Red
    exit 1
}

# Tauri expects platform-specific binary naming: ai-dj-analysis-x86_64-pc-windows-msvc.exe
$TauriExeName = "ai-dj-analysis-x86_64-pc-windows-msvc.exe"
$TauriExePath = Join-Path $OutputDir $TauriExeName
Copy-Item $ExePath $TauriExePath -Force
Write-Host "[OK] Copied to Tauri sidecar name: $TauriExeName" -ForegroundColor Green

Write-Host ""
Write-Host "=== Sidecar bundle complete ===" -ForegroundColor Cyan
