# =============================================================================
# DC-Ops build-box setup for WINDOWS.
#
# The QNN AOT build needs Linux x86_64. On Windows that's WSL2. This script
# ensures a WSL2 Ubuntu distro exists, then runs setup/setup.sh inside it.
#
# Usage (from a normal PowerShell, in the repo root):
#     ./setup/setup.ps1
#
# If you already have the QAIRT zip, point to it (Windows path is fine):
#     ./setup/setup.ps1 -QnnSdkZip "C:\Downloads\qairt-2.46.0.260424.zip"
# =============================================================================
param(
  [string]$Distro = "Ubuntu",
  [string]$QnnSdkZip = ""
)
$ErrorActionPreference = "Stop"

function Have-Wsl {
  try { wsl.exe --status *> $null; return $true } catch { return $false }
}

# 1. Ensure WSL is installed.
if (-not (Have-Wsl)) {
  Write-Host "WSL is not installed. Installing (this may require a reboot)..." -ForegroundColor Yellow
  wsl.exe --install -d $Distro
  Write-Host "`nWSL was just installed. REBOOT, finish the Ubuntu first-run (set a username/password)," -ForegroundColor Yellow
  Write-Host "then re-run this script." -ForegroundColor Yellow
  exit 0
}

# 2. Ensure the target distro exists.
$distros = (wsl.exe -l -q) -replace "`0","" | ForEach-Object { $_.Trim() } | Where-Object { $_ }
if ($distros -notcontains $Distro) {
  Write-Host "Installing WSL distro '$Distro'..." -ForegroundColor Yellow
  wsl.exe --install -d $Distro
  Write-Host "Finish the Ubuntu first-run (username/password), then re-run this script." -ForegroundColor Yellow
  exit 0
}

# 3. Translate this repo's path to a WSL path and run setup.sh inside WSL.
$repoWin = (Resolve-Path "$PSScriptRoot\..").Path
$repoWsl = (wsl.exe -d $Distro wslpath -a "$repoWin").Trim()
Write-Host "Repo (Windows): $repoWin"
Write-Host "Repo (WSL):     $repoWsl"

# Pass the QNN zip through as a WSL path if provided.
$envPrefix = ""
if ($QnnSdkZip) {
  if (-not (Test-Path $QnnSdkZip)) { throw "QnnSdkZip not found: $QnnSdkZip" }
  $zipWsl = (wsl.exe -d $Distro wslpath -a "$QnnSdkZip").Trim()
  $envPrefix = "QNN_SDK_ZIP='$zipWsl' "
}

Write-Host "`nRunning setup.sh inside WSL ($Distro)... (long: ExecuTorch QNN build)" -ForegroundColor Cyan
# -l => login shell so conda/profile init is available.
wsl.exe -d $Distro bash -lc "$envPrefix bash '$repoWsl/setup/setup.sh'"
