param(
  [string]$Prefix = "demo01",
  [string]$Gammas = "0,5,10",
  [int]$Iters = 2000,
  [string]$Seeds = "0",
  [string]$Baseline = "bench_baseline_demo01.json",
  [double]$Tol = 0.001,
  [string]$PythonExe = ""
)

# bench2.ps1: cross-environment launcher
# - Removes hard-coded Windows Python path
# - Auto-detects an available Python executable

$ProjectDir = $PSScriptRoot
$RepoRoot = (Resolve-Path (Join-Path $ProjectDir "..")).Path
$env:PYTHONPATH = $RepoRoot

Set-Location $ProjectDir

function Resolve-PythonExe {
  param([string]$Preferred)

  if ($Preferred -and $Preferred.Trim().Length -gt 0) {
    return $Preferred
  }

  if ($env:PYTHON_EXE -and $env:PYTHON_EXE.Trim().Length -gt 0) {
    return $env:PYTHON_EXE
  }

  $candidates = @("python", "python3", "py")
  foreach ($name in $candidates) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if ($null -ne $cmd) {
      return $name
    }
  }

  throw "No Python executable found. Provide -PythonExe or set PYTHON_EXE."
}

$PyExe = Resolve-PythonExe -Preferred $PythonExe
Write-Host "[bench2] Using Python executable: $PyExe"

& $PyExe .\main.py `
  --quick-bench --prefix $Prefix `
  --bench-iters $Iters `
  --bench-seeds $Seeds `
  --gammas $Gammas `
  --bench-compare $Baseline `
  --bench-tol $Tol

exit $LASTEXITCODE
