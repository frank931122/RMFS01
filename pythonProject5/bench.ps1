param(
  [string]$Prefix = "demo01",
  [string]$Gammas = "0,5,10",
  [int]$Iters = 2000,
  [string]$Seeds = "0",
  [string]$Baseline = "bench_baseline_demo01.json",
  [double]$Tol = 0.001
)

# bench.ps1 放在 pythonProject5 目录；$PSScriptRoot 就是它所在目录
$ProjectDir = $PSScriptRoot
$RepoRoot = (Resolve-Path (Join-Path $ProjectDir "..")).Path

# 让 "import pythonProject5..." 在命令行也能找到（等价于你手动 set PYTHONPATH）
$env:PYTHONPATH = $RepoRoot

# 你机器上可用的 Python（避免 Windows 的 Microsoft Store python 别名问题）
$PyExe = "C:\Users\16415\AppData\Local\Programs\Python\Python314\python.exe"

# 进入项目目录，保证相对路径一致
Set-Location $ProjectDir

& $PyExe .\main.py `
  --quick-bench --prefix $Prefix `
  --bench-iters $Iters `
  --bench-seeds $Seeds `
  --gammas $Gammas `
  --bench-compare $Baseline `
  --bench-tol $Tol

exit $LASTEXITCODE