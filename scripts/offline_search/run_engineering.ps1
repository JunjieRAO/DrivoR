param(
    [string]$Python = "python",
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$RunnerArgs
)
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$oldPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = "$repoRoot;$repoRoot\nuplan-devkit;$oldPythonPath"
    & $Python -m navsim.offline_search.run @RunnerArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
    $env:PYTHONPATH = $oldPythonPath
}
