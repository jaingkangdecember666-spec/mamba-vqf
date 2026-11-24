param(
    [string]$Python,
    [string]$Output = (Join-Path $PSScriptRoot 'trained')
)
$ErrorActionPreference = 'Stop'
if (-not $Python) {
    $environment = Join-Path $PSScriptRoot '.venv'
    $Python = Join-Path $environment 'Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $Python)) {
        if (Get-Command py -ErrorAction SilentlyContinue) {
            & py -3.12 -m venv $environment
        } else {
            & python -m venv $environment
        }
        if ($LASTEXITCODE -ne 0) { throw 'Could not create a Python environment.' }
    }
    & $Python -m pip install --disable-pip-version-check -r (Join-Path $PSScriptRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Could not install training dependencies.' }
}
& $Python -B (Join-Path $PSScriptRoot 'train.py') --output $Output
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
