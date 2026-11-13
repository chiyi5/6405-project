param([int]$Port = 8765)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$candidates = @(
    (Join-Path $PSScriptRoot '.venv\Scripts\python.exe'),
    (Join-Path (Split-Path -Parent $PSScriptRoot) '.venv\Scripts\python.exe'),
    'C:\Python314\python.exe'
)
foreach ($candidate in $candidates) {
    if (Test-Path -LiteralPath $candidate) {
        Write-Host "Open http://127.0.0.1:$Port in your browser. Ctrl+C stops the server."
        & $candidate (Join-Path $PSScriptRoot 'learning_app.py') --port $Port
        exit $LASTEXITCODE
    }
}
python (Join-Path $PSScriptRoot 'learning_app.py') --port $Port
