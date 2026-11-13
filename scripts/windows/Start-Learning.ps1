param([int]$Port = 8765)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)  # scripts\windows -> project root
Set-Location -LiteralPath $Root
$candidates = @(
    (Join-Path $Root '.venv\Scripts\python.exe'),
    (Join-Path (Split-Path -Parent $Root) '.venv\Scripts\python.exe'),
    'C:\Python314\python.exe'
)
foreach ($candidate in $candidates) {
    if (Test-Path -LiteralPath $candidate) {
        Write-Host "Open http://127.0.0.1:$Port in your browser. Ctrl+C stops the server."
        & $candidate (Join-Path $Root 'learning_app.py') --port $Port
        exit $LASTEXITCODE
    }
}
python (Join-Path $Root 'learning_app.py') --port $Port
