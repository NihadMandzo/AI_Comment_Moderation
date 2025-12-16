Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

# Ensure reload subprocess can import the 'app' package
$env:PYTHONPATH = $projectRoot

# If port 8000 is blocked on Windows, use 8010
C:/AI_Speech_Recognition/.venv/Scripts/python.exe -m uvicorn --app-dir $projectRoot --reload-dir $projectRoot app.main:app --reload --host 127.0.0.1 --port 8010
