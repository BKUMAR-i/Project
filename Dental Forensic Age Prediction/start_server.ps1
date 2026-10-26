Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$projectRoot = $PSScriptRoot
Set-Location $projectRoot

if (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'manage.py'))) {
    throw "manage.py was not found in $projectRoot. Run this script from the project root."
}

if ($args.Count -eq 0) {
    python manage.py runserver 127.0.0.1:8001
} else {
    python manage.py runserver @args
}