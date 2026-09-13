$ErrorActionPreference = "Stop"

$codeDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$logDir = Join-Path $codeDir "outputs\arc_16pancake_nuc600\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$stdoutPath = Join-Path $logDir "scan_full_grid_2025usd_direct_hts.stdout.log"
$stderrPath = Join-Path $logDir "scan_full_grid_2025usd_direct_hts.stderr.log"
$exitCodePath = Join-Path $logDir "scan_full_grid_2025usd_direct_hts.exit_code.txt"
$pidPath = Join-Path $logDir "scan_full_grid_2025usd_direct_hts.pid.txt"

Set-Content -LiteralPath $stdoutPath -Value ""
Set-Content -LiteralPath $stderrPath -Value ""
Set-Content -LiteralPath $pidPath -Value $PID
Remove-Item -LiteralPath $exitCodePath -ErrorAction SilentlyContinue

$env:FUSION_DEVICE = "arc_16pancake_nuc600"
$env:PYTHONPATH = "src"
$env:PYTHONUNBUFFERED = "1"
$env:WRITE_SCAN_XLSX = "1"
$env:SCAN_OUTPUT_CSV_NAME = "scan_full_grid_tidy_plant_opex_2025usd_direct_hts.csv"
$env:SCAN_OUTPUT_XLSX_NAME = "scan_full_grid_plant_opex_2025usd_direct_hts.xlsx"
$env:SCAN_MANIFEST_NAME = "manifest_plant_opex_2025usd_direct_hts.json"

Push-Location $codeDir
try {
    & python -u "scripts/8_economic/scan_full_grid.py" 1>> $stdoutPath 2>> $stderrPath
    $scanExitCode = $LASTEXITCODE
}
catch {
    $_ | Out-String | Add-Content -LiteralPath $stderrPath
    $scanExitCode = 1
}
finally {
    Pop-Location
}

Set-Content -LiteralPath $exitCodePath -Value $scanExitCode
exit $scanExitCode