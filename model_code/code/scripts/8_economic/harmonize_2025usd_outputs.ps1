$ErrorActionPreference = "Stop"

$codeDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$tableDir = Join-Path $codeDir "outputs\arc_16pancake_nuc600\tables"
$v2Csv = Join-Path $tableDir "scan_full_grid_tidy_plant_opex.csv"
$v3Csv = Join-Path $tableDir "scan_full_grid_tidy_plant_opex_2025usd.csv"
$rawCsv = Join-Path $tableDir "scan_full_grid_tidy_plant_opex_2025usd_raw_fresh_rerun.csv"
$v3Xlsx = Join-Path $tableDir "scan_full_grid_plant_opex_2025usd.xlsx"
$rawXlsx = Join-Path $tableDir "scan_full_grid_plant_opex_2025usd_raw_fresh_rerun.xlsx"
$report = Join-Path $tableDir "economic_2025usd_nonmonetary_harmonization.json"

foreach ($path in @($v2Csv, $v3Csv, $v3Xlsx)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required input is missing: $path"
    }
}
foreach ($path in @($rawCsv, $rawXlsx, $report)) {
    if (Test-Path -LiteralPath $path) {
        throw "Refusing to overwrite existing audit artifact: $path"
    }
}

$env:FUSION_DEVICE = "arc_16pancake_nuc600"
$env:PYTHONPATH = "src"
Push-Location $codeDir
try {
    & python "scripts/8_economic/harmonize_2025usd_nonmonetary.py" `
        --plant-v2 $v2Csv `
        --plant-v3 $v3Csv `
        --raw-backup $rawCsv `
        --report $report `
        --max-ulp 4
    if ($LASTEXITCODE -ne 0) {
        throw "Nonmonetary harmonization failed with exit code $LASTEXITCODE"
    }
    Move-Item -LiteralPath $v3Xlsx -Destination $rawXlsx
    & python "scripts/8_economic/export_scan_xlsx.py" $v3Csv $v3Xlsx
    if ($LASTEXITCODE -ne 0) {
        throw "Harmonized XLSX export failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}