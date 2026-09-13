$ErrorActionPreference = "Stop"

$codeDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$tableDir = Join-Path $codeDir "outputs\arc_16pancake_nuc600\tables"
$baselineCsv = Join-Path $tableDir "scan_full_grid_tidy_plant_opex_2025usd.csv"
$directCsv = Join-Path $tableDir "scan_full_grid_tidy_plant_opex_2025usd_direct_hts.csv"
$rawCsv = Join-Path $tableDir "scan_full_grid_tidy_plant_opex_2025usd_direct_hts_raw_fresh_rerun.csv"
$directXlsx = Join-Path $tableDir "scan_full_grid_plant_opex_2025usd_direct_hts.xlsx"
$rawXlsx = Join-Path $tableDir "scan_full_grid_plant_opex_2025usd_direct_hts_raw_fresh_rerun.xlsx"
$report = Join-Path $tableDir "economic_2025usd_direct_hts_nonmonetary_harmonization.json"

foreach ($path in @($baselineCsv, $directCsv, $directXlsx)) {
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
    & python "scripts/8_economic/harmonize_2025usd_direct_hts_nonmonetary.py" `
        --baseline $baselineCsv `
        --new $directCsv `
        --raw-backup $rawCsv `
        --report $report `
        --max-ulp 4
    if ($LASTEXITCODE -ne 0) {
        throw "Nonmonetary harmonization failed with exit code $LASTEXITCODE"
    }
    Move-Item -LiteralPath $directXlsx -Destination $rawXlsx
    & python "scripts/8_economic/export_scan_xlsx.py" $directCsv $directXlsx
    if ($LASTEXITCODE -ne 0) {
        throw "Harmonized XLSX export failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}