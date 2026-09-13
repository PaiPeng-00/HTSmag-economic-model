"""Single-source monetary normalization to constant 2025 U.S. dollars."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


PRICE_BASIS_YEAR = 2025
HTS_PRICE_CONVERSION_METHOD = "direct_scenario_assumption_2025usd"
CPI_U_ANNUAL_AVERAGE = {2020: 258.811, 2025: 321.943}
CPI_2025_OVER_2020 = (
    CPI_U_ANNUAL_AVERAGE[2025] / CPI_U_ANNUAL_AVERAGE[2020]
)
CONVERSION_TABLE_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "2025_USD_conversion_table.csv"
)


class PriceMappingAuditError(RuntimeError):
    """Raised before calculation when a source-year mapping is inconsistent."""


_SOURCE_NUMERIC_VALUE: dict[str, float] = {
    "C0 S1": 4.2,
    "C0 S2": 2.8,
    "C0 S3": 1.4,
    "c_core,VOM S1": 5.0,
    "c_core,VOM S2": 3.0,
    "c_core,VOM S3": 1.0,
    "p_PCS": 750.0,
    "c_PCS,VOM": 1.74,
    "f_PCS,FOM": 2.5,
    "phi_repl": 25.0,
    "p_HTS S1/S4": 100.0,
    "p_HTS S2/S5": 50.0,
    "p_HTS S3/S6": 10.0,
    "p_H2 S1": 9.0,
    "p_H2 S2": 6.0,
    "p_H2 S3": 3.0,
    "p_He S1": 331.0,
    "p_He S2": 529.0,
    "p_He S3": 662.0,
    "p_PS": 20.0,
}

_EXPECTED_SOURCE_YEAR: dict[str, int | None] = {
    "C0 S1": 2010,
    "C0 S2": 2010,
    "C0 S3": 2010,
    "c_core,VOM S1": 2019,
    "c_core,VOM S2": 2019,
    "c_core,VOM S3": 2019,
    "p_PCS": 2018,
    "c_PCS,VOM": 2019,
    "f_PCS,FOM": None,
    "phi_repl": None,
    "p_HTS S1/S4": 2025,
    "p_HTS S2/S5": 2025,
    "p_HTS S3/S6": 2025,
    "p_H2 S1": 2024,
    "p_H2 S2": 2024,
    "p_H2 S3": 2024,
    "p_He S1": 2014,
    "p_He S2": 2023,
    "p_He S3": 2025,
    "p_PS": 2025,
}

_PRICE_YEAR_IS_PROXY = {
    "p_He S1",
    "p_He S2",
    "p_PS",
}

_AFFECTED_CODE_FIELD: dict[str, str] = {
    "C0 S1": "BACKGROUND_CAPITAL_USD_BY_SCENARIO[S1]",
    "C0 S2": "BACKGROUND_CAPITAL_USD_BY_SCENARIO[S2,S4,S5,S6]",
    "C0 S3": "BACKGROUND_CAPITAL_USD_BY_SCENARIO[S3]",
    "c_core,VOM S1": "CORE_VOM_USD_PER_MWH_TH_BY_SCENARIO[S1]",
    "c_core,VOM S2": "CORE_VOM_USD_PER_MWH_TH_BY_SCENARIO[S2,S4,S5,S6]",
    "c_core,VOM S3": "CORE_VOM_USD_PER_MWH_TH_BY_SCENARIO[S3]",
    "p_PCS": "PCS_CAPITAL_COST_USD_PER_KWE",
    "c_PCS,VOM": "PCS_VOM_USD_PER_MWH_E",
    "f_PCS,FOM": "PCS_FOM_FRACTION_PER_YEAR",
    "phi_repl": "ANNUAL_COOLANT_REPLENISH_FRACTION",
    "p_HTS S1/S4": "HTS_PRICE_2025_USD_PER_KAM_BY_SCENARIO[S1,S4]",
    "p_HTS S2/S5": "HTS_PRICE_2025_USD_PER_KAM_BY_SCENARIO[S2,S5]",
    "p_HTS S3/S6": "HTS_PRICE_2025_USD_PER_KAM_BY_SCENARIO[S3,S6]",
    "p_H2 S1": "COOLANT_PRICE_2025_USD_PER_KG_BY_SCENARIO[S1][H2]",
    "p_H2 S2": "COOLANT_PRICE_2025_USD_PER_KG_BY_SCENARIO[S2,S4,S5,S6][H2]",
    "p_H2 S3": "COOLANT_PRICE_2025_USD_PER_KG_BY_SCENARIO[S3][H2]",
    "p_He S1": "COOLANT_PRICE_2025_USD_PER_KG_BY_SCENARIO[S1][He]",
    "p_He S2": "COOLANT_PRICE_2025_USD_PER_KG_BY_SCENARIO[S2,S4,S5,S6][He]",
    "p_He S3": "COOLANT_PRICE_2025_USD_PER_KG_BY_SCENARIO[S3][He]",
    "p_PS": "POWER_SUPPLY_PRICE_2025_USD_PER_A",
}

_DEPENDENT_LOCATIONS: dict[str, str] = {
    "C0": "Methods Eq. 18; Table S6; Note S10; Data S1; Fig. 5; Fig. 6; Results 4; Discussion",
    "OPEX": "Methods Eqs. 14-18; Table S6; Note S10; Data S1; Fig. 5; Fig. 6; Results 4; Discussion",
    "HTS": "Table S6; Note S10; Data S1; Fig. 5; Fig. 6 row labels; Results 4; Discussion",
    "coolant": "Table S6; Note S10; Data S1; Fig. 5; Fig. 6; Results 4",
    "power_supply": "Table S6; Note S10; Data S1; Fig. 5; Fig. 6",
}


def _dependency(parameter: str) -> str:
    if parameter.startswith("C0"):
        return _DEPENDENT_LOCATIONS["C0"]
    if parameter.startswith(("c_core", "p_PCS", "c_PCS", "f_PCS", "phi_repl")):
        return _DEPENDENT_LOCATIONS["OPEX"]
    if parameter.startswith("p_HTS"):
        return _DEPENDENT_LOCATIONS["HTS"]
    if parameter.startswith(("p_H2", "p_He")):
        return _DEPENDENT_LOCATIONS["coolant"]
    return _DEPENDENT_LOCATIONS["power_supply"]


def _load_conversion_rows() -> dict[str, dict[str, str]]:
    if not CONVERSION_TABLE_PATH.is_file():
        raise PriceMappingAuditError(
            f"2025-USD conversion table is missing: {CONVERSION_TABLE_PATH}"
        )
    with CONVERSION_TABLE_PATH.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required_columns = {
            "parameter", "old_value", "source_price_year",
            "CPI_2025_over_CPI_y", "new_value", "new_unit",
            "basis_note", "conversion_method",
        }
        missing_columns = required_columns - set(reader.fieldnames or ())
        if missing_columns:
            raise PriceMappingAuditError(
                f"2025-USD conversion table columns are missing: {sorted(missing_columns)}"
            )
        rows = {row["parameter"]: row for row in reader}
    expected = set(_SOURCE_NUMERIC_VALUE)
    if set(rows) != expected:
        raise PriceMappingAuditError(
            "2025-USD parameter set mismatch: "
            f"missing={sorted(expected - set(rows))}, extra={sorted(set(rows) - expected)}"
        )
    return rows


_CONVERSION_ROWS = _load_conversion_rows()


def _source_year(row: dict[str, str]) -> int | None:
    raw = row["source_price_year"].strip()
    return None if not raw else int(float(raw))


def validate_price_source_mapping() -> None:
    """Fail before calculation if provisional HTS/helium mappings drift."""
    for parameter, expected_year in _EXPECTED_SOURCE_YEAR.items():
        observed = _source_year(_CONVERSION_ROWS[parameter])
        if observed != expected_year:
            raise PriceMappingAuditError(
                f"source-year audit failed for {parameter}: "
                f"expected {expected_year}, observed {observed}"
            )
    for parameter, source_value in _SOURCE_NUMERIC_VALUE.items():
        row = _CONVERSION_ROWS[parameter]
        factor = float(row["CPI_2025_over_CPI_y"])
        converted = float(row["new_value"])
        expected = source_value * factor
        if not np.isclose(converted, expected, rtol=1e-12, atol=1e-12):
            raise PriceMappingAuditError(
                f"single-conversion audit failed for {parameter}: "
                f"source={source_value}, factor={factor}, "
                f"table={converted}, expected={expected}"
            )
    # Method working-document gate: HTS prices are direct constant-2025-US$
    # scenario assumptions. Literature supports only the range/basis and is not
    # treated as the direct source of the exact 100/50/10 values.
    for parameter in ("p_HTS S1/S4", "p_HTS S2/S5", "p_HTS S3/S6"):
        row = _CONVERSION_ROWS[parameter]
        if row["conversion_method"].strip() != HTS_PRICE_CONVERSION_METHOD:
            raise PriceMappingAuditError(
                f"HTS conversion method failed for {parameter}: "
                f"{row['conversion_method']!r}"
            )
        if parameter in _PRICE_YEAR_IS_PROXY:
            raise PriceMappingAuditError(
                f"Direct HTS scenario assumption cannot use proxy-year status: {parameter}"
            )
        if float(row["CPI_2025_over_CPI_y"]) != 1.0:
            raise PriceMappingAuditError(
                f"Direct HTS scenario assumption must use CPI factor 1.0: {parameter}"
            )
    for parameter in ("p_He S1", "p_He S2", "p_He S3"):
        if parameter not in _CONVERSION_ROWS:
            raise PriceMappingAuditError(
                f"method-document source mapping is absent for {parameter}"
            )


def converted_value(parameter: str) -> float:
    return float(_CONVERSION_ROWS[parameter]["new_value"])


def conversion_table_sha256() -> str:
    import hashlib

    digest = hashlib.sha256()
    with CONVERSION_TABLE_PATH.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def monetary_conversion_audit_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for parameter, row in _CONVERSION_ROWS.items():
        source_year = _source_year(row)
        source_value = _SOURCE_NUMERIC_VALUE[parameter]
        factor = float(row["CPI_2025_over_CPI_y"])
        converted = float(row["new_value"])
        rows.append({
            "parameter": parameter,
            "source_value": source_value,
            "source_value_display": row["old_value"],
            "source_price_year": source_year,
            "CPI_2025_over_CPI_y": factor,
            "value_2025_USD": converted,
            "value_2025_unit": row["new_unit"],
            "price_year_is_proxy": parameter in _PRICE_YEAR_IS_PROXY,
            "conversion_method": row["conversion_method"],
            "basis_note": row["basis_note"],
            "affected_code_field": _AFFECTED_CODE_FIELD[parameter],
            "dependent_manuscript_figure_locations": _dependency(parameter),
            "conversion_application_count": 1,
            "mapping_verification": "PASS",
        })
    rows.extend([
        {
            "parameter": "context Wade and Leuer Table IV capitalized cost",
            "source_value": 4220.6,
            "source_value_display": "4,220.6 million US$",
            "source_price_year": 2010,
            "CPI_2025_over_CPI_y": 1.476423487544484,
            "value_2025_USD": 4220.6 * 1.476423487544484,
            "value_2025_unit": "million 2025 US$",
            "price_year_is_proxy": False,
            "basis_note": "context and S1 anchor",
            "affected_code_field": "context only",
            "dependent_manuscript_figure_locations": "Introduction; Table/Note S10 context",
            "conversion_application_count": 1,
            "mapping_verification": "PASS",
        },
        {
            "parameter": "context Lindley ARC-like FOAK specific capital",
            "source_value": "29670-34500",
            "source_value_numeric_min": 29670.0,
            "source_value_numeric_max": 34500.0,
            "source_value_display": "29,670-34,500 US$ kW^-1",
            "source_price_year": 2020,
            "CPI_2025_over_CPI_y": CPI_2025_OVER_2020,
            "value_2025_USD": "36907-42916",
            "value_2025_USD_numeric_min": 29670.0 * CPI_2025_OVER_2020,
            "value_2025_USD_numeric_max": 34500.0 * CPI_2025_OVER_2020,
            "value_2025_unit": "2025 US$ kW^-1",
            "price_year_is_proxy": False,
            "basis_note": "context only",
            "affected_code_field": "context only",
            "dependent_manuscript_figure_locations": "Introduction and SI context",
            "conversion_application_count": 1,
            "mapping_verification": "PASS",
        },
        {
            "parameter": "context Lindley mature small-unit capital",
            "source_value": 1.5,
            "source_value_display": "~1.5 billion US$",
            "source_price_year": 2020,
            "CPI_2025_over_CPI_y": CPI_2025_OVER_2020,
            "value_2025_USD": 1.5 * CPI_2025_OVER_2020,
            "value_2025_unit": "billion 2025 US$",
            "price_year_is_proxy": False,
            "basis_note": "context only",
            "affected_code_field": "context only",
            "dependent_manuscript_figure_locations": "Introduction and SI context",
            "conversion_application_count": 1,
            "mapping_verification": "PASS",
        },
        {
            "parameter": "context ARC fabricated components",
            "source_value": "5.5-5.6",
            "source_value_numeric_min": 5.5,
            "source_value_numeric_max": 5.6,
            "source_value_display": "5.5-5.6 billion US$",
            "source_price_year": 2014,
            "CPI_2025_over_CPI_y": 1.3599241349013247,
            "value_2025_USD": "7.480-7.616",
            "value_2025_USD_numeric_min": 5.5 * 1.3599241349013247,
            "value_2025_USD_numeric_max": 5.6 * 1.3599241349013247,
            "value_2025_unit": "billion 2025 US$",
            "price_year_is_proxy": False,
            "basis_note": "context only",
            "affected_code_field": "context only",
            "dependent_manuscript_figure_locations": "Introduction and SI context",
            "conversion_application_count": 1,
            "mapping_verification": "PASS",
        },
        {
            "parameter": "context ARC magnet/structure subtotal",
            "source_value": "5.1-5.2",
            "source_value_numeric_min": 5.1,
            "source_value_numeric_max": 5.2,
            "source_value_display": "5.1-5.2 billion US$",
            "source_price_year": 2014,
            "CPI_2025_over_CPI_y": 1.3599241349013247,
            "value_2025_USD": "6.936-7.072",
            "value_2025_USD_numeric_min": 5.1 * 1.3599241349013247,
            "value_2025_USD_numeric_max": 5.2 * 1.3599241349013247,
            "value_2025_unit": "billion 2025 US$",
            "price_year_is_proxy": False,
            "basis_note": "context only",
            "affected_code_field": "context only",
            "dependent_manuscript_figure_locations": "Introduction and SI context",
            "conversion_application_count": 1,
            "mapping_verification": "PASS",
        },
    ])
    for audit_row in rows:
        audit_row.setdefault("conversion_method", "annual_average_cpi_u_to_2025usd")
    return rows


validate_price_source_mapping()

BACKGROUND_CAPITAL_USD_BY_SCENARIO = {
    "S1": converted_value("C0 S1") * 1e9,
    "S2": converted_value("C0 S2") * 1e9,
    "S3": converted_value("C0 S3") * 1e9,
    "S4": converted_value("C0 S2") * 1e9,
    "S5": converted_value("C0 S2") * 1e9,
    "S6": converted_value("C0 S2") * 1e9,
}
CORE_VOM_USD_PER_MWH_TH_BY_SCENARIO = {
    "S1": converted_value("c_core,VOM S1"),
    "S2": converted_value("c_core,VOM S2"),
    "S3": converted_value("c_core,VOM S3"),
    "S4": converted_value("c_core,VOM S2"),
    "S5": converted_value("c_core,VOM S2"),
    "S6": converted_value("c_core,VOM S2"),
}
HTS_PRICE_2025_USD_PER_KAM_BY_SCENARIO = {
    "S1": converted_value("p_HTS S1/S4"),
    "S2": converted_value("p_HTS S2/S5"),
    "S3": converted_value("p_HTS S3/S6"),
    "S4": converted_value("p_HTS S1/S4"),
    "S5": converted_value("p_HTS S2/S5"),
    "S6": converted_value("p_HTS S3/S6"),
}
COOLANT_PRICE_2025_USD_PER_KG_BY_SCENARIO = {
    "S1": {"He": converted_value("p_He S1"), "H2": converted_value("p_H2 S1")},
    "S2": {"He": converted_value("p_He S2"), "H2": converted_value("p_H2 S2")},
    "S3": {"He": converted_value("p_He S3"), "H2": converted_value("p_H2 S3")},
    "S4": {"He": converted_value("p_He S2"), "H2": converted_value("p_H2 S2")},
    "S5": {"He": converted_value("p_He S2"), "H2": converted_value("p_H2 S2")},
    "S6": {"He": converted_value("p_He S2"), "H2": converted_value("p_H2 S2")},
}
PCS_CAPITAL_COST_USD_PER_KWE = converted_value("p_PCS")
PCS_FOM_FRACTION_PER_YEAR = converted_value("f_PCS,FOM") / 100.0
PCS_VOM_USD_PER_MWH_E = converted_value("c_PCS,VOM")
ANNUAL_COOLANT_REPLENISH_FRACTION = converted_value("phi_repl") / 100.0
POWER_SUPPLY_PRICE_2025_USD_PER_A = converted_value("p_PS")