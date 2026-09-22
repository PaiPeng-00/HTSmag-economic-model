"""Generate current Fig4/S7 tables from a newly solved ledger directory."""
import argparse
from pathlib import Path
from pipeline import fig4_lcoe_v10_statistics as stats

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True, help="Directory containing the four realization parquet files")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise SystemExit("Output exists; use a new output directory")
stats.SOURCE = args.source.resolve()
stats.OUT = args.output.resolve()
stats.main()
