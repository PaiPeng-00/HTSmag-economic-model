"""Small numerical smoke checks, not a full parameter-sweep qualification."""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ["FUSION_DEVICE"] = "arc_16pancake_nuc600_v6_2"
sys.path.insert(0, str(ROOT / "model/model_code/code/src"))
sys.path.insert(0, str(ROOT / "pipeline"))
from fusion_tem import device as cfg
from fusion_tem.economic.annual_time import allocate_continuous_pulse_dwell
from build_v10_candidate_figure_inputs import crossings

assert np.isclose(cfg.V_mag, 1.5 * cfg.V_WP, rtol=1e-10)
assert np.isclose(cfg.V_mag * cfg.NUCLEAR_POWER_DENSITY, 3111.5253, rtol=1e-6)
schedule = allocate_continuous_pulse_dwell(8760, 2, .1)
assert np.isclose(schedule.pulse_hours + schedule.dwell_hours, 8760)
r = np.array([1., 2., 4.])
d = np.array([[0., .05, .2], [.2, .3, .4], [0., .01, .02], [.01, .2, .01]])
yes = np.ones_like(d, dtype=bool)
result = crossings(r, d, yes, yes, yes, .1)
assert np.isclose(result.Rj_tol_nOhm.iloc[0], 8/3)
assert np.isnan(result.Rj_tol_nOhm.iloc[1])
assert result.upper_censored.iloc[2]
assert result.nonmonotonic.iloc[3]
print(json.dumps({"status": "PASS", "checks": ["V10 geometry and nuclear heat", "annual time closure",
                 "joint crossing interpolation", "invalid prefix", "right censoring", "nonmonotonic prefix"],
                 "full_grid": "NOT RUN"}, indent=2))
