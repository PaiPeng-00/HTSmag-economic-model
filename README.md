# HTS TF magnet performance and fusion plant economics

Version 2.0.0 contains the model code and portable postprocessing for the
manuscript "Fusion economics are robust to performance degradation in
high-temperature superconducting magnets". Large frozen inputs and reference
outputs are provided through a companion Zenodo dataset. Hydrate that dataset
into a clean clone before replaying the postprocessing.

## Reproduce the manuscript postprocessing

```sh
python reproduce/hydrate_zenodo_data.py /path/to/zenodo_data
python -B run_analysis.py --group fig5 --output reproduction_runs/fig5
python -B run_analysis.py --group retention --output reproduction_runs/retention
```

The current Note S7 sensitivity fixes 200 tapes per turn and He cooling, then
minimizes each interpolated threshold over S1--S3 and all 61 sampled
turn-to-turn contact resistivities. The reported 4.2/20 K results are: no
qualifying interval at 1%; 1.2 nOhm/no qualifying interval at 5%; 3.6/37.1
nOhm at 10% (ratio 10.3); and 7.8/98.1 nOhm at 20% (ratio 12.5).

## Status

Software version: `2.0.0`. Repository:
<https://github.com/PaiPeng-00/HTSmag-economic-model>.

Companion dataset: <https://doi.org/10.5281/zenodo.22733646>.

Scientific source identity: `scientific_results_v7_splice_equivalent_20260906`. The MIT licence,
author order, affiliations and the first author's ORCID are confirmed. The
paper DOI has not yet been assigned. The code package has passed the clean-copy
tests and release scans recorded in the external validation report.
