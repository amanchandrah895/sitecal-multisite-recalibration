# sitecal-multisite-recalibration

Code for **"Well Calibrated on Average, Miscalibrated per Site: Hierarchical Recalibration of Machine Learning Models Under Multi-Site Distribution Shift"**.

A model trained at one site and deployed at many can be well calibrated when all sites are pooled and still miscalibrated at many of them. This repository provides a site-level calibration audit and hierarchical (empirical-Bayes) post-hoc recalibration of a frozen model. It also contains the full analysis of the case study: a LightGBM in-hospital mortality model trained on MIMIC-IV and transported to 207 hospitals in eICU-CRD.

## Repository structure

```
sitecal-multisite-recalibration/
├── sitecal/          Python package: calibration audit, hierarchical recalibration, risk analysis
│   ├── sitecal/
│   └── tests/
├── notebooks/        Case-study analysis, NB00–NB13 (Google Colab, run in order)
├── code/             SQL extraction queries and the shared analysis library used by the notebooks
├── simulation/       Simulation study and closed-form risk analysis
├── analysis/         Post-processing of notebook outputs (hospital-bootstrap CIs for the label budget)
├── requirements.txt
└── LICENSE
```

## Installation

```bash
pip install "git+https://github.com/amanchandrah895/sitecal-multisite-recalibration#subdirectory=sitecal"
```

## Usage

```python
from sitecal import site_audit, HierarchicalRecalibrator, break_even_labels

# Site-level audit: y outcomes, p predicted probabilities, site identifiers
a = site_audit(y, p, site)
a["tau"]                  # between-site SD of ln(O/E)
a["prediction_interval"]  # 95% prediction interval of O/E for a new site
a["pct_in_band"], a["chance_ceiling"]  # % of sites with O/E in 0.8-1.2, and the value under perfect calibration

# Hierarchical recalibration fitted on the network's labeled update data
rc = HierarchicalRecalibrator().fit(p_update, y_update, site_update)
p_adapted = rc.predict(p_new, site_new)

# A new site with a few labels borrows strength from the network
offset = rc.site_offset("new_site", local=(p_local, y_local))

# Labels at which site-only fitting stops losing to the network prior
break_even_labels(tau2=rc.tau2_, info=0.07)
```

Options of `HierarchicalRecalibrator`:

- `global_step`: start with a global logistic recalibration.
- `use_covariates`: predict the prior mean from site descriptors.
- `semi_supervised`: add a label-free estimate from unlabeled scores.

Tests: `pip install -e sitecal && pytest sitecal/tests`

## Data

The case study uses two databases that require credentialed access through PhysioNet:

- **MIMIC-IV v3.1**: https://doi.org/10.13026/kpb9-mt58
- **eICU Collaborative Research Database v2.0**: https://doi.org/10.13026/C2WM1R

**No data are included in this repository, and the data cannot be redistributed.** This covers raw extracts, harmonized analysis files, predictions and derived results. Credentialed users can regenerate every result with the notebooks.

## Reproducing the case study

1. Obtain credentialed PhysioNet access to both databases and enable access through Google BigQuery.
2. Copy the repository to Google Drive as `MyDrive/Plan A/v4/`, or set the environment variable `TRANSPORT_BASE` to another location.
3. Run the notebooks in Google Colab in order. Each notebook writes its outputs to `results/NBxx/`, and long steps are cached so a run resumes after a disconnect.

| Notebook | Content |
|---|---|
| NB00 | Setup and SQL dry run |
| NB01–NB02 | Extraction from MIMIC-IV and eICU-CRD into one harmonized long format |
| NB03 | Harmonization, cohort definition and feature audit |
| NB04 | Model development (LightGBM, logistic regression, XGBoost) |
| NB05 | External validation and comparison with APACHE IVa |
| NB06 | Hospital-level heterogeneity, funnel plots, meta-regression |
| NB07 | Pipeline-artifact ablation (four feature constructions) |
| NB08 | Updating strategies, 2014 → 2015 |
| NB09 | Label budget |
| NB10 | Reverse transport, temporal drift, subgroups |
| NB11 | Sensitivity analyses |
| NB12 | Chance ceilings, bootstrap intervals, multiplicity adjustment, one-stage model, leave-region-out, second outcome |
| NB13 | Method extensions, label-free comparators, risk-analysis inputs, MLP base model |

For a quick smoke test with small bootstrap and search sizes, set `os.environ['TRANSPORT_FAST'] = '1'` in the setup cell.

## Simulation study

The simulation needs no credentialed data:

```bash
pip install -r requirements.txt
cd simulation
SIM_R=50 python3 simulation.py sim_results covariate
SIM_R=50 python3 simulation.py sim_results label
python3 simulation.py sim_results aggregate
python3 simulation.py sim_results tau
```

## License

MIT (see `LICENSE`). The databases are governed by their own PhysioNet licenses and data use agreements.
