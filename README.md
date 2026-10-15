# sitecal-multisite-recalibration

Code for the paper **"Well Calibrated on Average, Miscalibrated per Site: Hierarchical Recalibration of Machine Learning Models Under Multi-Site Distribution Shift"** (submitted to IEEE Access).

A model trained at one site and deployed at many can be well calibrated when all sites are pooled and still miscalibrated at many of them. This repository contains:

- **`sitecal/`**: a small Python package for a site-level calibration audit (random-effects summary, prediction interval, funnel limits, Monte Carlo chance ceiling) and hierarchical (empirical-Bayes) post-hoc recalibration of a frozen model, with covariate-informed and semi-supervised variants and the closed-form risk analysis (Proposition 1).
- **`notebooks/`**: the complete analysis of the case study, a LightGBM in-hospital mortality model trained on MIMIC-IV and transported to 207 hospitals in eICU-CRD (NB00–NB13, run in Google Colab).
- **`code/`**: the SQL extraction queries, the shared analysis library used by the notebooks, and the simulation study.
- **`paper/`**: the LaTeX source of the manuscript and supplement, with the scripts that generate every table, number and new figure from the aggregated results.

## Quick start: the `sitecal` package

```bash
pip install "git+https://github.com/amanchandrah895/sitecal-multisite-recalibration#subdirectory=sitecal"
```

```python
from sitecal import site_audit, HierarchicalRecalibrator, break_even_labels

# Audit: y outcomes, p predicted probabilities, site site identifiers
a = site_audit(y, p, site)
a["tau"], a["prediction_interval"], a["pct_in_band"], a["chance_ceiling"]

# Hierarchical recalibration fitted on the network's labeled update data
rc = HierarchicalRecalibrator().fit(p_update, y_update, site_update)
p_adapted = rc.predict(p_new, site_new)

# A new site with a few labels borrows strength from the network
offset = rc.site_offset("new_site", local=(p_local, y_local))

# How many local labels before site-only fitting stops losing to the prior?
break_even_labels(tau2=rc.tau2_, info=0.07)
```

Run the tests with `pip install -e sitecal && pytest sitecal/tests`.

## Reproducing the case study

The case study uses two credentialed PhysioNet databases. **No patient-level or derived data are included in this repository**, and none may be redistributed under the PhysioNet data use agreements.

- **MIMIC-IV v3.1**: Johnson et al., *Sci Data* 2023; dataset DOI [10.13026/kpb9-mt58](https://doi.org/10.13026/kpb9-mt58).
- **eICU-CRD v2.0**: Pollard et al., *Sci Data* 2018; dataset DOI [10.13026/C2WM1R](https://doi.org/10.13026/C2WM1R).

To reproduce:

1. Obtain credentialed access to both databases on PhysioNet and link them to Google BigQuery.
2. Copy this repository to Google Drive as `MyDrive/Plan A/v4/`. The notebooks expect `MyDrive/Plan A/v4/code/transport_utils.py`; set the environment variable `TRANSPORT_BASE` to use another location.
3. Open the notebooks in Colab and run them in order, from NB00 to NB13. Colab already provides LightGBM, XGBoost, scikit-learn, SciPy and statsmodels.
4. Each notebook writes its aggregated outputs to `results/NBxx/` and a zip file. Long steps are cached and resume after a disconnect.

For a quick smoke test, set `os.environ['TRANSPORT_FAST'] = '1'` in the setup cell. That run uses tiny bootstrap and search sizes, and its results are not the paper's.

| Notebook | Content |
|---|---|
| NB00 | Setup and SQL dry run |
| NB01–NB02 | Extraction from MIMIC-IV and eICU-CRD (one harmonized long format) |
| NB03 | Harmonization, cohort and feature audit |
| NB04 | Model development (LightGBM, logistic regression, XGBoost) |
| NB05 | External validation and APACHE IVa comparison |
| NB06 | Hospital-level heterogeneity, funnel plots, meta-regression |
| NB07 | Pipeline-artifact ablation (four feature constructions) |
| NB08 | Updating strategies, 2014 → 2015 |
| NB09 | Label budget |
| NB10 | Reverse transport, temporal drift, subgroups |
| NB11 | Sensitivity analyses and paper assets |
| NB12 | Revision analyses (chance ceilings, bootstrap τ intervals, Holm adjustment, GLMM, leave-region-out, second outcome) |
| NB13 | Method extensions, label-free comparators, label budget 0–400, inputs for the risk analysis, MLP base model |

## Simulation study

```bash
cd code/simulation
SIM_R=50 python3 simulation.py sim_results covariate
SIM_R=50 python3 simulation.py sim_results label
python3 simulation.py sim_results aggregate
python3 simulation.py sim_results tau
```

The script imports `../transport_utils.py`. The full run takes about 30 minutes on two cores.

## Rebuilding the paper

`paper/build.sh <results_dir> <results_dir> <simulation_dir>` regenerates every table and number macro from the notebook outputs and the simulation, then compiles the supplement and the manuscript. The supplement must be compiled first, because the manuscript cites its table numbers. Building needs a LaTeX installation with the IEEEtran bibliography style and the aggregated `results/` folder produced by the notebooks.

## Citation

If you use this code, please cite the paper. Full reference to follow on publication.

## License

The code is released under the MIT License (see `LICENSE`). The databases are subject to their own PhysioNet licenses and data use agreements.
