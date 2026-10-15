# sitecal

Site-level calibration audit and hierarchical (empirical-Bayes) recalibration for a frozen model deployed at many sites.
Part of the repository for "Well Calibrated on Average, Miscalibrated per Site: Hierarchical Recalibration of Machine Learning
Models Under Multi-Site Distribution Shift".

## Audit

```python
from sitecal import site_audit
a = site_audit(y, p, site)          # y: outcomes, p: predicted probabilities, site: site ids
a["tau"], a["prediction_interval"]   # between-site SD of ln(O/E) and 95% PI for a new site
a["pct_in_band"], a["chance_ceiling"]  # % sites with O/E in 0.8-1.2, and what perfect calibration would give
a["pct_outside_95"], a["expected_outside_95"]  # funnel exceedances, observed vs expected
```

## Hierarchical recalibration

```python
from sitecal import HierarchicalRecalibrator
rc = HierarchicalRecalibrator(use_covariates=True).fit(p_upd, y_upd, site_upd, Z=descriptors)
p_new = rc.predict(p_eval, site_eval, Z=descriptors)
# a new site with a handful of labels:
off = rc.site_offset("new", z=z_new, local=(p_local, y_local))
```

Options: `global_step` (Algorithm 1, line 1), `use_covariates` (shrink toward similar sites; Fay-Herriot prior mean),
`semi_supervised` (add an unlabeled EM label-shift estimate as a biased measurement). Sites share only their offset
and its standard error (`site_summary`), so the network prior can be built without pooling individual data.

Risk analysis: `offset_risks`, `break_even_labels` and `shrinkage_wins` implement Proposition 1 (when shrinkage beats
site-only fitting and the prior alone; how many labels a site needs).

Zero-event sites: the event count is clipped to [0.5, n - 0.5] before solving for the offset.

## Tests

```
pip install -e . && pytest tests
```

License: MIT.
