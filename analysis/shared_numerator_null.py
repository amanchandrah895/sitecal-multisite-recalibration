"""Spearman correlation between the hospital-level O/E of two models expected from their shared numerator alone.

Outcomes are simulated under perfect calibration of model 1 (deaths ~ Binomial(n, E1/n) per hospital); deaths in the
subset of stays scored by model 2 are drawn by binomial thinning. Uses the per-hospital tables written by NB06.
Usage: python3 shared_numerator_null.py <results_dir> [n_sim]"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

R = sys.argv[1]
B = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
L = pd.read_csv(f"{R}/NB06/per_hospital_lightgbm.csv")
A = pd.read_csv(f"{R}/NB06/per_hospital_apache.csv")
j = L[["hospital_id", "n", "expected", "eligible", "ln_oe"]].merge(
    A[["hospital_id", "n", "expected", "ln_oe"]], on="hospital_id", suffixes=("_l", "_a"))
j = j[j["eligible"]].dropna()
obs = spearmanr(j["ln_oe_l"], j["ln_oe_a"])[0]
rng = np.random.default_rng(1)
frac = (j["n_a"] / j["n_l"]).clip(0, 1).values
rho = []
for _ in range(B):
    o_l = rng.binomial(j["n_l"].astype(int).values, (j["expected_l"] / j["n_l"]).clip(1e-6, 1 - 1e-6).values)
    o_a = rng.binomial(o_l, frac)
    rho.append(spearmanr(np.log(np.maximum(o_l, 0.5) / j["expected_l"]), np.log(np.maximum(o_a, 0.5) / j["expected_a"]))[0])
print(f"hospitals {len(j)}; observed rho {obs:.3f}; null mean {np.mean(rho):.3f} "
      f"(95% range {np.percentile(rho, 2.5):.3f} to {np.percentile(rho, 97.5):.3f})")
