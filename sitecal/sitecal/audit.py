"""Site-level calibration audit: per-site O/E, random-effects summary, chance ceiling and funnel limits."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .core import random_effects


def per_site(y, p, site):
    """Observed and expected events, O/E, ln(O/E) and its standard error for every site."""
    df = pd.DataFrame(dict(y=np.asarray(y, float), p=np.asarray(p, float), site=np.asarray(site)))
    g = df.groupby("site").agg(n=("y", "size"), observed=("y", "sum"), expected=("p", "sum")).reset_index()
    g["oe"] = g["observed"] / g["expected"]
    with np.errstate(divide="ignore", invalid="ignore"):
        g["ln_oe"] = np.where(g["observed"] > 0, np.log(g["oe"]), np.nan)
        g["ln_oe_se"] = np.where(g["observed"] > 0, np.sqrt((1 - g["observed"] / g["n"]) / g["observed"]), np.nan)
    return g


def funnel_outside(observed, expected, level=0.95):
    """True where the observed count lies outside exact Poisson control limits around the expected count."""
    a = (1 - level) / 2
    lo, hi = stats.poisson.ppf(a, expected), stats.poisson.ppf(1 - a, expected)
    return (observed < lo) | (observed > hi)


def chance_ceiling(p, site, band=(0.8, 1.2), n_sim=1000, seed=0, sites=None):
    """Expected site-level statistics if the model were perfectly calibrated (y ~ Bernoulli(p)).

    Returns the expected percentage of sites with O/E inside `band`, its 95% range, the expected median
    |ln O/E| and the expected percentages outside the 95% and 99.8% funnel limits."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(dict(p=np.asarray(p, float), site=np.asarray(site)))
    if sites is not None:
        df = df[df["site"].isin(sites)]
    groups = [g["p"].values for _, g in df.groupby("site")]
    E = np.array([g.sum() for g in groups])
    O = np.zeros((n_sim, len(groups)))
    for j, pj in enumerate(groups):
        for r0 in range(0, n_sim, 250):
            r1 = min(n_sim, r0 + 250)
            O[r0:r1, j] = (rng.random((r1 - r0, len(pj))) < pj).sum(1)
    oe = O / E
    inband = 100 * ((oe >= band[0]) & (oe <= band[1])).mean(1)
    out95 = 100 * funnel_outside(O, E, 0.95).mean(1)
    out998 = 100 * funnel_outside(O, E, 0.998).mean(1)
    return dict(pct_in_band=inband.mean(), pct_in_band_range=(np.percentile(inband, 2.5), np.percentile(inband, 97.5)),
                median_abs_ln_oe=float(np.median(np.abs(np.log(np.maximum(O, 0.5) / E)), axis=1).mean()),
                pct_outside_95=out95.mean(), pct_outside_998=out998.mean())


def site_audit(y, p, site, band=(0.8, 1.2), min_n=100, min_events=10, n_sim=1000, seed=0):
    """Complete site-level calibration audit of predicted probabilities `p` for outcomes `y` across sites.

    Returns a dict with the per-site table, the case-level pooled O/E, the random-effects mean O/E with CI,
    tau (between-site SD of ln O/E), the 95% prediction interval for a new site's O/E, the observed percentage
    of eligible sites inside `band` with its chance ceiling, and observed versus expected funnel exceedances."""
    ps = per_site(y, p, site)
    elig = ps[(ps["n"] >= min_n) & (ps["observed"] >= min_events)]
    re = random_effects(elig["ln_oe"], elig["ln_oe_se"])
    cc = chance_ceiling(p, site, band, n_sim, seed, sites=set(elig["site"]))
    fun = ps[ps["expected"] >= 1]
    return dict(per_site=ps, sites=len(elig), pooled_oe=float(ps["observed"].sum() / ps["expected"].sum()),
                re_mean_oe=float(np.exp(re["mu"])), re_ci=tuple(np.exp(re["ci"])), tau=re["tau"], i2=re["i2"],
                prediction_interval=tuple(np.exp(re["pi"])),
                pct_in_band=100 * float(elig["oe"].between(*band).mean()), chance_ceiling=cc["pct_in_band"],
                median_abs_ln_oe=float(elig["ln_oe"].abs().median()), null_median_abs_ln_oe=cc["median_abs_ln_oe"],
                pct_outside_95=100 * float(funnel_outside(fun["observed"], fun["expected"], 0.95).mean()),
                expected_outside_95=cc["pct_outside_95"],
                pct_outside_998=100 * float(funnel_outside(fun["observed"], fun["expected"], 0.998).mean()),
                expected_outside_998=cc["pct_outside_998"])
