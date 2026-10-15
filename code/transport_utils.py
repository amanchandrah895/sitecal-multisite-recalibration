"""
transport_utils.py -- shared analysis code for the v4 transportability study.

Everything statistical lives here so that every notebook uses exactly the
same definitions. Sections:
  1. Configuration, paths, notebook bookkeeping
  2. Feature definitions
  3. Core metrics (discrimination, calibration hierarchy, ECE, net benefit)
  4. Bootstrap (iid and hospital-cluster)
  5. Random-effects meta-analysis, prediction intervals, funnel limits
  6. Recalibration / model-updating strategies (incl. empirical-Bayes)
  7. Plot style and helpers
"""
from __future__ import annotations

import json
import os
import shutil
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# 1. Configuration
# =============================================================================
BASE = os.environ.get("TRANSPORT_BASE", "/content/drive/MyDrive/Plan A/v4")
DATA_RAW = os.path.join(BASE, "data", "raw")
DATA_AN = os.path.join(BASE, "data", "analysis")
MODELS = os.path.join(BASE, "models")
RESULTS = os.path.join(BASE, "results")

SEED = 42

# pre-specified analysis constants (pre-specified analysis constants)
MIMIC_DEV_GROUPS = ["2008 - 2010", "2011 - 2013", "2014 - 2016"]
EICU_UPDATE_YEAR = 2014
EICU_EVAL_YEAR = 2015
MIN_HOSP_N = 100          # per-hospital analyses: patients in the evaluated data
MIN_HOSP_EVENTS = 10      # ... and deaths
MIN_UPDATE_N = 50         # recalibration strategies: patients in the update data
OE_ACCEPT = (0.8, 1.2)    # "acceptable" calibration-in-the-large
DCA_THRESHOLDS = np.round(np.arange(0.02, 0.51, 0.01), 2)
KEY_THRESHOLDS = [0.1, 0.2]

_CURRENT = {"nb": None, "dir": None, "t0": None}


def init_notebook(nb_id: str, fast: bool | None = None):
    """Create folders, set seeds, remember which notebook is running."""
    for d in (DATA_RAW, DATA_AN, MODELS, RESULTS):
        os.makedirs(d, exist_ok=True)
    out = os.path.join(RESULTS, nb_id)
    os.makedirs(out, exist_ok=True)
    np.random.seed(SEED)
    _CURRENT.update(nb=nb_id, dir=out, t0=time.time())
    fast = FAST if fast is None else fast
    print(f"[{nb_id}] BASE={BASE}")
    print(f"[{nb_id}] results -> {out}   FAST_MODE={fast}")
    return out


# FAST mode (env TRANSPORT_FAST=1) shrinks bootstrap / search sizes for testing
FAST = os.environ.get("TRANSPORT_FAST", "0") == "1"


def n_boot(default=1000):
    return 50 if FAST else default


def out_path(name: str) -> str:
    return os.path.join(_CURRENT["dir"], name)


def save_table(df: pd.DataFrame, name: str, show: bool = True, index=False):
    p = out_path(name if name.endswith(".csv") else name + ".csv")
    df.to_csv(p, index=index)
    if show:
        with pd.option_context("display.max_columns", 50, "display.width", 200):
            print(df.head(40).to_string(index=index))
    print(f"  saved -> {p}")
    return p


def save_json(obj, name: str):
    p = out_path(name if name.endswith(".json") else name + ".json")
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    return p


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return str(o)


def finish_notebook(summary: dict):
    """Write summary.json, zip the results folder, print a paste-back block."""
    nb = _CURRENT["nb"]
    summary = dict(summary)
    summary["_notebook"] = nb
    summary["_runtime_min"] = round((time.time() - _CURRENT["t0"]) / 60, 1)
    summary["_fast_mode"] = FAST
    save_json(summary, "summary.json")
    zpath = shutil.make_archive(os.path.join(RESULTS, f"{nb}_results"), "zip", _CURRENT["dir"])
    print("=" * 70)
    print(f"{nb} COMPLETE in {summary['_runtime_min']} min")
    print(f"Download this file into your local 'Plan A/v4/results/' folder:\n  {zpath}")
    print("=" * 70)
    print(json.dumps(summary, indent=1, default=_json_default)[:6000])
    return zpath


def cached(name: str, fn, *args, **kwargs):
    """Checkpoint helper for free Colab: compute once, reload after disconnects."""
    p = out_path(f"_cache_{name}.pkl")
    if os.path.exists(p):
        print(f"  [cache] loading {name}")
        return pd.read_pickle(p)
    res = fn(*args, **kwargs)
    pd.to_pickle(res, p)
    return res


# =============================================================================
# 2. Feature definitions
# =============================================================================
VITALS = ["hr", "sbp", "dbp", "mbp", "rr", "spo2", "temp"]
LABS = ["creat", "bun", "sodium", "potassium", "chloride", "bicarb", "glucose",
        "wbc", "hgb", "plt", "lactate", "bili", "albumin", "inr", "hct"]
DEMO = ["age", "sex_male"]

V1_VITAL = [f"{v}_{s}" for v in VITALS for s in ("min", "max", "mean")]
V2_VITAL = [f"raw_{v}_{s}" for v in VITALS for s in ("min", "max", "mean")]
LAB_FEATS = [f"{l}_{s}" for l in LABS for s in ("min", "max")]
COUNT_FEATS = [f"cnt_{v}" for v in VITALS] + [f"cnt_{l}" for l in LABS] + ["cnt_gcs"]

FEATURES_V1 = DEMO + V1_VITAL + LAB_FEATS + ["gcs_min"]


def feature_set(variant: str = "V1", df: pd.DataFrame | None = None):
    """V1 hourly values (primary); V2 raw-resolution values; V3 V1+counts;
    V4 V1+missingness indicators (indicators created on the fly)."""
    if variant == "V1":
        return list(FEATURES_V1)
    if variant == "V2":
        return DEMO + V2_VITAL + LAB_FEATS + ["gcs_min"]
    if variant == "V3":
        return FEATURES_V1 + COUNT_FEATS
    if variant == "V4":
        return FEATURES_V1 + [f"miss_{c}" for c in FEATURES_V1 if c not in DEMO]
    raise ValueError(variant)


def add_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    ind = {f"miss_{c}": df[c].isna().astype(int) for c in FEATURES_V1 if c not in DEMO}
    return pd.concat([df, pd.DataFrame(ind, index=df.index)], axis=1)


def load_analysis():
    m = pd.read_parquet(os.path.join(DATA_AN, "mimic_analysis.parquet"))
    e = pd.read_parquet(os.path.join(DATA_AN, "eicu_analysis.parquet"))
    return m, e


# =============================================================================
# 3. Core metrics
# =============================================================================
EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expit(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


def irls(X, y, offset=None, ridge=1e-8, max_iter=100, tol=1e-9):
    """Logistic regression by Newton/IRLS. Returns (beta, cov) or (nan, nan)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, k = X.shape
    off = np.zeros(n) if offset is None else np.asarray(offset, float)
    beta = np.zeros(k)
    R = ridge * np.eye(k)
    for _ in range(max_iter):
        eta = X @ beta + off
        mu = expit(eta)
        w = np.clip(mu * (1 - mu), 1e-10, None)
        H = X.T @ (X * w[:, None]) + R
        g = X.T @ (y - mu) - R @ beta
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return np.full(k, np.nan), np.full((k, k), np.nan)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
        if np.any(np.abs(beta) > 50):
            return np.full(k, np.nan), np.full((k, k), np.nan)
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.full((k, k), np.nan)
    return beta, cov


def calibration_intercept_slope(y, p):
    """Weak calibration (Van Calster 2016):
      CITL  = intercept of logit(P(y)) = a + 1*logit(p)   (offset model)
      slope = b  in logit(P(y)) = a + b*logit(p)
    Returns dict with estimates and SEs."""
    y = np.asarray(y, float)
    lp = logit(p)
    out = dict(citl=np.nan, citl_se=np.nan, slope=np.nan, slope_se=np.nan)
    if y.sum() == 0 or y.sum() == len(y):
        return out
    b, c = irls(np.ones((len(y), 1)), y, offset=lp)
    out["citl"], out["citl_se"] = b[0], np.sqrt(c[0, 0])
    b2, c2 = irls(np.column_stack([np.ones(len(y)), lp]), y)
    out["slope"], out["slope_se"] = b2[1], np.sqrt(c2[1, 1])
    return out


def _spline_basis(x, knots):
    """Restricted (natural) cubic spline basis, Harrell parameterisation."""
    x = np.asarray(x, float)
    k = np.asarray(knots, float)
    K = len(k)
    cols = [x]
    denom = (k[-1] - k[0]) ** 2
    for j in range(K - 2):
        t = (np.clip(x - k[j], 0, None) ** 3
             - np.clip(x - k[-2], 0, None) ** 3 * (k[-1] - k[j]) / (k[-1] - k[-2])
             + np.clip(x - k[-1], 0, None) ** 3 * (k[-2] - k[j]) / (k[-1] - k[-2]))
        cols.append(t / denom)
    return np.column_stack(cols)


def flexible_calibration(y, p, n_knots=4, grid=None):
    """Moderate calibration curve: logistic regression of y on a restricted
    cubic spline of logit(p) (Austin & Steyerberg 2019). Returns calibrated
    probabilities for each p (and optionally on a grid)."""
    y = np.asarray(y, float)
    lp = logit(p)
    if y.sum() < 5 or (len(y) - y.sum()) < 5:
        return None, None
    if y.sum() < 30:
        n_knots = 3
    q = {3: [10, 50, 90], 4: [5, 35, 65, 95], 5: [5, 27.5, 50, 72.5, 95]}[n_knots]
    knots = np.unique(np.percentile(lp, q))
    if len(knots) < 3:                       # many ties (e.g. isotonic output)
        knots = np.unique(np.percentile(np.unique(lp), q))
    if len(knots) < 3:
        return None, None
    B = np.column_stack([np.ones(len(lp)), _spline_basis(lp, knots)])
    beta, _ = irls(B, y, ridge=1e-6)
    if np.any(np.isnan(beta)):
        return None, None
    fitted = expit(B @ beta)
    grid_out = None
    if grid is not None:
        lg = logit(grid)
        Bg = np.column_stack([np.ones(len(lg)), _spline_basis(lg, knots)])
        grid_out = expit(Bg @ beta)
    return fitted, grid_out


def ici_metrics(y, p):
    fitted, _ = flexible_calibration(y, p)
    if fitted is None:
        return dict(ici=np.nan, e50=np.nan, e90=np.nan, emax=np.nan)
    d = np.abs(fitted - np.asarray(p, float))
    return dict(ici=d.mean(), e50=np.median(d), e90=np.percentile(d, 90), emax=d.max())


def ece(y, p, n_bins=10):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1], right=False), 0, n_bins - 1)
    tot = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return tot


def auroc_se_hanley(auc, n1, n0):
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    v = (auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n0 - 1) * (q2 - auc ** 2)) / (n1 * n0)
    return np.sqrt(max(v, 1e-12))


def evaluate(y, p, full=True, ece15_=True):
    """All metrics for one set of predictions."""
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    n, o = len(y), y.sum()
    e = p.sum()
    r = dict(n=n, events=int(o), prevalence=o / n if n else np.nan, mean_pred=p.mean(),
             oe=o / e if e > 0 else np.nan)
    ok = 0 < o < n
    r["auroc"] = roc_auc_score(y, p) if ok else np.nan
    r["auprc"] = average_precision_score(y, p) if ok else np.nan
    brier = np.mean((p - y) ** 2)
    bnull = r["prevalence"] * (1 - r["prevalence"])
    r["brier"] = brier
    r["scaled_brier"] = 1 - brier / bnull if bnull > 0 else np.nan
    r.update(calibration_intercept_slope(y, p))
    if full:
        r.update(ici_metrics(y, p))
        r["ece10"] = ece(y, p, 10)
        if ece15_:
            r["ece15"] = ece(y, p, 15)
    return r


def net_benefit(y, p, t):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    n = len(y)
    pos = p >= t
    tp = np.sum(pos & (y == 1))
    fp = np.sum(pos & (y == 0))
    return tp / n - fp / n * t / (1 - t)


def net_benefit_all(y, t):
    prev = np.mean(y)
    return prev - (1 - prev) * t / (1 - t)


def dca_table(y, preds: dict, thresholds=None):
    thresholds = DCA_THRESHOLDS if thresholds is None else thresholds
    rows = []
    for t in thresholds:
        row = dict(threshold=t, treat_all=net_benefit_all(y, t), treat_none=0.0)
        for k, p in preds.items():
            row[k] = net_benefit(y, p, t)
        rows.append(row)
    return pd.DataFrame(rows)


def threshold_stats(y, p, t):
    y = np.asarray(y, int)
    pos = np.asarray(p) >= t
    tp = int(np.sum(pos & (y == 1)))
    fp = int(np.sum(pos & (y == 0)))
    fn = int(np.sum(~pos & (y == 1)))
    tn = int(np.sum(~pos & (y == 0)))
    n = len(y)
    return dict(threshold=t, flagged_per_100=100 * pos.mean(), sensitivity=tp / max(tp + fn, 1),
                specificity=tn / max(tn + fp, 1), ppv=tp / max(tp + fp, 1),
                tp_per_100=100 * tp / n, fp_per_100=100 * fp / n, net_benefit=net_benefit(y, p, t))


# =============================================================================
# 4. Bootstrap
# =============================================================================
BOOT_METRICS = ["auroc", "auprc", "brier", "scaled_brier", "oe", "citl", "slope", "ici", "ece10"]


def bootstrap(y, p, clusters=None, B=None, seed=SEED, metrics=BOOT_METRICS, label=""):
    """Percentile bootstrap CIs. If `clusters` is given (e.g. hospital id),
    whole clusters are resampled (accounts for within-hospital correlation)."""
    B = n_boot() if B is None else B
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    rng = np.random.default_rng(seed)
    point = evaluate(y, p)
    if clusters is not None:
        cl = pd.Series(np.arange(len(y))).groupby(np.asarray(clusters)).apply(np.array).values
    draws = []
    t0 = time.time()
    for b in range(B):
        if b and b % 100 == 0:
            el = time.time() - t0
            print(f"    bootstrap {label} {b}/{B}  elapsed {el/60:.1f} min, about {(B-b)*el/b/60:.1f} min left", flush=True)
        if clusters is None:
            idx = rng.integers(0, len(y), len(y))
        else:
            pick = rng.integers(0, len(cl), len(cl))
            idx = np.concatenate([cl[i] for i in pick])
        yy, pp = y[idx], p[idx]
        if yy.sum() in (0, len(yy)):
            continue
        r = evaluate(yy, pp, full=("ici" in metrics), ece15_=False)
        draws.append({m: r.get(m, np.nan) for m in metrics})
    d = pd.DataFrame(draws)
    rows = []
    for m in metrics:
        v = d[m].dropna() if m in d else pd.Series(dtype=float)
        rows.append(dict(metric=m, estimate=point.get(m, np.nan),
                         lo=np.percentile(v, 2.5) if len(v) else np.nan,
                         hi=np.percentile(v, 97.5) if len(v) else np.nan))
    return pd.DataFrame(rows), point


def fmt_ci(est, lo, hi, nd=3):
    return f"{est:.{nd}f} ({lo:.{nd}f}-{hi:.{nd}f})"


def paired_bootstrap_diff(y, p1, p2, metric="auroc", clusters=None, B=None, seed=SEED):
    """CI for metric(p1) - metric(p2) on the same patients."""
    B = n_boot() if B is None else B
    y = np.asarray(y, float)
    rng = np.random.default_rng(seed)
    if clusters is not None:
        cl = pd.Series(np.arange(len(y))).groupby(np.asarray(clusters)).apply(np.array).values
    diffs = []
    for _ in range(B):
        if clusters is None:
            idx = rng.integers(0, len(y), len(y))
        else:
            idx = np.concatenate([cl[i] for i in rng.integers(0, len(cl), len(cl))])
        yy = y[idx]
        if yy.sum() in (0, len(yy)):
            continue
        a = evaluate(yy, np.asarray(p1)[idx], full=False)[metric]
        b = evaluate(yy, np.asarray(p2)[idx], full=False)[metric]
        diffs.append(a - b)
    est = evaluate(y, p1, False)[metric] - evaluate(y, p2, False)[metric]
    diffs = np.array(diffs)
    return dict(metric=metric, diff=est, lo=np.percentile(diffs, 2.5), hi=np.percentile(diffs, 97.5),
                p_two_sided=2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))


# =============================================================================
# 5. Per-hospital metrics, meta-analysis, funnel plots
# =============================================================================
def per_hospital(df, pcol, ycol="mortality", hcol="hospital_id", min_n=MIN_HOSP_N,
                 min_events=MIN_HOSP_EVENTS, full=True):
    """Metrics per hospital + the SEs needed for meta-analysis (Debray 2017)."""
    rows = []
    for h, g in df.groupby(hcol):
        y = g[ycol].values.astype(float)
        p = g[pcol].values.astype(float)
        n, o = len(y), y.sum()
        e = p.sum()
        r = dict(hospital_id=h, n=n, events=int(o), expected=e,
                 eligible=bool(n >= min_n and o >= min_events))
        r["oe"] = o / e if e > 0 else np.nan
        r["ln_oe"] = np.log(o / e) if (o > 0 and e > 0) else np.nan
        r["ln_oe_se"] = np.sqrt((1 - o / n) / o) if o > 0 else np.nan
        if 0 < o < n:
            auc = roc_auc_score(y, p)
            r["auroc"] = auc
            se = auroc_se_hanley(auc, o, n - o)
            auc_c = np.clip(auc, 0.01, 0.99)
            r["logit_auroc"] = np.log(auc_c / (1 - auc_c))
            r["logit_auroc_se"] = se / (auc_c * (1 - auc_c))
            cs = calibration_intercept_slope(y, p)
            r.update(cs)
            if full:
                r.update(ici_metrics(y, p))
        rows.append(r)
    return pd.DataFrame(rows)


def meta_analysis(est, se, method="DL", hksj=True):
    """Random-effects meta-analysis (DerSimonian-Laird tau^2) with the
    Hartung-Knapp-Sidik-Jonkman CI and a 95% prediction interval
    (Higgins 2009: t_{k-2} * sqrt(tau^2 + SE(mu)^2))."""
    est = np.asarray(est, float)
    se = np.asarray(se, float)
    m = np.isfinite(est) & np.isfinite(se) & (se > 0)
    est, se = est[m], se[m]
    k = len(est)
    if k < 3:
        return dict(k=k, mu=np.nan, lo=np.nan, hi=np.nan, tau2=np.nan, tau=np.nan,
                    pi_lo=np.nan, pi_hi=np.nan, i2=np.nan, q=np.nan)
    w = 1 / se ** 2
    mu_fe = np.sum(w * est) / np.sum(w)
    q = np.sum(w * (est - mu_fe) ** 2)
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (q - (k - 1)) / c)
    ws = 1 / (se ** 2 + tau2)
    mu = np.sum(ws * est) / np.sum(ws)
    if hksj:
        var_mu = np.sum(ws * (est - mu) ** 2) / ((k - 1) * np.sum(ws))
        var_mu = max(var_mu, 1 / np.sum(ws))   # HKSJ safeguard
        tq = stats.t.ppf(0.975, k - 1)
    else:
        var_mu = 1 / np.sum(ws)
        tq = stats.norm.ppf(0.975)
    se_mu = np.sqrt(var_mu)
    tpi = stats.t.ppf(0.975, k - 2)
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    return dict(k=k, mu=mu, lo=mu - tq * se_mu, hi=mu + tq * se_mu, tau2=tau2, tau=np.sqrt(tau2),
                pi_lo=mu - tpi * np.sqrt(tau2 + se_mu ** 2), pi_hi=mu + tpi * np.sqrt(tau2 + se_mu ** 2),
                i2=i2, q=q)


def summarise_hospitals(ph: pd.DataFrame, label: str):
    """Meta-analyse O/E (log), c-statistic (logit) and slope across eligible hospitals."""
    e = ph[ph["eligible"]]
    out = []
    for name, est, se, back in [
        ("O/E", "ln_oe", "ln_oe_se", np.exp),
        ("c-statistic", "logit_auroc", "logit_auroc_se", expit),
        ("calibration slope", "slope", "slope_se", lambda x: x),
        ("calibration-in-the-large", "citl", "citl_se", lambda x: x),
    ]:
        if est not in e:
            continue
        ma = meta_analysis(e[est], e[se])
        out.append(dict(model=label, measure=name, k=ma["k"], pooled=back(ma["mu"]),
                        ci_lo=back(ma["lo"]), ci_hi=back(ma["hi"]),
                        pi_lo=back(ma["pi_lo"]), pi_hi=back(ma["pi_hi"]),
                        tau=ma["tau"], i2=ma["i2"]))
    if "oe" in e:
        acc = e["oe"].between(*OE_ACCEPT)
        out.append(dict(model=label, measure=f"% hospitals with O/E in {OE_ACCEPT}", k=len(e),
                        pooled=100 * acc.mean(), ci_lo=np.nan, ci_hi=np.nan, pi_lo=np.nan,
                        pi_hi=np.nan, tau=np.nan, i2=np.nan))
    return pd.DataFrame(out)


def funnel_limits(expected_grid, levels=(0.95, 0.998)):
    """Exact Poisson control limits for O/E around 1 as a function of the
    expected number of deaths (Spiegelhalter 2005)."""
    rows = []
    for E in expected_grid:
        r = dict(expected=E)
        for lv in levels:
            a = (1 - lv) / 2
            lo = stats.poisson.ppf(a, E) / E
            hi = stats.poisson.ppf(1 - a, E) / E
            r[f"lo_{lv}"], r[f"hi_{lv}"] = lo, hi
        rows.append(r)
    return pd.DataFrame(rows)


def outside_funnel(ph, level):
    a = (1 - level) / 2
    E = ph["expected"].values
    O = ph["events"].values
    lo = stats.poisson.ppf(a, E)
    hi = stats.poisson.ppf(1 - a, E)
    return (O < lo) | (O > hi)


# =============================================================================
# 6. Recalibration / updating strategies
# =============================================================================
def fit_intercept(p, y, clip_events=True):
    """Intercept-only update: logit(p_new) = a + logit(p). Robust to 0 or n
    events (continuity 0.5). Returns (a, se)."""
    y = np.asarray(y, float)
    lp = logit(p)
    n, o = len(y), y.sum()
    target = np.clip(o, 0.5, n - 0.5) if clip_events else o
    a = np.log(max(target, 1e-9) / max(np.sum(expit(lp)), 1e-9))
    for _ in range(100):
        mu = expit(a + lp)
        g = target - mu.sum()
        h = np.sum(mu * (1 - mu))
        step = g / max(h, 1e-12)
        a += step
        if abs(step) < 1e-10:
            break
    mu = expit(a + lp)
    se = 1 / np.sqrt(max(np.sum(mu * (1 - mu)), 1e-12))
    return a, se


def fit_logistic(p, y):
    """Logistic recalibration: logit(p_new) = a + b*logit(p). Returns (a, b)."""
    y = np.asarray(y, float)
    if y.sum() < 2 or (len(y) - y.sum()) < 2:
        return np.nan, np.nan
    beta, _ = irls(np.column_stack([np.ones(len(y)), logit(p)]), y)
    return beta[0], beta[1]


def apply_logistic(p, a, b=1.0):
    return expit(a + b * logit(p))


def fit_isotonic(p, y):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS)
    iso.fit(np.asarray(p, float), np.asarray(y, float))
    return iso


def eb_site_intercepts(p, y, sites):
    """Empirical-Bayes (random-intercept) site recalibration.
    1) per-site intercept a_h and SE via fit_intercept;
    2) DerSimonian-Laird estimate of between-site variance tau^2 and mean mu;
    3) shrink: a_h* = mu + B_h (a_h - mu), B_h = tau^2 / (tau^2 + se_h^2).
    Small sites borrow strength from the network; large sites keep their own
    estimate. Returns (dict site->a_shrunk, mu, tau2, table)."""
    df = pd.DataFrame(dict(p=np.asarray(p, float), y=np.asarray(y, float), s=np.asarray(sites)))
    rows = []
    for s, g in df.groupby("s"):
        a, se = fit_intercept(g["p"].values, g["y"].values)
        rows.append(dict(site=s, n=len(g), events=g["y"].sum(), a=a, se=se))
    t = pd.DataFrame(rows)
    ma = meta_analysis(t["a"], t["se"], hksj=False)
    mu, tau2 = ma["mu"], ma["tau2"]
    t["B"] = tau2 / (tau2 + t["se"] ** 2)
    t["a_eb"] = mu + t["B"] * (t["a"] - mu)
    return dict(zip(t["site"], t["a_eb"])), mu, tau2, t


def eb_shrink_single(a_local, se_local, mu, tau2):
    B = tau2 / (tau2 + se_local ** 2)
    return mu + B * (a_local - mu)


STRATEGY_LABELS = {
    "S0_none": "No recalibration",
    "S1a_global_intercept": "Global intercept",
    "S1b_global_logistic": "Global logistic",
    "S1c_global_isotonic": "Global isotonic",
    "S2a_local_intercept": "Local intercept",
    "S2b_local_logistic": "Local logistic",
    "S2c_local_isotonic": "Local isotonic",
    "S3_eb": "Global logistic + EB local intercept",
    "S4_apache": "APACHE IVa (reference)",
    "S5_refit": "Refit on target (LightGBM)",
}


def run_strategies(upd: pd.DataFrame, ev: pd.DataFrame, pcol: str, ycol="mortality",
                   hcol="hospital_id", apache_col="apache_pred", refit_pred_col=None):
    """Fit every updating strategy on `upd` (update data) and return a copy
    of `ev` (evaluation data) with one prediction column per strategy.
    Local strategies are fitted per hospital with that hospital's update data;
    hospitals without update data fall back to the global model."""
    ev = ev.copy()
    pu, yu = upd[pcol].values, upd[ycol].values
    pe = ev[pcol].values
    ev["S0_none"] = pe
    a0, _ = fit_intercept(pu, yu)
    ev["S1a_global_intercept"] = apply_logistic(pe, a0)
    ag, bg = fit_logistic(pu, yu)
    ev["S1b_global_logistic"] = apply_logistic(pe, ag, bg)
    iso_g = fit_isotonic(pu, yu)
    ev["S1c_global_isotonic"] = iso_g.transform(pe)

    # local (per hospital) fits
    fails = dict(S2b=0)
    loc_a, loc_ab, loc_iso = {}, {}, {}
    for h, g in upd.groupby(hcol):
        pg, yg = g[pcol].values, g[ycol].values
        loc_a[h] = fit_intercept(pg, yg)[0]
        a, b = fit_logistic(pg, yg)
        if not np.isfinite(a) or not np.isfinite(b) or b <= 0:
            fails["S2b"] += 1
            a, b = loc_a[h], 1.0          # fallback: intercept only
        loc_ab[h] = (a, b)
        loc_iso[h] = fit_isotonic(pg, yg) if 0 < yg.sum() < len(yg) else None
    hv = ev[hcol].values
    a_arr = np.array([loc_a.get(h, a0) for h in hv])
    ab_arr = np.array([loc_ab.get(h, (ag, bg)) for h in hv])
    s2a = apply_logistic(pe, a_arr)
    s2b = apply_logistic(pe, ab_arr[:, 0], ab_arr[:, 1])
    s2c = np.empty(len(ev))
    for h in np.unique(hv):
        m = hv == h
        iso = loc_iso.get(h)
        s2c[m] = iso.transform(pe[m]) if iso is not None else iso_g.transform(pe[m])
    ev["S2a_local_intercept"], ev["S2b_local_logistic"], ev["S2c_local_isotonic"] = s2a, s2b, s2c

    # empirical Bayes on top of the global logistic recalibration
    pu_g = apply_logistic(pu, ag, bg)
    eb, mu, tau2, eb_tab = eb_site_intercepts(pu_g, yu, upd[hcol].values)
    shift = np.array([eb.get(h, mu) for h in hv])
    ev["S3_eb"] = expit(logit(ev["S1b_global_logistic"].values) + shift)

    if apache_col in ev:
        ev["S4_apache"] = ev[apache_col].values
    if refit_pred_col is not None:
        ev["S5_refit"] = ev[refit_pred_col].values
    info = dict(global_intercept=a0, global_a=ag, global_b=bg, eb_mu=mu, eb_tau2=tau2,
                eb_tau=np.sqrt(tau2), local_logistic_fallbacks=fails["S2b"],
                n_update_hospitals=upd[hcol].nunique())
    return ev, info, eb_tab


# =============================================================================
# 7. Plot style
# =============================================================================
# categorical order from the validated reference palette (light mode)
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100",
         magenta="#e87ba4", green="#008300", violet="#4a3aa7", red="#e34948",
         grey="#8a8985", ink="#0b0b0b", ink2="#52514e", grid="#e6e5e0")
SERIES = [C["blue"], C["orange"], C["aqua"], C["yellow"], C["magenta"], C["green"], C["violet"], C["red"]]


def set_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "font.size": 9,
        "font.family": "DejaVu Sans", "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.labelsize": 9, "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6,
        "xtick.color": C["ink2"], "ytick.color": C["ink2"], "lines.linewidth": 2,
        "legend.frameon": False, "legend.fontsize": 8, "axes.prop_cycle": mpl.cycler(color=SERIES),
    })


def save_fig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(out_path(f"{name}.{ext}"), bbox_inches="tight")
    print(f"  figure -> {out_path(name + '.png')}")


def plot_calibration(ax, curves: dict, title="", hist_p=None):
    """curves: label -> (y, p). Draws spline calibration curves + ideal line."""
    grid = np.linspace(0.005, 0.8, 200)
    ax.plot([0, 0.8], [0, 0.8], ls="--", lw=1, color=C["grey"], label="Ideal")
    for i, (lab, (y, p)) in enumerate(curves.items()):
        _, g = flexible_calibration(y, p, grid=grid)
        if g is None:
            continue
        lo, hi = np.percentile(p, [0.5, 99.5])
        m = (grid >= lo) & (grid <= hi)
        ax.plot(grid[m], g[m], color=SERIES[i % 8], label=lab)
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel("Predicted risk")
    ax.set_ylabel("Observed risk (spline-smoothed)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    if hist_p is not None:
        ax2 = ax.inset_axes([0, 0, 1, 0.12], sharex=ax)
        ax2.hist(np.clip(hist_p, 0, 0.8), bins=80, color=C["grey"], alpha=0.6)
        ax2.axis("off")
