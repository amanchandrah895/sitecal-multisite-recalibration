"""Core numerical helpers: logits, intercept/logistic recalibration, random-effects meta-analysis."""
from __future__ import annotations

import numpy as np
from scipy import stats

EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expit(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


def fit_offset(p, y, continuity=0.5, max_iter=100, tol=1e-10):
    """Logit offset a such that sum(expit(logit(p) + a)) equals the observed events.

    Sites with zero or all events have no finite solution; the event count is then clipped to
    [continuity, n - continuity]. Returns (a, standard error)."""
    y = np.asarray(y, float)
    lp = logit(p)
    n, o = len(y), y.sum()
    target = np.clip(o, continuity, n - continuity)
    a = np.log(max(target, 1e-12) / max(expit(lp).sum(), 1e-12))
    for _ in range(max_iter):
        mu = expit(a + lp)
        step = (target - mu.sum()) / max(np.sum(mu * (1 - mu)), 1e-12)
        a += step
        if abs(step) < tol:
            break
    mu = expit(a + lp)
    return float(a), float(1 / np.sqrt(max(np.sum(mu * (1 - mu)), 1e-12)))


def fit_logistic_recal(p, y, max_iter=100, tol=1e-10):
    """Logistic recalibration logit(q) = a + b * logit(p) by Newton-Raphson. Returns (a, b)."""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y)), logit(p)])
    beta = np.array([0.0, 1.0])
    for _ in range(max_iter):
        mu = expit(X @ beta)
        H = X.T @ (X * (mu * (1 - mu))[:, None]) + 1e-8 * np.eye(2)
        step = np.linalg.solve(H, X.T @ (y - mu))
        beta += step
        if np.max(np.abs(step)) < tol:
            break
    return float(beta[0]), float(beta[1])


def random_effects(est, se, hksj=True):
    """DerSimonian-Laird random-effects meta-analysis with Hartung-Knapp CI and 95% prediction interval."""
    est, se = np.asarray(est, float), np.asarray(se, float)
    keep = np.isfinite(est) & np.isfinite(se) & (se > 0)
    est, se = est[keep], se[keep]
    k = len(est)
    if k < 3:
        raise ValueError("need at least three sites")
    w = 1 / se ** 2
    mu_fe = np.sum(w * est) / np.sum(w)
    q = np.sum(w * (est - mu_fe) ** 2)
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (q - (k - 1)) / c)
    ws = 1 / (se ** 2 + tau2)
    mu = np.sum(ws * est) / np.sum(ws)
    if hksj:
        var_mu = max(np.sum(ws * (est - mu) ** 2) / ((k - 1) * np.sum(ws)), 1 / np.sum(ws))
        tq = stats.t.ppf(0.975, k - 1)
    else:
        var_mu, tq = 1 / np.sum(ws), stats.norm.ppf(0.975)
    tpi = stats.t.ppf(0.975, k - 2)
    half = tpi * np.sqrt(tau2 + var_mu)
    return dict(k=k, mu=mu, ci=(mu - tq * np.sqrt(var_mu), mu + tq * np.sqrt(var_mu)), tau2=tau2, tau=np.sqrt(tau2),
                pi=(mu - half, mu + half), i2=max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0)


def meta_regression(est, se, Z):
    """Mixed-effects meta-regression with method-of-moments residual tau^2. Z must include an intercept column.
    Returns (coefficients, residual tau^2)."""
    a, v, X = np.asarray(est, float), np.asarray(se, float) ** 2, np.asarray(Z, float)
    W = np.diag(1 / v)
    P = W - W @ X @ np.linalg.pinv(X.T @ W @ X) @ X.T @ W
    tau2 = max(0.0, (float(a @ P @ a) - (len(a) - X.shape[1])) / np.trace(P))
    W2 = np.diag(1 / (v + tau2))
    return np.linalg.pinv(X.T @ W2 @ X) @ X.T @ W2 @ a, tau2


def em_label_shift(q, prior_source, max_iter=1000, tol=1e-10):
    """Saerens-Latinne-Decaestecker EM estimate of the target prior under pure label shift (no labels needed).
    Returns (target prior, logit offset = logit(target) - logit(source)). Unreliable when sites also differ in
    case mix (covariate shift): the case-mix difference is then mistaken for a prevalence change."""
    q = np.clip(np.asarray(q, float), EPS, 1 - EPS)
    pi = prior_source
    for _ in range(max_iter):
        w1 = q * pi / prior_source
        w0 = (1 - q) * (1 - pi) / (1 - prior_source)
        new = float(np.mean(w1 / (w1 + w0)))
        if abs(new - pi) < tol:
            pi = new
            break
        pi = new
    pi = min(max(pi, EPS), 1 - EPS)
    return pi, float(logit(pi) - logit(prior_source))
