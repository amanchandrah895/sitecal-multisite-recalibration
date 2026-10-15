"""Closed-form risk of site-offset estimators and predicted label-budget curves.

For a site with true next-period offset a_h (relative to the prior mean m_h, deviation d_h), n local labels and
per-label Fisher information I_h = mean p(1-p), the local offset estimate has variance s^2 = 1 / (n I_h).
Mean squared error of the offset (plus drift omega^2 between update and evaluation periods):
    prior only (n = 0):    d_h^2 + omega^2
    site only:             s^2 + omega^2
    shrinkage, factor B:   (1 - B)^2 d_h^2 + B^2 s^2 + omega^2,     B = t2 / (t2 + s^2)
If d_h^2 is replaced by its expectation t2 (the prior is right on average), the shrinkage risk is t2 s^2/(t2+s^2).
On the ln(O/E) scale the offset error is multiplied by kappa_h = sum p(1-p) / sum p, and the evaluation sample adds
sampling noise (1 - O/n)/O. Predicted mean squared ln(O/E) is averaged over the hospitals of a set.
"""
import numpy as np
import pandas as pd


def offset_risk(d2, s2, t2, omega2, rho=1.0):
    """Deployment-period offset a = m + rho * d + omega (rho = persistence of the update-period deviation d)."""
    s2 = np.asarray(s2, float)
    fin = np.isfinite(s2)
    B = np.where(fin, t2 / (t2 + np.where(fin, s2, 1.0)), 0.0)
    s2z = np.where(fin, s2, 0.0)
    return dict(prior=rho ** 2 * d2 + omega2, local=(1 - rho) ** 2 * d2 + s2 + omega2,
                shrink=(B - rho) ** 2 * d2 + B ** 2 * s2z + omega2)


def persistence(sites, mu):
    """Errors-in-variables estimate of the persistence rho of hospital offsets from the update to the evaluation
    period, and the residual drift variance omega^2: a15 - mu = rho (a14 - mu) + omega."""
    x, y = sites["a14"] - mu, sites["a15"] - mu
    v14 = float(np.mean(x ** 2 - sites["s14"] ** 2))
    rho = float(np.mean(x * y) / v14)
    om2 = max(0.0, float(np.mean((y - rho * x) ** 2 - sites["s15"] ** 2 - rho ** 2 * sites["s14"] ** 2)))
    return rho, om2


def predicted_curves(sites, params, ns=(0, 25, 50, 100, 200, 400), rho=1.0):
    """sites: H_theory_sites table; params: one-row H_theory_params. Returns predicted mean squared ln(O/E).
    rho=None: estimate persistence and drift from the two years (persistence); rho=1: pure drift model."""
    p = params.iloc[0]
    if rho is None:
        rho, om2 = persistence(sites, float(p["mu_net"]))
    else:
        om2 = float(p["omega2_drift"])
    s = sites.copy()
    kappa = s["kappa15"].values if "kappa15" in s else np.ones(len(s))
    noise = (1 - s["o15"] / s["n15"]) / s["o15"] if "o15" in s else s["s15"] ** 2 * kappa ** 2
    d2_eb = (s["a14"] - p["mu_net"]) ** 2 - s["s14"] ** 2          # unbiased per-site deviation from the prior mean
    d2_cov = (s["a14"] - s["m_cov"]) ** 2 - s["s14"] ** 2
    rows = []
    for setname, keep_fn in [("all", lambda n: s["n14"] >= n), ("fixed35", lambda n: s["large"].astype(bool))]:
        for n in ns:
            k = keep_fn(n).values
            s2 = np.full(len(s), np.inf) if n == 0 else 1.0 / (n * s["info14"].values)
            r_eb = offset_risk(d2_eb.values, s2, float(p["tau2_net"]), om2, rho)
            r_cv = offset_risk(d2_cov.values, s2, float(p["tau2_res_net"]), om2, rho)
            curves = {"EB": r_eb["shrink"], "Covariate EB": r_cv["shrink"]}
            if n > 0:
                curves["Local intercept"] = r_eb["local"]
            for m, r in curves.items():
                rows.append(dict(hospital_set=setname, n=n, hospitals=int(k.sum()), method=m,
                                 pred_mean_sq_ln_oe=float(np.mean(kappa[k] ** 2 * r[k] + noise.values[k])),
                                 pred_offset_mse=float(np.mean(r[k])), mean_eval_noise=float(np.mean(noise.values[k]))))
    return pd.DataFrame(rows)


def compare(pred, observed):
    m = pred.merge(observed[["hospital_set", "n", "method", "mean_sq_ln_oe", "mean_eval_noise"]]
                   .rename(columns={"mean_eval_noise": "obs_eval_noise"}), on=["hospital_set", "n", "method"], how="left")
    return m


def crossover_conditions(t2, s2):
    """Shrinkage beats site-only fitting iff d^2 < 2 t2 + s2; it beats the prior alone iff d^2 > t2 s2 / (t2 + 2 s2)."""
    return dict(beats_local_if_d2_below=2 * t2 + s2, beats_prior_if_d2_above=t2 * s2 / (t2 + 2 * s2))


if __name__ == "__main__":
    import sys
    d = sys.argv[1]
    sites = pd.read_csv(f"{d}/H_theory_sites.csv"); params = pd.read_csv(f"{d}/H_theory_params.csv")
    obs = pd.read_csv(f"{d}/G_label_budget.csv")
    for rho in (1.0, None):
        c = compare(predicted_curves(sites, params, rho=rho), obs)
        print("rho", rho); print(c.round(4).to_string())
        print("corr", np.corrcoef(c.pred_mean_sq_ln_oe, c.mean_sq_ln_oe)[0, 1],
              "median rel", np.median(np.abs(c.pred_mean_sq_ln_oe / c.mean_sq_ln_oe - 1)))
