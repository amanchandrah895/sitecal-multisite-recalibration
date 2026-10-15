"""Simulation study with known truth for site-level recalibration.

Each replicate creates H sites. A frozen model outputs the logit x of a patient. The true risk at site h is
expit(b_h * x + d_h) with site offset d_h = gamma * z_h + u_h, u_h ~ N(0, tau^2), slope b_h ~ N(1, tau_b^2),
and z_h an observed site descriptor. Two shift mechanisms:
  * 'covariate': sites also differ in case mix (mean of x), so prevalence differences are partly explained by the model;
  * 'label':     pure prior-probability shift (x | y fixed across sites; only the prevalence changes).
Each site has n labeled update cases (and plentiful unlabeled scores); calibration is evaluated on a large test
sample using the noise-free error ln(sum p_true / sum p_pred) of each site.
Usage: python3 simulation.py <out_dir> [covariate|label|aggregate|tau|all]   (SIM_R env var = replicates; paper: 50)
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))  # ../code holds transport_utils.py
import transport_utils as tu  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "sim_results"
os.makedirs(OUT, exist_ok=True)
lg = lambda p: np.log(p / (1 - p))


def em_prior(q, pi_s, iters=300):
    q = np.clip(q, 1e-6, 1 - 1e-6); pi = pi_s
    for _ in range(iters):
        w1 = q * pi / pi_s; w0 = (1 - q) * (1 - pi) / (1 - pi_s)
        new = float(np.mean(w1 / (w1 + w0)))
        if abs(new - pi) < 1e-9:
            break
        pi = new
    return pi


def meta_reg(a, s, Z):
    v = s ** 2; W = np.diag(1 / v)
    P = W - W @ Z @ np.linalg.pinv(Z.T @ W @ Z) @ Z.T @ W
    t2 = max(0.0, (a @ P @ a - (len(a) - Z.shape[1])) / np.trace(P))
    W2 = np.diag(1 / (v + t2)); g = np.linalg.pinv(Z.T @ W2 @ Z) @ Z.T @ W2 @ a
    return g, t2


def one_rep(rng, scen, H, n, tau, gamma=0.15, tau_b=0.08, n_unl=1000, n_test=3000, delta0=-0.1, n_net=400):
    """Sites 0..H-1 form the network (n_net labels each; they supply the global recalibration and the prior);
    sites H..2H-1 are the target sites (n labels each), on which every strategy is evaluated."""
    HN = H; H = 2 * H
    z = rng.normal(0, 1, H)
    d = delta0 + gamma * z + rng.normal(0, tau, H)
    if scen == "covariate":
        b = 1 + rng.normal(0, tau_b, H)
        m_h = rng.normal(-2.6, 0.4, H)
        draw_x = lambda h, k: rng.normal(m_h[h], 1.3, k)
        true_logit = lambda h, x: b[h] * x + d[h]
        # source: m = -2.6, d = 0, b = 1
        xs = rng.normal(-2.6, 1.3, 200000); pi_s = float(tu.expit(xs).mean())
    else:  # pure label shift: x | y ~ N(mu_y, 1); model calibrated at source prevalence pi_s
        pi_s = 0.09; mu0, mu1 = 0.0, 1.6
        llr = lambda x: (mu1 - mu0) * x - (mu1 ** 2 - mu0 ** 2) / 2
        pi_h = tu.expit(lg(pi_s) + d)

        def draw_xy(h, k):
            y = rng.random(k) < pi_h[h]
            x_ = np.where(y, rng.normal(mu1, 1, k), rng.normal(mu0, 1, k))
            return llr(x_) + lg(pi_s), y.astype(float)
        true_logit = lambda h, x: x + d[h]
    U, T, UL = [], [], []
    for h in range(H):
        nn = n_net if h < HN else n
        if scen == "covariate":
            xu = draw_x(h, nn); yu = (rng.random(nn) < tu.expit(true_logit(h, xu))).astype(float)
            xt = draw_x(h, n_test); xl = draw_x(h, n_unl)
        else:
            xu, yu = draw_xy(h, nn); xt, _ = draw_xy(h, n_test); xl, _ = draw_xy(h, n_unl)
        U.append((xu, yu)); T.append((xt, tu.expit(true_logit(h, xt)))); UL.append(xl)
    # --- strategies (return per-site test logit offsets/maps)
    xall = np.concatenate([U[h][0] for h in range(HN)]); yall = np.concatenate([U[h][1] for h in range(HN)])
    ag, bg = tu.fit_logistic(tu.expit(xall), yall)
    if not np.isfinite(ag):
        ag, bg = 0.0, 1.0
    pi_net = float(yall.mean())
    ah, sh, eh = np.zeros(H), np.zeros(H), np.zeros(H)
    loc = np.zeros(H); prev = np.zeros(H); em_raw = np.zeros(H)
    for h in range(H):
        xu, yu = U[h]
        q = tu.apply_logistic(tu.expit(xu), ag, bg)
        ah[h], sh[h] = tu.fit_intercept(q, yu)
        loc[h] = tu.fit_intercept(tu.expit(xu), yu)[0]
        rate = np.clip(yu.mean(), 0.5 / len(yu), 1 - 0.5 / len(yu)); prev[h] = lg(rate) - lg(pi_s)
        ql = tu.apply_logistic(tu.expit(UL[h]), ag, bg)
        eh[h] = lg(em_prior(ql, pi_net)) - lg(pi_net)
        em_raw[h] = lg(em_prior(tu.expit(UL[h]), pi_s)) - lg(pi_s)
    net = slice(0, HN)
    ma = tu.meta_analysis(ah[net], sh[net], hksj=False); mu, t2 = ma["mu"], max(ma["tau2"], 1e-8)
    Z = np.column_stack([np.ones(H), z]); g, t2r = meta_reg(ah[net], sh[net], Z[net]); t2r = max(t2r, 1e-8); mz = Z @ g
    eb = mu + t2 / (t2 + sh ** 2) * (ah - mu)
    cov = mz + t2r / (t2r + sh ** 2) * (ah - mz)
    sb2 = max(0.0, np.var(eh[net] - ah[net], ddof=1) - np.mean(sh[net] ** 2))   # EM-bias variance from the network
    ve = sb2 + 1e-4
    semi = (mz / t2r + ah / sh ** 2 + eh / ve) / (1 / t2r + 1 / sh ** 2 + 1 / ve)
    res = {}
    for name in ["No updating", "Global logistic", "Site-only intercept", "EB", "Covariate EB", "Semi-supervised EB",
                 "EM label shift (no labels)", "Prevalence correction"]:
        err = np.zeros(H - HN)
        for i, h in enumerate(range(HN, H)):
            xt, pt = T[h]
            if name == "No updating":
                lp = xt
            elif name == "Global logistic":
                lp = ag + bg * xt
            elif name == "Site-only intercept":
                lp = xt + loc[h]
            elif name == "EB":
                lp = ag + bg * xt + eb[h]
            elif name == "Covariate EB":
                lp = ag + bg * xt + cov[h]
            elif name == "Semi-supervised EB":
                lp = ag + bg * xt + semi[h]
            elif name == "EM label shift (no labels)":
                lp = xt + em_raw[h]
            else:
                lp = xt + prev[h]
            err[i] = np.log(pt.sum() / tu.expit(lp).sum())
        res[name] = err
    true_sd = float(np.std(res["No updating"], ddof=1))
    return res, true_sd


def run_grid(scen, taus, ns, R, H=100, seed=0):
    rng = np.random.default_rng(seed); rows = []
    for tau in taus:
        for n in ns:
            for r in range(R):
                res, _ = one_rep(rng, scen, H, n, tau)
                for k, err in res.items():
                    rows.append(dict(scenario=scen, tau=tau, n=n, rep=r, method=k, mse=float(np.mean(err ** 2)),
                                     median_abs=float(np.median(np.abs(err))),
                                     pct_in_band=100 * float(np.mean(np.abs(err) <= np.log(1.2))),
                                     pct_extreme=100 * float(np.mean(np.abs(err) > np.log(3)))))
    return pd.DataFrame(rows)


def tau_check(R=200, H=97, seed=1):
    """Does DerSimonian-Laird tau of ln(O/E), estimated from sampled outcomes in sites of realistic size,
    recover the true between-site SD of the calibration error? (covariate scenario, no updating)"""
    rng = np.random.default_rng(seed); rows = []
    for tau in [0.0, 0.1, 0.2, 0.3]:
        for r in range(R // 4):
            z = rng.normal(0, 1, H); d = -0.1 + 0.15 * z + rng.normal(0, tau, H); m_h = rng.normal(-2.6, 0.4, H)
            sizes = rng.integers(100, 1500, H); lo, se, true = [], [], []
            for h in range(H):
                x = rng.normal(m_h[h], 1.3, sizes[h]); p = tu.expit(x + d[h]); y = rng.random(sizes[h]) < p
                O, E = y.sum(), tu.expit(x).sum()
                if O < 1:
                    continue
                lo.append(np.log(O / E)); se.append(np.sqrt((1 - O / sizes[h]) / O)); true.append(np.log(p.sum() / E))
            ma = tu.meta_analysis(np.array(lo), np.array(se), hksj=False)
            rows.append(dict(tau_offset=tau, true_sd_ln_oe=float(np.std(true, ddof=1)), dl_tau=ma["tau"]))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    t0 = time.time()
    taus = [0.05, 0.1, 0.2, 0.3, 0.4]; ns = [25, 50, 100, 200, 400, 800]
    R = int(os.environ.get("SIM_R", "20"))
    part = sys.argv[2] if len(sys.argv) > 2 else "all"
    if part in ("covariate", "label"):
        g = run_grid(part, taus, ns, R, seed=11 if part == "covariate" else 12)
        g.to_csv(os.path.join(OUT, f"sim_grid_raw_{part}.csv"), index=False)
    if part in ("all", "aggregate"):
        if part == "all":
            grid = pd.concat([run_grid("covariate", taus, ns, R, seed=11), run_grid("label", taus, ns, R, seed=12)])
        else:
            grid = pd.concat([pd.read_csv(os.path.join(OUT, f"sim_grid_raw_{p}.csv")) for p in ("covariate", "label")])
        grid.to_csv(os.path.join(OUT, "sim_grid_raw.csv"), index=False)
        agg = grid.groupby(["scenario", "tau", "n", "method"]).agg(
            reps=("mse", "size"), mse=("mse", "mean"), mse_se=("mse", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
            median_abs=("median_abs", "mean"), pct_in_band=("pct_in_band", "mean"), pct_extreme=("pct_extreme", "mean")).reset_index()
        agg.to_csv(os.path.join(OUT, "sim_grid.csv"), index=False)
    if part in ("tau", "all"):
        tc = tau_check()
        tc.to_csv(os.path.join(OUT, "sim_tau_check_raw.csv"), index=False)
        tc.groupby("tau_offset").agg(true_sd=("true_sd_ln_oe", "mean"), dl_tau=("dl_tau", "mean"),
                                     dl_tau_sd=("dl_tau", "std")).reset_index().to_csv(os.path.join(OUT, "sim_tau_check.csv"), index=False)
    print(part, "done in", round(time.time() - t0, 1), "s")
