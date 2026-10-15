import numpy as np

from sitecal import HierarchicalRecalibrator, em_label_shift, site_audit
from sitecal.core import expit, logit


def make(H=60, n=300, tau=0.3, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, H)
    d = 0.2 * z + rng.normal(0, tau, H)
    rows = []
    for h in range(H):
        x = rng.normal(-2.5, 1.2, n)
        y = rng.random(n) < expit(x + d[h])
        rows.append((np.full(n, h), expit(x), y.astype(float)))
    site = np.concatenate([r[0] for r in rows]); p = np.concatenate([r[1] for r in rows]); y = np.concatenate([r[2] for r in rows])
    return site, p, y, z, d


def test_audit_detects_heterogeneity():
    site, p, y, _, _ = make(tau=0.4)
    a = site_audit(y, p, site, min_n=100, min_events=5, n_sim=200)
    assert a["tau"] > 0.2
    assert a["pct_in_band"] < a["chance_ceiling"]
    assert a["pct_outside_95"] > a["expected_outside_95"]


def test_audit_null_model():
    rng = np.random.default_rng(1)
    site = np.repeat(np.arange(50), 400); p = expit(rng.normal(-2.5, 1.2, len(site)))
    y = (rng.random(len(site)) < p).astype(float)
    a = site_audit(y, p, site, min_n=100, min_events=5, n_sim=200)
    assert a["tau"] < 0.1


def test_shrinkage_beats_local_with_few_labels():
    site, p, y, z, d = make(H=80, n=400, tau=0.3, seed=2)
    rc = HierarchicalRecalibrator(use_covariates=True).fit(p, y, site, Z={h: [z[h]] for h in range(80)})
    rng = np.random.default_rng(3)
    err_eb, err_loc = [], []
    for h in range(80, 120):  # new sites with 25 labels
        zz = rng.normal(); dd = 0.2 * zz + rng.normal(0, 0.3)
        x = rng.normal(-2.5, 1.2, 25); yl = (rng.random(25) < expit(x + dd)).astype(float)
        off = rc.site_offset(h, z=[zz], local=(expit(x), yl))
        from sitecal.core import fit_offset
        loc = fit_offset(expit(x), yl)[0]
        err_eb.append((off + rc.ab_[0] - dd) ** 2); err_loc.append((loc - dd) ** 2)
    assert np.mean(err_eb) < np.mean(err_loc)


def test_em_label_shift_recovers_prior():
    rng = np.random.default_rng(4)
    n = 20000; y = rng.random(n) < 0.2
    x = np.where(y, rng.normal(1.5, 1, n), rng.normal(0, 1, n))
    llr = 1.5 * x - 1.5 ** 2 / 2
    q = expit(llr + logit(0.1))
    pi, _ = em_label_shift(q, 0.1)
    assert abs(pi - 0.2) < 0.02


def test_theory_shrinkage_never_worse_on_average():
    from sitecal import offset_risks, break_even_labels
    for tau2 in [0.01, 0.05, 0.2]:
        for s2 in [0.005, 0.05, 0.5]:
            r = offset_risks(tau2, s2)
            assert r["shrinkage"] <= min(r["prior"], r["site"]) + 1e-12
    assert abs(break_even_labels(0.1, 0.08) - 125) < 1e-9


def test_semi_supervised_runs_and_zero_event_site():
    site, p, y, z, d = make(H=40, n=200, tau=0.2, seed=5)
    unl = {h: p[site == h] for h in range(40)}
    rc = HierarchicalRecalibrator(use_covariates=True, semi_supervised=True).fit(p, y, site, Z={h: [z[h]] for h in range(40)}, unlabeled=unl)
    off = rc.site_offset("new", z=[0.0], local=(np.full(30, 0.05), np.zeros(30)), unlabeled=np.full(500, 0.05))
    assert np.isfinite(off)
