"""Hospital-cluster bootstrap CIs for the label-budget comparison (NB13 subsamples) against no updating.
Usage: python3 label_budget_bootstrap.py <results_dir> <out_csv>  (results_dir = notebook outputs, credentialed users only)"""
import sys
import numpy as np
import pandas as pd

R = sys.argv[1]
lb = pd.read_csv(f"{R}/NB13/G_label_budget_hospital_reps.csv")
s0 = pd.read_csv(f"{R}/NB08/per_hospital_by_strategy.csv")
s0 = s0[s0["strategy"] == "S0_none"].set_index("hospital_id")["ln_oe"]
H = pd.read_csv(f"{R}/NB13/H_theory_sites.csv").set_index("hospital_id")
inband = lambda oe: ((oe >= 0.8) & (oe <= 1.2)).astype(float)
lb["inband"] = inband(lb["oe"])
# per hospital x n x method: mean over repeats
ph = lb.groupby(["n", "method", "hospital_id"]).agg(sq=("sq_ln_oe", "mean"), ab=("abs_ln_oe", "mean"), ib=("inband", "mean")).reset_index()
none = pd.DataFrame({"sq": s0 ** 2, "ab": s0.abs(), "ib": inband(np.exp(s0))})
rng = np.random.default_rng(2026); B = 2000
rows = []
for setname, hosp_fn in [("all", lambda n: H.index[H["n14"] >= n]), ("fixed35", lambda n: H.index[H["large"].astype(bool)])]:
    for n in [25, 50, 100, 200, 400]:
        hs = np.array([h for h in hosp_fn(n) if h in none.index])
        for m in ["EB", "Local intercept", "Covariate EB"]:
            d = ph[(ph["n"] == n) & (ph["method"] == m)].set_index("hospital_id").reindex(hs)
            ok = d["sq"].notna().values; hh = hs[ok]; d = d.loc[hh]; z = none.loc[hh]
            diff_sq = (d["sq"] - z["sq"]).values; diff_ib = 100 * (d["ib"] - z["ib"]).values
            bs_sq, bs_ib = [], []
            for _ in range(B):
                i = rng.integers(0, len(hh), len(hh))
                bs_sq.append(diff_sq[i].mean()); bs_ib.append(diff_ib[i].mean())
            rows.append(dict(hospital_set=setname, n=n, method=m, hospitals=len(hh),
                             d_mean_sq=diff_sq.mean(), d_mean_sq_lo=np.percentile(bs_sq, 2.5), d_mean_sq_hi=np.percentile(bs_sq, 97.5),
                             d_pct_band=diff_ib.mean(), d_pct_band_lo=np.percentile(bs_ib, 2.5), d_pct_band_hi=np.percentile(bs_ib, 97.5)))
out = pd.DataFrame(rows)
out.to_csv(sys.argv[2], index=False)
print(out.round(4).to_string())
