"""Hierarchical (empirical-Bayes) post-hoc recalibration of a frozen model across many sites."""
from __future__ import annotations

import numpy as np

from .core import em_label_shift, expit, fit_logistic_recal, fit_offset, logit, meta_regression, random_effects


def site_summary(p, y, global_ab=(0.0, 1.0)):
    """The only quantity a site has to share with the network: its logit offset and standard error
    relative to the (globally recalibrated) model."""
    a, b = global_ab
    return fit_offset(expit(a + b * logit(p)), y)


class HierarchicalRecalibrator:
    """Shrink site-specific logit offsets toward a network prior.

    Parameters
    ----------
    global_step : bool
        Fit a global logistic recalibration on all labeled network data first (Algorithm 1, line 1).
    use_covariates : bool
        Covariate-informed prior (Fay-Herriot): the prior mean of a site's offset is Z @ gamma rather than one
        network mean. Requires a site-descriptor matrix in `fit` and `predict`.
    semi_supervised : bool
        Also use an unlabeled EM label-shift estimate of each site's offset, treated as a biased measurement
        whose bias variance is estimated across the network.

    Usage
    -----
    >>> rc = HierarchicalRecalibrator().fit(scores, y, site)            # network update data
    >>> p_new = rc.predict(scores_new, site_new)                       # sites seen in fit
    >>> p_new = rc.predict(scores_new, site_new, local=(p_loc, y_loc))  # a new site with a few labels
    """

    def __init__(self, global_step=True, use_covariates=False, semi_supervised=False):
        self.global_step = global_step
        self.use_covariates = use_covariates
        self.semi_supervised = semi_supervised

    # ------------------------------------------------------------------ fitting
    def fit(self, p, y, site, Z=None, unlabeled=None):
        """p, y, site: labeled update data of the network. Z: dict site -> descriptor vector (no intercept).
        unlabeled: dict site -> array of unlabeled scores (for the semi-supervised variant)."""
        p, y, site = np.asarray(p, float), np.asarray(y, float), np.asarray(site)
        self.ab_ = fit_logistic_recal(p, y) if self.global_step else (0.0, 1.0)
        self.sites_ = list(dict.fromkeys(site.tolist()))
        a, s = np.zeros(len(self.sites_)), np.zeros(len(self.sites_))
        for i, h in enumerate(self.sites_):
            m = site == h
            a[i], s[i] = site_summary(p[m], y[m], self.ab_)
        self.offsets_ = dict(zip(self.sites_, a))
        self.ses_ = dict(zip(self.sites_, s))
        re = random_effects(a, s, hksj=False)
        self.mu_, self.tau2_ = re["mu"], re["tau2"]
        if self.use_covariates:
            if Z is None:
                raise ValueError("use_covariates=True needs Z")
            X = np.column_stack([np.ones(len(self.sites_)), np.array([np.asarray(Z[h], float) for h in self.sites_])])
            self.gamma_, self.tau2_res_ = meta_regression(a, s, X)
        self.pi_net_ = float(y.mean())
        if self.semi_supervised:
            if unlabeled is None:
                raise ValueError("semi_supervised=True needs unlabeled scores")
            e = np.array([self._em(unlabeled[h]) for h in self.sites_])
            self.sigma_b2_ = max(0.0, float(np.var(e - a, ddof=1) - np.mean(s ** 2)))
        return self

    def _em(self, scores):
        q = expit(self.ab_[0] + self.ab_[1] * logit(scores))
        return em_label_shift(q, self.pi_net_)[1]

    def _prior_mean(self, h, z=None):
        if not self.use_covariates:
            return self.mu_, self.tau2_
        return float(np.r_[1.0, np.asarray(z, float)] @ self.gamma_), self.tau2_res_

    def site_offset(self, h, z=None, local=None, unlabeled=None):
        """Posterior mean offset of site h. `local` = (scores, labels) of the site; for sites seen in fit the
        stored offset is used. `unlabeled` = unlabeled scores of the site (semi-supervised variant)."""
        m, t2 = self._prior_mean(h, z)
        t2 = max(t2, 1e-8)
        prec, num = 1 / t2, m / t2
        if local is not None:
            a, s = site_summary(local[0], local[1], self.ab_)
        elif h in self.offsets_:
            a, s = self.offsets_[h], self.ses_[h]
        else:
            a = s = None
        if a is not None:
            prec += 1 / s ** 2
            num += a / s ** 2
        if self.semi_supervised and unlabeled is not None:
            ve = self.sigma_b2_ + 1e-4
            prec += 1 / ve
            num += self._em(unlabeled) / ve
        return num / prec

    def predict(self, p, site, Z=None, local=None, unlabeled=None):
        """Recalibrated probabilities. Z, local and unlabeled may be dicts keyed by site."""
        p, site = np.asarray(p, float), np.asarray(site)
        out = np.empty(len(p))
        for h in np.unique(site):
            m = site == h
            off = self.site_offset(h, None if Z is None else Z.get(h), None if local is None else local.get(h),
                                   None if unlabeled is None else unlabeled.get(h))
            out[m] = expit(self.ab_[0] + self.ab_[1] * logit(p[m]) + off)
        return out
