"""Closed-form risk of site-offset estimators (Proposition 1 of the paper)."""
from __future__ import annotations

import numpy as np


def offset_risks(tau2, s2, d2=None, omega2=0.0):
    """Mean squared error of the prior alone, the site-only estimate and the shrinkage estimate of a site offset.

    tau2: between-site variance of offsets around the prior mean; s2: sampling variance of the site's own estimate;
    d2: squared deviation of this site from the prior mean (defaults to tau2, the average site); omega2: drift variance
    between the update and the deployment period."""
    d2 = tau2 if d2 is None else d2
    B = tau2 / (tau2 + s2)
    return dict(prior=d2 + omega2, site=s2 + omega2, shrinkage=(1 - B) ** 2 * d2 + B ** 2 * s2 + omega2, B=B)


def break_even_labels(tau2, info):
    """Number of labels at which a site-only offset stops losing to the prior: s^2 = 1/(n * info) = tau2."""
    return 1.0 / (tau2 * info)


def information_per_label(p):
    """Fisher information per labeled case for a logit offset: mean p(1-p) of the site's predicted risks."""
    p = np.asarray(p, float)
    return float(np.mean(p * (1 - p)))


def shrinkage_wins(tau2, s2, d2):
    """(beats site-only fitting, beats the prior alone) for a site with squared deviation d2."""
    return d2 < 2 * tau2 + s2, d2 > tau2 * s2 / (tau2 + 2 * s2)
