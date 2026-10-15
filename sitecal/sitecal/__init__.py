"""sitecal: site-level calibration audit and hierarchical recalibration for models deployed at many sites."""
from .audit import chance_ceiling, funnel_outside, per_site, site_audit
from .core import em_label_shift, fit_logistic_recal, fit_offset, meta_regression, random_effects
from .recalibrate import HierarchicalRecalibrator, site_summary
from .theory import break_even_labels, information_per_label, offset_risks, shrinkage_wins

__version__ = "0.1.0"
__all__ = ["site_audit", "per_site", "chance_ceiling", "funnel_outside", "HierarchicalRecalibrator", "site_summary",
           "em_label_shift", "fit_offset", "fit_logistic_recal", "random_effects", "meta_regression",
           "offset_risks", "break_even_labels", "information_per_label", "shrinkage_wins"]
