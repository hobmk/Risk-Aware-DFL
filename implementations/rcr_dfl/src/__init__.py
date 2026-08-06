from .capm import CAPMResult, fit_capm
from .dataset import RCRRollingMVODataset, chronological_split
from .effective_covariance import build_effective_covariance, project_to_psd
from .losses import (
    RCRLossOutput,
    combined_loss,
    compute_rcr_losses,
    markowitz_cost,
    rcr_regret,
)
from .model import ReturnMLP
from .residual_risk import (
    CorrelationScaling,
    MatrixDiagnostics,
    correlation_matrix,
    covariance_matrix,
    matrix_diagnostics,
    normalize_covariance_trace,
    scale_correlation_to_covariance,
    shrink_correlation,
)

__all__ = [
    "CAPMResult",
    "fit_capm",
    "RCRRollingMVODataset",
    "chronological_split",
    "CorrelationScaling",
    "MatrixDiagnostics",
    "covariance_matrix",
    "correlation_matrix",
    "shrink_correlation",
    "scale_correlation_to_covariance",
    "normalize_covariance_trace",
    "matrix_diagnostics",
    "build_effective_covariance",
    "project_to_psd",
    "RCRLossOutput",
    "markowitz_cost",
    "rcr_regret",
    "combined_loss",
    "compute_rcr_losses",
    "ReturnMLP",
]
