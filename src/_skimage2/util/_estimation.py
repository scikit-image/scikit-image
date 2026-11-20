"""Re-export estimation helpers, so lazy_loader can pick them up via the stub."""

from .._shared.utils import FailedEstimation, FailedEstimationAccessError

__all__ = ['FailedEstimation', 'FailedEstimationAccessError']
