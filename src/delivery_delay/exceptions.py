"""Custom exceptions used by the application layer."""


class DeliveryDelayError(Exception):
    """Base error for project-specific failures."""


class DataLoadingError(DeliveryDelayError):
    """Raised when a tabular file cannot be loaded safely."""


class TargetDetectionError(DeliveryDelayError):
    """Raised when no usable prediction target can be found."""


class ModelTrainingError(DeliveryDelayError):
    """Raised when no candidate model can be trained successfully."""


class ArtifactError(DeliveryDelayError):
    """Raised when a saved model artifact is missing or invalid."""

