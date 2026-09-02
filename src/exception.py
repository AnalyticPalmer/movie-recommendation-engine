"""Project-specific exception types."""


class RecommendationEngineError(Exception):
    """Base exception for expected recommendation engine failures."""


class ConfigurationError(RecommendationEngineError):
    """Raised when project configuration is invalid or incomplete."""
