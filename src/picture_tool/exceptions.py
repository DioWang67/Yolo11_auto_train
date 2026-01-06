"""
Custom exception classes for picture_tool package.

This module provides a hierarchy of exception classes to handle
different error scenarios in a structured way.
"""


class PictureToolError(Exception):
    """Base exception for all picture_tool errors."""

    pass


class ConfigurationError(PictureToolError):
    """Raised when configuration is invalid or missing."""

    pass


class DependencyError(PictureToolError):
    """Raised when required dependencies are not available."""

    pass


class ValidationError(PictureToolError):
    """Raised when input validation fails."""

    pass


class DatasetError(PictureToolError):
    """Raised when dataset is invalid or corrupted."""

    pass


class ModelError(PictureToolError):
    """Raised when model loading or inference fails."""

    pass


class ExportError(PictureToolError):
    """Raised when export operations fail."""

    pass


class PipelineError(PictureToolError):
    """Raised when pipeline execution fails."""

    pass
