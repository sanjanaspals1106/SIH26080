"""Errors raised by Stage 1 ingestion. Every one carries a message that says what to do next."""

from __future__ import annotations


class IngestionError(Exception):
    """Base class for all ingestion errors."""


class MissingInputError(IngestionError):
    """A file or folder that ingestion needs does not exist (or is empty)."""


class MissingCredentialsError(IngestionError):
    """The ECDS key is not configured. Secrets are never read from source code or committed files."""


class DownloadError(IngestionError):
    """A download failed or produced no file."""


class ValidationError(IngestionError):
    """Data was read but does not match what the PRD expects (variables, grid, dates, geometry)."""
