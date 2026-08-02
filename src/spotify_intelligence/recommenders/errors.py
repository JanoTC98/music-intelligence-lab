from __future__ import annotations


class ArtifactNotFoundError(Exception):
    """Raised when a recommender artifact does not exist on disk."""


class IncompatibleArtifactError(Exception):
    """Raised when a recommender artifact is not compatible with the request."""
