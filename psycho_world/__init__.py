"""
Psycho-World package initialization.

This module exposes high-level factories for constructing the POMDP components
defined in the accompanying research outline.
"""

from importlib import metadata


def get_version() -> str:
    """
    Return the installed package version.
    """
    try:
        return metadata.version("psycho_world")
    except metadata.PackageNotFoundError:
        return "0.1.0-dev"


__all__ = ["get_version"]
