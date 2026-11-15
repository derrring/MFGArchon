"""The installed version of mfgarchon: one owner, importing nothing but ``importlib.metadata`` (#2938)."""

from importlib.metadata import PackageNotFoundError, version


def _installed_version() -> str:
    try:
        return version("mfgarchon")
    except PackageNotFoundError:
        return "0.0.0-dev"  # not installed: running from a source tree


MFGARCHON_VERSION: str = _installed_version()
