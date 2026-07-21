"""Issue #1540: mfgarchon.core.plugin_system must import and discover without pkg_resources.

pkg_resources was removed from setuptools >= 81, so the module-level `import pkg_resources` made
`import mfgarchon.core.plugin_system` raise ModuleNotFoundError on any modern setuptools — undetected
because nothing in the package imports the module. It now uses the stdlib importlib.metadata.
"""

from __future__ import annotations


def test_plugin_system_imports_without_pkg_resources():
    """The module must import (it did not on setuptools>=81) and no longer import pkg_resources."""
    import inspect

    import mfgarchon.core.plugin_system as plugin_system

    src = inspect.getsource(plugin_system)
    # The import statement and the pkg_resources API call must be gone (the word may still appear in
    # an explanatory comment, so match the load-bearing forms, not the bare token).
    assert "import pkg_resources" not in src
    assert "pkg_resources.iter_entry_points" not in src
    assert "importlib.metadata" in src


def test_discover_plugins_runs_and_returns_list():
    """PluginManager.discover_plugins() must run through the importlib.metadata entry-point path
    without raising (returns an empty list when no `mfgarchon.plugins` entry points are installed)."""
    from mfgarchon.core.plugin_system import PluginManager

    result = PluginManager().discover_plugins()
    assert isinstance(result, list)
