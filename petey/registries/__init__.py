"""Registry data for petey's pluggable pipeline.

Each submodule holds the *data* for one registry kind — the dict literals
that map names to configs. The dispatch logic (how parsers are called,
how clients are built) stays in petey.extract; this package is just the
configuration surface.

Add a new entry by mutating the dict at runtime, e.g.::

    from petey.registries.parsers import PLUGIN_PARSERS
    PLUGIN_PARSERS["my_parser"] = "my_pkg.pdf:extract_pages"

The legacy import paths (``from petey.extract import API_PARSERS`` etc.)
continue to work — extract.py re-exports these names.
"""
