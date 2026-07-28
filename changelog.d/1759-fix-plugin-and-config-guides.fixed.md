The plugin-development and configuration-system guides taught workflows that could not run
(#1759). `plugin_development.md` imported `SolverResult` from a module that does not exist and
passed five field names the real signature does not have; `configuration_system.md` documented
`mfgarchon.config.structured_schemas` and thirteen `*Schema` classes, removed in v0.19.4 when
the config system moved to one Pydantic schema authority. Both rewritten against what exists,
with the removal and its commit named so a reader is not left guessing whether the module was
renamed. Doc-API baseline 250 -> 218.
