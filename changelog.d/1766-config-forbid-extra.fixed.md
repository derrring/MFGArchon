Configuration models reject unknown fields instead of dropping them (#1766). Pydantic ignores
extras by default, so `PicardConfig(anderson_acceleration=True)` constructed cleanly, discarded
the field, and left `anderson_memory` at 0 — Anderson off while the caller had just asked for it.
`BaseConfig` was a bare alias for Pydantic's `BaseModel`, exported but inherited by nothing; it is
now a real base owning `extra="forbid"`, and the 24 config models inherit it. Deprecated aliases
are unaffected: they are translated by a `mode="before"` validator before the check runs, and the
alias map is a class attribute (`PicardConfig.LEGACY_FIELD_ALIASES`) rather than a local inside
that validator, so the bridge can honour it too. The OmegaConf bridge drops top-level keys with no
matching field — interpolation anchors are a legitimate idiom there — but warns which, since a
silent drop at a transport boundary is how a real typo would disappear; nested keys still reach
their own model and still raise. The bridge's allow-set unions the field names with that alias
map: filtering on fields alone dropped a top-level `damping_factor` before the validator could see
it, reverting the value to its default and reporting a documented alias as a typo.
