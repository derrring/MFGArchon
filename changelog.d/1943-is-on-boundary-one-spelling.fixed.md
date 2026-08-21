`is_on_boundary` accepts one spelling of its tolerance argument across every geometry (#1943).

`GeometryProtocol` declares `tolerance=`, and eighteen implementers use it. The `ImplicitDomain`
family used `tol=`. Positional calls worked either way, so **any keyword call broke on one side of
the split** — and which side was decided only by which geometry the caller happened to be handed:

```
Hyperrectangle.is_on_boundary(pts, tol=0.03)        -> [True, False]
Hyperrectangle.is_on_boundary(pts, tolerance=0.03)  -> TypeError
```

`ImplicitApplicator._detect_boundary_points` used the keyword form — an applicator that exists *for*
this family, calling it in the spelling this family did not accept.

The issue counts seven diverging classes; measured, six of them **inherit** the method, so there was
one definition to change. `tol=` remains as a deprecated alias, and passing both raises rather than
picking a winner — accepting both would introduce the class of defect this removes.

The census is **behavioural rather than a name count**: it asserts every implementer *accepts*
`tolerance`, walking `mfgarchon.geometry` so a new geometry joins by existing, and it skips classes
whose signature is `**kwargs` (a separate concern, #2020). A subclass re-overriding with `tol=` is
caught, which a census keyed on the base class would miss.

Mutation-checked: reverting the rename fails both the protocol test and the census. A control keeps
the positional call pinned, since a rename that broke it would be invisible to the keyword tests.
