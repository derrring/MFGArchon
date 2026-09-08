The `still_refused` test fixture no longer accepts a free-form retirement message. A class-3 defect
pin fires on an ABSENCE -- the guard stopped raising -- and an absence has more causes than the fix
the pin demands, so a message naming one cause is wrong whenever a different cause produced it.
Measured three times in one session, each caught by an independent reviewer and none by the author:
once, replacing a helper's body with `return None` produced a byte-identical message declaring an
unrelated fix had landed.

The caller now states what was `observed` and enumerates `causes`; the fixture renders the message.
A single-cause message is refused unless the caller also passes `premise` (a check that excludes the
others, which the fixture runs in a `finally` inside the block so it fires on the raising and
non-raising paths alike) together with `premise_establishes` (the sentence saying what it proves, which the fixture
renders and cannot itself verify), or `excluded` (a sentence naming what already rules them out). The contract
is enforced whenever the block is entered, not when the pin eventually retires.

What it does not do, measured rather than assumed: nothing here finds a MISSING cause, and an
earlier draft that also checked the wording of `observed` was dropped because every evasion tried
walked past it and it caught nothing the structural rule did not.
