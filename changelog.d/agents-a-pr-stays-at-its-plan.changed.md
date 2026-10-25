- **AGENTS.md: a PR stays at its plan** (user ruling, 2026-10-25), a new subsection after § *Local patches are the
  anti-pattern*.
  - The approved plan fixes a PR's scope, as its class line fixes its population. A finding that is neither in the plan
    nor a member of the class goes through *Before filing an issue*, and the work continues on the plan; its author
    does not widen the PR on their own judgement. A member of the class is decided by § *Local patches*, and a blocker
    in the diff, including a defect the diff newly exposes, is fixed by the PR.
  - A finding is structural when it falsifies the plan's premise or acceptance check, concerns a convention or owner
    several implementations share, is a default-path wrong result present at the base sha in code the PR already
    changes, or changes a denominator of #2512's counts (`scripts/oracle_pins.txt`). Its author sends it to the plan's
    approver in one line and keeps working the plan. The approver rules whether the PR takes it or it goes through
    *Before filing an issue*. A PR that takes it records this in the issue comment that holds the plan's approval, and
    from review 1 on, its fix is a class fix with its own class line.
  - § *Branches & PRs*: related low-risk fixes are batched into one PR when the plan is made.
