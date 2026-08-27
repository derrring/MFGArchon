- **The ruff release check authenticates, and refuses a reply that is not a version** (Issue
  #2134). `check-ruff-updates.yml` fetched the latest tag from `api.github.com` unauthenticated,
  where the limit is 60 requests/hour per IP and GitHub-hosted runners share addresses. A
  rate-limited body carries no `tag_name`, so `jq -r` printed the string `null`, which compares
  unequal to the pinned version — the workflow concluded an update was available and the next step
  ran `pip install ruff==null`. The job failed monthly, on a schedule with no reader, naming pip
  rather than the API. The request now sends `GITHUB_TOKEN`, and the reply is checked for a version
  shape before use, so a token that expires or a limit that moves fails where the diagnostic can
  name the cause. `tests/unit/test_ruff_fetch_refuses_a_non_version_2134.py` runs the guard under
  bash against `null`, an empty reply, an unstripped `v0.16.0` and two API error strings.
