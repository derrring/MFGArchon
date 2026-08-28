- **The ruff release check authenticates, and refuses a reply that is not a version** (Issue
  #2134). `check-ruff-updates.yml` fetched the latest tag from `api.github.com` unauthenticated,
  where the limit is 60 requests/hour per IP and GitHub-hosted runners share addresses. A
  rate-limited body carries no `tag_name`, so `jq -r` printed the string `null`, which compares
  unequal to the pinned version — the workflow concluded an update was available and the next step
  ran `pip install ruff==null`, monthly, on a schedule with no reader. The request now sends
  `GITHUB_TOKEN` and carries `--max-time`, and the reply must match `^[0-9]+\.[0-9]+\.[0-9]+$`
  before it reaches `$GITHUB_OUTPUT`.
- **`-e` was never going to catch this.** A `run:` block with no `shell:` key runs under
  `bash -e {0}`, without `pipefail`, so the exit status of `curl … | jq … | sed …` is `sed`'s and
  is 0 however badly `curl` failed. The shape check is the only thing between the API and the rest
  of the job.
- **Two narrower holes in the same line.** `sed 's/v//'` was unanchored and ate the first `v`
  anywhere, turning a `v0.14.0-preview` tag into `0.14.0-preiew`; and the first shape check was a
  `case` glob, `[0-9]*.[0-9]*.[0-9]*`, where `*` matches dots — it admitted `0.17.0rc1`,
  `1.2.3.4.5.6` and `2026.8.28`. Now `s/^v//` and a bash regex.
- The test runs the step under a real `bash` with `curl` stubbed, rather than locating the check by
  regex and inspecting it. The inspecting version passed when the check was moved *after*
  `$GITHUB_OUTPUT`, went red when the check was replaced by a stricter one, and raised `KeyError`
  when any `uses:`-only job was added to the file.
