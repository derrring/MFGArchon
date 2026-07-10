- **Fail-fast ratchet baseline tightened.** `scripts/fail_fast_baseline.json` re-baselined to current
  counts (hasattr 172->164, silent_pass 70->60) so the monotone CI ratchet bites again; it had gone
  stale after the v0.21.0 fail-loud fixes reduced live counts below the recorded ceiling.
