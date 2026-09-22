# OMOMO prepared replay

The maintained optimizer is now the source-independent
[`retarget_sequence` backend](sequence_retargeting.md), also available through
`python -m omniretargeting.main --solver sequence`. It reads robot dimensions and
limits from MuJoCo and applies temporal penalties in seconds.

`scripts/omomo_sequence_trial.py` is a diagnostic data wrapper around that same
API. It loads the original cached source/contact payload and a recorded initial
trajectory, then writes to `out/omomo_trial_20260921/general_sequence/<clip>`:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python \
  scripts/omomo_sequence_trial.py --trial-dir out/omomo_trial_20260921 \
  --clip sub16_largebox_000
```

The prepared wrapper can use the pinned upstream initial trajectories stored in
`upstream_reference`; the normal CLI does not require those files. There is no
29-joint or 30-fps restriction in the wrapper or optimizer. `--source-cache`,
`--seed`, `--output-dir` and `--iterations` control replay inputs and output only.
The three original clips use the same solver parameters.

The previous G1-specific experiment and its accepted artifacts remain under
`upstream_sequence` as provenance. New results, source/geometry audits, jitter
comparisons and playback are written under `general_sequence`. A sequence
proposal must complete bounded collision correction and the final independent
artifact checks before replacing a published playback.
