# Opt-in video source denoising

`--source-denoise {off,auto}` defaults to `off`. This is an explicitly authorized,
controlled exception to source-motion immutability for video estimates. It
creates an in-memory reference; source files, scene registration, object poses,
scale, solver policy and collision audits are unchanged. Off follows the previous
numerical path. A provenance field is added to the NPZ even when off.

## Processing

SMPL-X raw AMASS, OMOMO and TRUMANS adapters filter parameters before SMPL-X FK,
so joints, rotations, soles, markers and skin share one reference. Raw translation,
root orientation and 21 local body rotations are treated separately. Shape and
hand/face parameters are preserved. Positions-only/processed SMPL-X inputs cannot
be denoised and raise an explicit error in auto mode. Positive source fps is
required; filtering precedes optional output resampling.

Video pipelines often publish an inference rate below the video rate by
repeating each estimate (HER: 30 Hz inference held for two 59 Hz rows). The
staircase is not motion, and it dominates every derivative (HER root
acceleration RMS ~68 m/s², almost all above 15 Hz). When `inference_frame_ids`
is present, or rows repeat root and body rotations exactly in short regular runs
(2–5 rows; long static runs do not qualify), each run is one sample placed at
its run centre. All channels of such a proven video estimate are then smoothed
at the sample rate and returned to every row with a C² spline (`RotationSpline`
for rotations), extrapolating at most half a run at the ends.

Without holds, a channel is filtered only when its third difference is
dominated by alternating white noise: robust lag-one correlation below -0.5
(i.i.d. noise measures about -0.67) and noise above 1 mm, 0.20° (root) or 0.35°
(body). Correlation alone cannot separate mocap (OMOMO/TRUMANS -0.02 to -0.63);
the floors do (mocap noise at most 0.2 mm and 0.27°). Clean mocap stays bit-identical.

The kernel is a Whittaker smoother with a third-difference penalty, response
`1/(1+λ(2 sin(πf/fps))^6)`: 36 dB/octave without the sidelobes of a local
polynomial, whose leakage left 15–30 Hz noise in acceleration. Generalized
cross-validation picks λ per channel from the data (cut-off 2–20 Hz), so the
amount of smoothing follows the actual noise. In one loop, Huber weights on
the jerk penalty release sparse large events (takeoff, landing, impacts) and
Tukey bisquare data weights reject samples that no smooth-with-sparse-jerk path
explains (e.g. HER single-sample contact-alignment z switches of ~0.6 m);
GCV is re-evaluated on the inliers. Rotations are smoothed as matrices and
projected back to SO(3) (chordal mean), continuous across ±π.

Regression tests keep clean 5 Hz motion, a synthetic backflip and static periods
exactly; recover a held 30 Hz signal at 60 Hz within 2 mm; reject a single 0.6 m
outlier sample without reporting a step; and in a noisy slow clip with one
ballistic flight keep the flight acceleration within 1.5 m/s² of -g and the apex
within 1 cm. They do not certify every real acrobatic clip.

Remaining root increments (after outlier pairs that return within three samples
are left to the bisquare weights) with a deviation above `max(0.12 m, 12 * median residual)` from
local motion are treated as isolated jumps. The estimated discontinuity is
replaced by a quintic transition spanning at least 0.24 s, longer for large jumps
(using a nominal 15 m/s² transition acceleration scale). Before/after positions
and timing away from the transition remain intact. Each repair records the frame,
vector, magnitude and transition endpoints; this is not a hard dynamics bound.

After FK, a conservative support heuristic selects low (within 35 mm of the
observed sole floor), slow (<0.16 m/s) fixed sole material patches, over >=0.15 s.
A patch is locked only if its entire correction stays below 25 mm; the blended
root correction is capped at 20 mm and ramped across interval boundaries. Two
feet share the correction, so this cannot certify two simultaneous exact locks.
The same shift updates translation, joints and vertices before surface sampling.
This heuristic is explicitly marked as physically unverified; it cannot infer
support on arbitrary elevated furniture or certify no foot slip. World foot
jitter/slip must be checked on resulting robot motions.

Only on an actually processed trajectory, source contact preparation debounces
intervals shorter than 0.1 s (ceil to frames). Gap merging remains within valid
same-surface geometry and the configured contact band; unknown geometry and real
departures are not bridged. Short touches are preserved when signed normal
velocity shows a resolved approach/rebound above 0.15 m/s. Other blips are removed.
The original geometric observations are retained. Accepted short gaps carry
separate touch modes/confidence; they cannot create indefinite holds from the
velocity gate or silently acquire a sticking anchor. The source's full-body table
carries the opt-in metadata; generic solver contracts do not depend on SMPL-X.

## Traceability and review

NPZ `source_denoise_json`, adjacent `<output>.denoise.json` and the fast report
contain mode/enabled/applied, reason, noise estimates, per-channel thresholds,
hold resampling, GCV cut-off, released/outlier counts, jump repairs, support corrections and contact merge/removal/impact
intervals. `source_joints_w` is the reference actually supplied to the solver;
compare auto motion against off prepared data to measure fidelity to the original.
TRUMANS release joints are retained exactly when no filtering is needed; when
filtering is applied, FK supplies the consistent modified reference.

`scripts/review_source_denoise.py --off-dir ... --auto-dir ... --output ...`
compares 0.2 s high-pass jitter, acceleration, world-foot jitter under the
same original contact mask, plus tangential slip of fixed collision material
points (sole centres alone conflate rolls with slip), source bone directions, contact counts and qpos
differences. Standard pose/contact/joint/pair/reset audits remain necessary.
Broadband video noise can overlap real motion frequencies; this heuristic has no
guarantee for every motion, impacts or reconstruction failure. Independent audit
residuals remain defects, not reasons to change audit thresholds or silently rerun.
