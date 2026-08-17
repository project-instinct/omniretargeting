# Round 0 Contract

## What this round commits to

Remove the base height bistability introduced on this branch, without giving up
the foot orientation and contact quality already achieved.

Concretely, in this round:

- extend the diagnostic so base height continuity is measurable at all, since the
  defect escaped notice because the existing smoothness numbers covered joint
  angles and not the floating base (task1, AC-1)
- record the two baselines it needs to be judged against (task2, AC-1, AC-4)
- separate source skeleton scaling from scene scaling, so the unscaled-scene path
  can bring source proportions to the robot without touching the scene
  (task4, AC-2)
- derive that scale from measured pelvis-to-ankle lengths rather than declaring it
  (task5, AC-5)
- pin the rescaling so the contacting foot lands on the scene anchor, carrying the
  anchor across gaps (task6, AC-1, AC-2)
- verify continuity, foot quality and whole-body smoothness (task7-9)

## What this round does not commit to

- task10, the scale-invariance metamorphic test, and task11, the four-robot
  matrix, are queued rather than deferred. They validate generality, and the
  mechanism has to reach its own acceptance threshold on the reference clip
  before spending a matrix run on it.
- task12, decoupling scene scaling from the scaled-terrain CLI output path, is
  queued: it is real housekeeping but it is not on the path to this defect.

## How it will be judged

AC-1's numeric gate is the one that matters: the base height difference between
contact-active and contact-free frames under three centimetres. Everything else
is a guard against fixing that by breaking something else.
