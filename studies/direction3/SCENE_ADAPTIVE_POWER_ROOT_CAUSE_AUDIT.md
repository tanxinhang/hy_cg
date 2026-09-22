# Root-cause audit: rejected scene-adaptive power allocation

## Controlled comparison

Both arms use the same frozen 600 m scene, 25 looks, TP-UIC receiver,
communication blocklength, report limits, 4.8 W fleet budget, and 1 W per-UAV
cap. Only the power vector changes.

| Quantity | Incumbent | Geometry max-min |
|---|---:|---:|
| Worst delivered PD | 0.6066 | 0.4938 |
| Target-3 best local PD | 0.4024 | 0.4457 |
| Target-3 cooperation gain | 0.2042 | 0.0481 |
| Target-3 summed receiver information | 6.2270 | 4.0538 |
| Selected reports | 4 | 2 |
| Active transmitters | 3 | 4 |

## Root cause

The geometry allocator solves two separate surrogates. Sensing power is scored
by normalized geometric target gain, while communication power is scored by
each sender's single best reachable direct link. It does not include receiver
evidence, selected target, fusion destination, TP-UIC residual covariance, or
delivered PD.

Consequently it assigns nearly zero communication power to UAVs 0, 1, 4, and
5. Three of those UAVs (0, 1, and 4) carried useful reports in the incumbent.
The new scheduler can retain only two reports, even though their success
probabilities are high. Target 3 gets a somewhat stronger best local receiver,
but loses most distributed evidence.

The sensing surrogate also changes the active set from `{2,4,5}` to
`{0,2,4,5}`. It raises summed information for targets 1 and 2 but reduces the
target-3 receiver-information sum by about 35%. This is the opposite of the
actual max-min need.

## Decision

The failure is an objective mismatch, not evidence that heterogeneous or
scene-adaptive power is ineffective. The geometry max-min policy is rejected
for joint sensing/communication claims. No new optimizer should be introduced
until candidate powers are evaluated by actual worst-target delivered PD and
the communication schedule is recomputed for each candidate.

## Rejected shortcut

An evidence-times-link linear utility was prototyped during the audit. It put
the entire 0.96 W communication budget on one UAV because a linear reward has
no finite-blocklength saturation or diminishing return. The shortcut was
removed before promotion. A universal positive communication floor is also
not justified: zero communication power is valid for a UAV that is not a
reporter. Starving a useful reporter is already penalized by its FBL erasure
probability and the resulting delivered-PD calculation.

## Communication-floor experiment

A configurable per-UAV constraint `P_comm,j >= 0.05 W` was tested on the same
frozen scene. It removed near-zero communication powers, but retained only two
reports and reduced worst delivered PD further from 0.4938 to 0.4733. Target-3
local PD was 0.4278 and its cooperation gain only 0.0455, versus 0.2042 for the
incumbent. The floor consumed sensing power without restoring the specific
evidence-bearing routes used by the incumbent.

Decision: retain the floor as an optional feasibility constraint, defaulting to
zero, but reject `0.05 W for every UAV` as a performance policy. A useful second
layer must be conditional on the selected reporting set, not universal.
