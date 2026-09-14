# V1 consistency audit and remediation

This record tracks the review of commit `84ae01c` and the corrective changes
made before freezing the next V1 revision.

| Finding | Verified | Resolution |
|---|---:|---|
| Detector-PD proposed method inherited a reference selector's link budget | Yes | Removed. `proposed_c2f_adaptive_pd` and the full-refinement detector-PD control now choose their own set using the configured positive-marginal rule, price, and safety caps. |
| Paper objective differed from the runtime objective | Yes | Unified on the detector-predicted weak-target utility minus a per-remote-report price. The paper and runtime now use the same post-channel moments. |
| Swerling-I label conflicted with independent Gamma looks | Yes | Renamed the local detector to an independent fast-fluctuation, Swerling-II-like model. The existing Gamma likelihood is retained. |
| Sensing candidates were restricted by the inter-UAV reporting adjacency | Yes | Removed the `edge_mask[i,j]` requirement from sensing-table construction and candidate generation. Communication range now constrains only `j -> f_q`. |
| Observations produced at the fusion UAV were discarded | Yes | Local evidence is feasible with success probability one, zero payload, and zero reporting delay. It is excluded from MAC slots and transmitter interference. |
| Nearest-target fusion was described too strongly | Yes | Retained as a prediction-only heuristic. It is not presented as a jointly optimal fusion/selection solution. |
| Fixed blocklength was described as a link-dependent delay price | Yes | Renamed to a per-remote-report communication price. Link-dependent latency requires adaptive blocklength and remains future work. |
| `D_min=3` was far below the working deflection scale | Yes | The canonical fairness deficit now uses `P_D^req=0.95`; utility is saturated at the declared design point. `D_min` remains only for legacy reproduction. |
| No full-refinement detector-aligned control | Yes | Added `proposed_c2f_full_pd`, which evaluates every feasible candidate with fine DD refinement and uses the same detector-PD greedy rule. |
| No CI workflow | Yes | Added GitHub Actions jobs for unit tests, a target-local smoke run, and the checked-in release gate. |

The waveform-level OTFS check and the MC=100 diagnostic sweeps remain scope
boundaries, not hidden validation claims. The main comparison is rerun at
MC=1000 after every model-contract change; diagnostic controls retain their
sample size in labels and captions.

## Metric contract

`selected_observations_mean` counts sensing triplets admitted to fusion.
`selected_links_mean` is retained for CSV compatibility but now counts only
remote reports. Consequently,

```text
B_mean_bits = selected_links_mean * 640
T_mean_ms   = selected_links_mean * 2048 / 1.92e6 * 1e3
```

under the canonical serial fixed-blocklength MAC. Local observations contribute
to detection but not to either communication quantity.
