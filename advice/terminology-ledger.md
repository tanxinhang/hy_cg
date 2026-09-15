# Terminology ledger

| Canonical term | First-use definition | Avoid |
|---|---|---|
| OTFS-ISAC | orthogonal time frequency space integrated sensing and communication | alternating OTFS/ISAC spellings |
| cooperative observation | target-dependent bistatic tuple `(i,j,q)` | calling it only a communication link |
| reporting leg | directed soft-report link `j -> f_q` | reusing the sensing pair `i -> j` |
| fusion UAV | target-specific fusion UAV `f_q` | fusion centre when it is not a separate node |
| orthogonal serial reporting | one report payload per reporting slot | active-set reporting in canonical results |
| sensing-waveform leakage | continuous sensing component entering a communication receiver | payload interference |
| predicted belief | Gaussian state `(xhat,P)` from an upstream tracker | target truth, perfect knowledge |
| confirmation or re-detection | target-specific sensing after coarse acquisition | blind first acquisition |
| fair sensing potential | soft-min detection utility plus deficit penalty | sum-PD objective |
| exact-marginal greedy | greedy addition using the exact potential increment | first-order greedy |
| upper-resource reference | all feasible reports used | detection upper bound |
| finite-instance audit | sampled monotonicity/submodularity checks | theorem, guarantee |
| active evidence acquisition | joint acquisition, transport, and fusion of detection-relevant observations | version-specific active sensing names |
| detector-consistent information | KL or Jeffreys information derived from the implemented exact LLR | calling information the final detection metric |
| multi-view information complementarity | scenario-dependent views whose joint worst-case value exceeds their singleton values | generic diversity when complementarity is meant |
| decision-dependent information | information value $I_{a,s}(\mathbf{x})$ affected by network decisions | treating active sensing coefficients as universally fixed |
| active column | one target--fusion mixed-mode observation bundle with resource and detector values | bundle/column used interchangeably without definition |
| operational detection metric | calibrated worst-scenario $P_D$ at the target $P_{FA}$ | optimizing KL and reporting it as $P_D$ |
| full-load interference envelope | conservative column-separable sensing-power externality model | exact selected-load interference |
| low-RCS evidence rescue | acquisition–transport–fusion co-design that lowers a constrained detectable-RCS threshold | cooperative illumination used as the novelty by itself |
| minimum detectable RCS | smallest RCS meeting worst-target $P_D$ and $P_{FA}$ requirements under declared constraints | an interpolated threshold from sparse or non-monotone points |
| generated evidence | scenario-wise exact-LLR information available before report loss | received information |
| received evidence | generated evidence retained at the final fusion UAV after report erasures | raw echo power |
| network evidence loss | generated minus received detector-consistent information | generic packet count |
| robust evidence retention | minimum scenario-wise received/generated evidence ratio | detection probability |
| receiver-side processing | matched filtering, local exact LLR, and optional DD refinement at the echo receiver | charging all processing to fusion |
| fusion-side processing | fixed, per-observation, and aggregation computation at the final fusion UAV | receiver signal processing |
| receiver-local micro-fusion | local summation of same-receiver exact LLRs before one protected aggregate report | a current performance claim without packet-reliability modelling |
