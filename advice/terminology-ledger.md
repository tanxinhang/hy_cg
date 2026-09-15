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
