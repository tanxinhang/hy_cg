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
