# V1.2: Joint Fusion–Bundle Optimization

## One-sentence argument

In hard-capacitated multi-UAV cooperative detection, we jointly choose each
target's fusion UAV and observation bundle through a lexicographic bundle
master, using detector-aligned coarse-to-fine pricing to scale beyond exact
enumeration, and validate optimization correctness against an exhaustive
small-system oracle; performance promotion remains conditional on independent
large-sample Monte Carlo evidence.

## Terminology ledger

| Canonical term | Definition | Avoided variants |
|---|---|---|
| observation | One bistatic transmitter–receiver–target tuple `(i,j,q)` | link, measurement, report when the distinction matters |
| remote report | An observation whose receiver `j` differs from fusion UAV `f_q` | communication link |
| observation bundle | A feasible set of observations for one `(q,f)` pair | anchor set, selected links |
| restricted master problem (RMP) | The lexicographic allocation problem over the currently generated bundles | global greedy |
| pricing oracle | The per-`(q,f)` search for a negative-reduced-cost bundle | heuristic score |
| exact pricing | Enumeration of the shortlisted bundle space | global exactness |
| greedy pricing | Detector-marginal construction used when exact pricing is too large | exact column generation |
| active-target false-alarm probability | False alarms divided by false trials for targets with a non-empty detector | configured `P_FA` |
| radar cross section (RCS) | Scheduler-side target echo parameter `sigma_q` in square metres | antenna gain, an optimization variable |

## 1. System boundary

The physical layer is inherited unchanged from the V1.1 true-erasure model.
There are `M` UAVs and `Q` known target hypotheses. UAV `i` illuminates target
`q`, UAV `j` receives the bistatic echo, and one UAV `f_q` fuses the selected
soft observations. Target association, trajectory design, transmit-power
allocation, and waveform-level channel estimation are upstream or fixed. V1.2
changes only the scheduling and shared-resource allocation layer.

For target `q`, let `E_q` be the set of delay–Doppler-valid bistatic
observations. An observation is local when `j=f_q`; otherwise receiver `j`
must send one packet to `f_q`. Remote feasibility therefore depends on the
jointly chosen fusion destination through range, rate, and finite-blocklength
packet reliability.

## 2. Detection model

### 2.1 RCS-to-SINR physical chain

RCS is explicit in the inherited bistatic radar equation. Let `sigma_q` be
the scheduler-side RCS of target `q`, `lambda` the wavelength, and `d_iq` and
`d_jq` the transmitter--target and target--receiver distances. The propagation
gain is

```text
g^s_ijq(sigma_q) = lambda^2 sigma_q
                    / [(4 pi)^3 d_iq^2 d_jq^2].
```

It is deliberately separated from the dimensionless hardware gain
`G^hw_ij = G^tx_i G^rx_j / L^sys_ij`. With effective sensing power
`P_tilde^s_i`, processing gain `G_p`, delay--Doppler and collision capture
`eta^DD_ijq eta^col_ijq`, optional waveform capture `eta^wf_ijq`, thermal
noise `N_0`, residual sensing interference `I^s_j`, and optional waveform
impairment ratio `INR^wf_ijq`, the implemented sensing SINR is

```text
gamma_ijq(sigma_q)
  = P_tilde^s_i G^hw_ij G_p eta^DD_ijq eta^col_ijq eta^wf_ijq
    lambda^2 sigma_q
    / [(4 pi)^3 d_iq^2 d_jq^2
       (N_0 + I^s_j) (1 + INR^wf_ijq)].
```

Thus, with geometry, power, interference, and impairments fixed, doubling RCS
doubles sensing SINR (a `3.01 dB` increase). In the current headline model,
the scheduler uses the configured mean RCS, `sigma_q = bar{sigma}_q`; it does
not observe a current-CPI fluctuating RCS before scheduling. This avoids
granting non-causal side information and keeps RCS separate from antenna and
processing gains.

### 2.2 SINR-to-bundle detection utility

For sensing SINR `gamma_ijq(sigma_q)` and `L` independent looks, define the centred
finite-look local log-likelihood statistic

```text
a_ijq = gamma_ijq / (1 + gamma_ijq),
s_ijq = a_ijq (X_ijq - L).
```

Under `H0`, `X~Gamma(L,1)`; under `H1`, `X~Gamma(L,1+gamma_ijq)`. Hence

```text
E0[s] = 0,
Var0[s] = L a_ijq^2,
E1[s] = L gamma_ijq^2 / (1 + gamma_ijq),
Var1[s] = L gamma_ijq^2.
```

For a remote observation, `A_ijqf~Bernoulli(chi_jf)` is the packet-success
indicator and the received statistic is `s_tilde=A*s`. A failed report is an
exact zero, not Gaussian replacement noise. Local observations use `A=1`.
The post-report moments therefore retain both attenuation and mixture
variance. Deflection weights are proportional to the received mean gap divided
by the received `H0` variance; the fused statistic is

```text
F_q = sum_(i,j in b) w_ijqf s_tilde_ijqf.
```

Final Monte Carlo detection uses a threshold calibrated to the implemented
true-erasure `H0` mixture for every method. Bundle scheduling uses the faster
moment-matched Cornish–Fisher prediction
`p_qfb(sigma_q) = P_hat_D(F_qfb > tau_qfb | sigma_q)`. Consequently, RCS
enters the master problem through every precomputed bundle coefficient
`p_qfb`; its absence from the decision-variable list does not mean that it is
absent from the physical or detector model. Thus V1.2 claims
detector-aligned scheduling, not exact optimization of empirical Monte Carlo
`P_D`. The prediction gap is an explicit experiment target.

## 3. Joint bundle formulation

For every target `q`, candidate fusion UAV `f`, and feasible observation bundle
`b in B_qf`, precompute the following quantities conditional on the declared
RCS scenario:

```text
p_qfb : predicted detection probability,
r_qfb : number of remote reports,
a_qfbj: receiver-j processing load,
c_qfb : fusion processing load = |b|.
```

The current optimization is therefore conditional on fixed exogenous RCS
values; it does not optimize RCS. If target RCS is uncertain at scheduling
time, a stochastic or robust extension must create scenario-dependent
coefficients `p^s_qfb = p_qfb(sigma^s_q)` and enforce deficits per scenario,
for example `d_qs >= P_D_req - sum_fb p^s_qfb u_qfb`, followed by a declared
expectation, worst-case, or CVaR objective. That extension is outside the
current V1.2 claim.

Let `u_qfb` be binary and equal to one when target `q` uses `(f,b)`. Let
`d_q>=0` denote the deficit from the required operating point `P_D_req`, and
let `d_max` upper-bound every target deficit. The optimization is

```text
lexmin (d_max, sum_q d_q, sum_qfb r_qfb u_qfb,
        sum_qfb c_qfb u_qfb)
```

subject to

```text
sum_fb u_qfb = 1                                           for every q,
d_q >= P_D_req - sum_fb p_qfb u_qfb                       for every q,
d_max >= d_q                                               for every q,
sum_qb u_qfb <= C_f_target                                 for every f,
sum_qfb a_qfbj u_qfb <= C_j_rx                            for every j,
sum_qb c_qfb u_qfb <= C_f_fusion                          for every f,
sum_qfb r_qfb u_qfb <= K_report,
sum_qfb c_qfb u_qfb <= K_total.
```

The per-target observation cap and local-observation cap are enforced while a
bundle is generated. The empty bundle is allowed, because assigning a target
to a fusion UAV and spending sensing-processing resources are distinct
decisions. An empty bundle has `p=r=c=a=0` but still consumes one target slot
at its selected fusion UAV.

### Proposition 1: representation equivalence

If `B_qf` contains every feasible observation subset for every `(q,f)`, the
bundle formulation is equivalent to the original joint fusion-assignment and
observation-selection problem.

Proof sketch: any feasible original solution maps to exactly one selected
bundle for each target. Conversely, every selected bundle uniquely determines
`f_q` and all observation variables. The bundle resource vector is the sum of
the original variable loads, so all hard constraints and the target utility are
preserved. This is a bijection between feasible decisions, not a relaxation.

### Proposition 2: lexicographic correctness of the RMP

The implementation solves four mixed-integer programs sequentially. After
each tier, it adds an upper bound at the achieved optimum before solving the
next tier. Therefore the returned integer solution is lexicographically
optimal over the generated column set, up to the declared numerical locking
tolerance. No scalar trade-off weight is introduced.

## 4. Coarse-to-fine column generation

The complete bundle set grows combinatorially. V1.2 first ranks local and
remote observations separately using coarse detector predictions. It retains
both families, evaluates generated bundles using the refined delay–Doppler
table, and initializes the RMP with:

1. an empty bundle for every admissible `(q,f)` pair;
2. detector-marginal greedy prefixes;
3. at least one singleton from each available local/remote family.

The algorithm then relaxes bundle integrality and performs column generation
in lexicographic order:

1. minimize the relaxed worst deficit and price new columns;
2. once no improving first-tier column remains, lock the worst deficit;
3. minimize total deficit and price again;
4. solve the accumulated RMP as the exact four-tier integer program.

Let `beta_q`, `pi_f_target`, `pi_j_rx`, `pi_f_fusion`, `pi_report`, and
`pi_total` be non-negative LP scarcity prices, and let `nu_q` be the target
assignment dual. A candidate bundle has reduced cost

```text
rc(q,f,b) = -nu_q - beta_q p_qfb + pi_f_target
            + sum_j pi_j_rx a_qfbj
            + pi_f_fusion c_qfb
            + pi_report r_qfb + pi_total c_qfb.
```

A negative reduced cost improves the current relaxed master tier. These prices
come from hard constraints; they are not replacements for the removed manual
`lambda_c`.

For at most `bundle_exact_pricing_max_candidates` shortlisted observations,
the pricing oracle enumerates every subset up to the per-target cap. Otherwise
it uses multi-start detector-marginal greedy pricing. Consequently:

- exact pricing plus a complete candidate pool certifies LP column-generation
  convergence for the first two tiers;
- the final integer solution is exact for the generated RMP;
- greedy pricing is a scalable heuristic and carries no global-optimality
  claim;
- small-system comparison with the exhaustive joint oracle is the required
  optimization-quality evidence.

## 5. Algorithm

```text
Input: belief-side link tables, hard capacities, P_D_req
Output: fusion plan f_q and selected observations S_q

1  Build coarse local/remote candidate lists for every (q,f).
2  Generate empty, singleton-family, and greedy-prefix seed bundles.
3  for tier in [worst deficit, total deficit]:
4      repeat up to I_CG iterations:
5          solve the tier's LP relaxation and obtain constraint duals;
6          for every (q,f), solve exact or greedy reduced-cost pricing;
7          add every new negative-reduced-cost bundle;
8      until no improving bundle is found.
9  Sequentially solve the four integer RMP tiers, locking each optimum.
10 Return the chosen fusion UAV and observation bundle for every target.
```

With `N_qf` retained observations and per-target cap `K_q`, exact pricing costs
`O(sum_{r=0}^{K_q} C(N_qf,r))` per pair. Greedy pricing costs approximately
`O(S K_q N_qf)` detector evaluations for `S` starts. The integer RMP contains
one binary variable per generated bundle plus `Q+1` continuous deficit
variables.

## 6. Reproducibility and paired evaluation

Detection randomness is keyed by
`(seed, trial, q, i, j, hypothesis, false-alarm index)`. Two algorithms that
share an observation therefore share its packet draw and local statistic even
when their other selected observations differ. Threshold quadrature is also
deterministic. Reported results must include active-target `P_FA`, its
trial-cluster interval, mean and weak-target `P_D`, remote reports, processing
utilization, generated-column count, pricing iterations, and small-oracle
lexicographic gaps.

## 7. Required evidence ladder

1. **Correctness:** hard-budget unit tests and calibrated active-target `P_FA`.
2. **Optimization:** exact-pricing V1.2 versus the joint oracle for `M=3/4`,
   `Q=2/3`.
3. **Architecture:** joint bundle versus fixed V1.1 fusion plus joint bundle.
4. **Cooperation:** joint bundle versus local-only bundle.
5. **Selection:** joint bundle versus budgeted sensing-SINR and V1.1
   detector-aware sequential selection.
6. **Scale/stress:** fixed physical profile with a preregistered report-budget
   frontier; no RCS or radar-gain retuning after seeing results.
7. **Promotion:** independent `MC=200` screening followed by `MC=1000` only if
   calibration, oracle gap, and paired baseline comparisons pass.

## 8. Claim–evidence map

| Claim | Evidence | Status |
|---|---|---|
| Oracle and proposed feasible sets match | Local-cap regression plus shared hard-budget predicate | supported |
| RMP solves the formal lexicographic objective | Four sequential MILPs with locked tiers | supported |
| Exact small pricing can recover the joint oracle | Automated regression and oracle runner | supported for tested small systems |
| Keyed Monte Carlo strengthens paired comparisons | Selection-set invariance regression | supported |
| V1.2 improves detection over fixed fusion, V1.1 sequential selection, and budgeted sensing-SINR | Independent MC=200 paired comparison | supported for the frozen compact-800m protocol |
| Two remote reports improve mean detection over local-only bundles | MC=200 paired comparison | not resolved; 95% interval crosses zero |
| Greedy pricing scales to the full 15-UAV/10-target regime | Runtime/column-count sweep | needs evidence |
| OTFS waveform behavior supports the scheduling surrogate | Waveform-level validation | outside this optimization revision; needs evidence |

## 中文结构说明

- 核心创新被限定为“硬资源约束下的联合 fusion–bundle 分配”，不宣称
  cooperative bistatic sensing 本身的新颖性。
- `p_hat_qfb` 是调度侧解析预测，最终 `P_D/P_FA` 来自统一校准后的 Monte
  Carlo detector；两者没有混写成同一个量。
- local anchor 已从硬约束降为候选 bundle family，优化器可以主动舍弃弱
  local observation。
- exact、restricted exact 和 greedy heuristic 的可宣称范围分别写明，避免
  将 RMP 最优误写成完整组合空间全局最优。
- 当前尚缺正式大样本性能证据，因此本文档没有制造“显著优于 baseline”之类
  结果性表述。
