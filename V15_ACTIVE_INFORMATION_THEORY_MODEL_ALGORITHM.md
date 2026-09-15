# V1.5-System: Global Active Information Bundle Scheduling

## One-sentence argument

In a single-CPI multi-UAV ISAC network, mixed-mode exact-LLR active columns connect received-information pricing to a global energy/CPU/report-constrained master, while paired Monte Carlo evidence measures—without assuming—the downstream detection consequence.

## Terminology ledger

| Canonical term | Meaning | Avoided ambiguity |
|---|---|---|
| received KL information | KL divergence retained after an observed packet-erasure channel | not “effective SNR” |
| Jeffreys divergence | sum of the two directed KL divergences | not a new detector statistic |
| active observation | target, bistatic link, and acquisition-mode tuple `(q,i,j,m)` | not merely a selected link |
| aspect scenario | one path-dependent finite RCS-factor realization | not a scalar RCS interval |
| information-certified pricing | branch-and-bound pricing with a valid information upper bound | not a certificate for final `P_D` |
| mixed-mode exact LLR | exact LLR sum in which each observation carries its own `gamma_a` and `L_a` | not the V1.4 global-look detector |
| active column | `(target, fusion, observations, modes)` with scenario `P_D`, information, energy, CPU and report coefficients | not a link-only bundle |
| full-load power envelope | all UAV interferers radiate at the largest declared sensing-power scale | conservative cross-column coupling, not selected-load equality |
| complete-pool certificate | exact branch-and-bound result over every feasible physical link admitted by the configured full-pool limit | stronger than shortlist exactness |

## Scope and system boundary

V1.5-System remains an opt-in research path and does not mutate the frozen V1.4 detector or its published result files. It designs one coherent processing interval using discrete power, look-count, and delay–Doppler-refinement modes. It assumes that the fusion destination observes whether each report arrived and that packet success is independent of the target hypothesis. The present aspect law is a transparent finite-scenario abstraction, not a calibrated target signature.

The implementation boundary is deliberate:

1. received information is the pricing objective and the bounding quantity;
2. worst-case finite scenarios are enforced during bundle generation;
3. mixed-mode exact-LLR Monte Carlo now evaluates every active column at the declared `P_FA`, while a larger promotion campaign is still required before replacing V1.4;
4. the global master is exact over its generated columns and uses a conservative full-load interference envelope;
5. trajectory control, learned RCS signatures, and joint column-and-scenario generation remain outside this single-CPI module.

## Theory

### Local observation model

For one path and one sensing mode, let the `L` independent complex observations satisfy

\[
H_0:y_\ell\sim\mathcal{CN}(0,\sigma^2),\qquad
H_1:y_\ell\sim\mathcal{CN}(0,(1+\gamma)\sigma^2).
\]

The forward and reverse divergences are

\[
D(P_1\Vert P_0)=L\{\gamma-\log(1+\gamma)\},
\]

\[
D(P_0\Vert P_1)=L\left\{\log(1+\gamma)-\frac{\gamma}{1+\gamma}\right\}.
\]

Their sum is the Jeffreys divergence

\[
J(P_1,P_0)=L\frac{\gamma^2}{1+\gamma}.
\]

This is exactly the separation between the centered exact-LLR means already used by the detector. Thus, the new information objective is tied algebraically to the existing likelihood model rather than introduced as an unrelated heuristic.

### Proposition 1: observed erasure scaling

Let `E` denote report arrival, with `Pr(E=1)=chi`, independent of the hypothesis, and let the fusion node observe `E`. When `E=0`, both hypotheses induce the same missing-report atom. When `E=1`, the node observes the local statistic. Splitting KL divergence over these two disjoint events gives

\[
D(P_1^{\mathrm{rx}}\Vert P_0^{\mathrm{rx}})
=\chi D(P_1\Vert P_0).
\]

The same identity holds for the reverse and Jeffreys divergences. It is exact under the stated erasure model; it is not asserted for replacement-by-noise or hypothesis-dependent packet loss.

### Proposition 2: additivity

For conditionally independent received observations, the joint likelihood factorizes. The KL divergence of the product distributions is therefore the sum of their divergences. For a bundle `B` and aspect scenario `s`,

\[
I_s(B)=\sum_{a\in B}\chi_a L_a
\left[\gamma_{a,s}-\log(1+\gamma_{a,s})\right].
\]

This modularity is what permits an auditable pricing oracle and a valid optimistic bound.

## Model upgrade

An active observation is `a=(i,j,q,m)`. Mode `m` carries a power scale `p_m`, an independent-look count `L_m`, a refinement flag, and normalized energy `e_m=p_m L_m`. The nominal coarse or refined sensing SINR supplied by the existing physical model is transformed as

\[
\gamma_{a,s}=p_m\,g_s(\phi_{ijq})\,\gamma^{\mathrm{coarse/fine}}_{ijq}.
\]

Here `phi_ijq` is the horizontal bistatic viewing-bisector azimuth. For scenario angle `theta_s`, the current transparent aspect abstraction is

\[
g_s(\phi)=g_{\min}+(1-g_{\min})\cos^2(\theta_s-\phi).
\]

Unlike a single scalar RCS multiplier, this construction can reverse the ranking of two paths across scenarios and therefore represents actual multi-view diversity. The defaults expose three modes: eco `(0.5,8,coarse)`, nominal `(1.0,16,fine)`, and intensive `(1.25,32,fine)`. Every field is validated before execution.

## Algorithm upgrade

For each target–fusion pair, binary variable `x_ijm` selects at most one mode for each physical link. The pricing problem is

\[
\max_x\; \min_{s\in\mathcal S}\sum_{ijm}x_{ijm}I_{ijm,s}
-\lambda_E\sum_{ijm}x_{ijm}e_m
-\lambda_R\sum_{ijm}x_{ijm}r_{ij},
\]

subject to one-mode-per-link, observation-count, normalized-energy, and physical/report-feasibility constraints. `r_ij` is one for a remote report and zero for evidence already local to the fusion node.

The default candidate pool is now the capped union of three families: robust singletons, per-scenario leaders, and links belonging to the strongest complementary pairs. This prevents the elementary failure in which `(10,0)` and `(0,10)` are both discarded because each has zero singleton worst-case value. A `full` strategy bypasses shortlisting when the complete physical pool lies below the declared safety limit.

The oracle performs an exact depth-first branch-and-bound search over the retained mode choices. At a partial node, its optimistic bound adds, separately for each scenario, the largest remaining per-link information increments that could occupy the unfilled slots. It ignores future non-negative energy and reporting costs. Every feasible continuation is componentwise no larger than this relaxation, so the minimum of the relaxed scenario totals is a valid upper bound. A node is safely pruned when this bound cannot exceed the incumbent.

If the declared node limit is not reached, the result is exact over the declared pool. The certificate scope is returned explicitly as `shortlist_union`, `declared_pool`, or `complete_pool`. If the node limit is reached, the root relaxation is returned as a conservative upper bound and the nonzero incumbent gap is reported.

## Reproducible experiment and current evidence

`tools/run_active_information_v15.py` compares joint link-and-mode choice (`active_modes`) with robust link choice restricted to the nominal mode (`fixed_nominal`). Both methods share the same 100 target instances, fusion decisions, four aspect scenarios, four-observation cap, and normalized energy budget of 64. The configuration and command arguments are written to an artifact-specific immutable manifest.

The historical Phase-A 10-trial run in `results_active_information_v15/run_0c4d9ba05082` produced:

| Method | Mean worst-scenario information | Median | Mean energy | Mean observations | Exact rate | Maximum certificate gap |
|---|---:|---:|---:|---:|---:|---:|
| fixed nominal | 43.3951 | 1.8405 | 64.0 | 4.0 | 1.00 | 0 |
| active modes | 49.2893 | 2.1570 | 64.0 | 4.0 | 1.00 | 0 |

The paired mean gain was 5.8942, or 13.58%. The 100% non-worse rate is a dominance sanity check because fixed nominal is a feasible active-mode restriction; it is not the headline result. The large mean–median separation motivated the distributional and detection analysis below.

## Mixed-mode exact-LLR detector

Each observation now carries its own mode-specific `gamma_a` and `L_a`. For scenario `s`, its exact contribution is

\[
\ell_{a,s}=-L_a\log(1+\gamma_{a,s})
+\frac{\gamma_{a,s}}{1+\gamma_{a,s}}X_a,
\]

where `X_a|H0 ~ Gamma(L_a,1)` and `X_a|H1 ~ Gamma(L_a,1+gamma_a,s)`. The fusion statistic is `Lambda_q=sum_a E_a ell_a`, with observed Bernoulli erasures. H0 calibration and independent H0/H1 evaluation use deterministic streams keyed by target, fusion, scenario, physical link and physical mode parameters. The strict decision `Lambda > threshold` matches V1.4 and correctly handles the erasure atom at zero.

## Power and computation coupling

The physical table builder accepts one sensing-power scale per UAV before it forms desired signals or leakage fields. Consequently, increasing transmitter `i` changes both its echo numerator and the interference denominators at other receivers. Active columns are valued under a full-load envelope `p_bar=max_m p_m` for every interfering UAV:

\[
\gamma_{ijq,m,s}^{env}=
\frac{p_m S_{ijq}g_s(\phi_{ijq})}
{N_{ij}+\sum_{k\ne i}\bar p_k I_{kj}}.
\]

This is conservative and column-separable; it does not claim equality to the interference produced by the selected columns. Refinement is no longer free. Observation compute cost is

\[
C_a=L_a(C_{MF}+C_{LLR})+\mathbf 1_{refined}C_{DD},
\]

and the active column also includes the existing fusion fixed, per-observation and cubic costs.

## Global active-column master

An active column contains target, fusion UAV, physical observations, sensing modes, scenario information, scenario `P_D/P_FA`, per-transmitter energy and load, receiver load, reports and CPU cycles. The global binary master lexicographically minimizes

\[
(d_{max},\sum_q d_q,E_{total},R_{total},C_{total}),
\qquad
d_q\ge P_D^{req}-P_{D,qc}^{worst},
\]

subject to exactly one column per target and fleet-wide fusion-target, transmitter-energy, transmitter-load, receiver-load, fusion-observation, CPU, report and total-observation budgets. KL remains a pricing and candidate-generation quantity; worst-scenario `P_D@P_FA` determines the reliability deficit. The MILP result is exact over the generated active-column pool.

## Multi-seed paired evidence

The pair-level run `results_active_information_v15_system/run_80db6de3b0ec` used 90 paired target instances from nine geometry clusters, 2,048 H0 calibration samples and 4,096 independent evaluation samples per scenario. Mean scenario PFA was 0.05024 for active modes and 0.05000 for fixed nominal.

| Paired active minus fixed nominal | Estimate | 95% cluster-bootstrap CI |
|---|---:|---:|
| received-information mean | 4.6456 | [1.8370, 8.8870] |
| received-information median | 0.5389 | [0.0885, 0.9434] |
| worst-scenario `P_D` mean | 0.03025 | [0.01998, 0.04006] |
| worst-scenario `P_D` median | 0.00439 | [0.00073, 0.01648] |

The information gain had `P10/P50/P90 = 0/0.5389/8.0569`, mean `Delta log(1+I)=0.1421`, and normalized mean gain 0.1654. Among 22 weak-information pairs (`I_fixed <= 1`), mean and median information gains were 0.0845 and 0.0091. Active information improved in 80% of instances; worst-scenario `P_D` improved in 61.1% and was non-worse within `1e-6` in 94.4%. These results support a positive average detection consequence, not pointwise monotonicity.

The factorial diagnostic showed that the improvement cannot be attributed to DD refinement in this preset: nominal already uses refinement, so refinement-only exactly matched fixed nominal. Power-only and looks-only both had positive information and `P_D` mean-gain confidence intervals. Looks-only achieved the largest mean information gain (6.0599), while the joint declared mode set achieved 4.6456 because it does not contain every Cartesian power–look–refinement combination. Thus the present claim is joint discrete mode selection, not an isolated causal superiority of any one component.

The fleet-level run `results_active_system_v15/run_8386b9d8389f` used nine three-UAV/two-target system instances and 234 generated columns per instance. Column selection used 16,384 evaluation samples per scenario, whereas all reported detection values came from an independent 32,768-sample holdout replay. Under shared UAV energy, CPU, report and load caps, active columns raised heldout mean-target `P_D` from 0.24779 to 0.26496. The paired gain was 0.01717 with a cluster-bootstrap 95% CI of [0.00360, 0.03571]. Heldout worst-target `P_D` increased from 0.13877 to 0.15158, but its paired CI [-0.00066, 0.03345] included zero; this endpoint is therefore inconclusive. Mean scenario PFA remained near target (0.05021 versus 0.04957). This is system-level support for mean detection on a small diagnostic, not a worst-target or full-scale promotion result.

## Claim–evidence map

| Claim | Evidence | Status |
|---|---|---|
| Jeffreys divergence equals the exact-LLR mean gap | analytic identity plus unit test | established under the local Gamma model |
| observed independent erasure scales KL by `chi` | analytic decomposition plus unit test | established under true erasure |
| aspect scenarios create path-dependent rankings | controlled-azimuth unit test | established for the declared abstraction |
| pricing preserves scenario-complementary pairs | adversarial `(10,0)/(0,10)` unit test | established for the union strategy |
| pricing is exact over its declared pool when untruncated | brute-force parity, complete-pool scope test and zero gaps | established for tested finite instances |
| mixed-mode detector controls scenario PFA | independent calibrated/evaluation Monte Carlo | supported at mean PFA near 0.05 |
| active modes improve robust received information | 90 paired targets, nine clusters | supported for the declared preset |
| information gain has a positive average detection consequence | paired worst-`P_D` cluster-bootstrap CI excludes zero | supported, not pointwise guaranteed |
| global active columns improve heldout mean-target detection | nine small-system paired instances; CI excludes zero | preliminary system-level support |
| global active columns improve heldout worst-target detection | nine small-system paired instances; CI crosses zero | inconclusive |

## Assumptions and missing evidence

- Conditional independence is used for information additivity; a future latent physical covariance model should replace hand-set correlation sensitivities before correlated-information claims are made.
- Aspect scenarios are synthetic. Measured or electromagnetically simulated signatures are required for target-specific claims.
- Mode energy remains normalized look-power energy. Hardware timing and Joule calibration are required before platform-energy claims are made.
- The full-load power envelope is conservative; selected-load iterative or decomposition methods are needed to recover less conservative cross-column interference coupling.
- The global master is exact over 234 generated columns, not the complete fleet-wide active-column universe.
- The nine-system diagnostic is too small for release promotion. Larger system sizes, more geometry clusters and independent detector draws remain required before V1.5-System replaces V1.4 results.

## Recommended manuscript outline

1. **Problem:** passive fusion ranking cannot decide how evidence should be acquired.
2. **Information bridge:** derive KL, Jeffreys, erasure scaling, and additivity from the exact likelihood model.
3. **Active model:** define link–mode observations, energy, refinement, and path-dependent aspect scenarios.
4. **Robust pricing:** state the max–min problem, branch-and-bound relaxation, and certificate boundary.
5. **Evidence:** report identities, brute-force parity, certificates, and paired information results before any detection claim.
6. **Discussion:** separate current finite-scenario single-CPI evidence from future correlated, trajectory-aware, and hardware-calibrated extensions.

## 中文结构说明

本次升级把论证主线从“融合端如何排序已有观测”推进到“系统应主动采集哪些观测”。理论层以现有精确 LLR 为起点，证明 KL、Jeffreys 散度、真实擦除缩放和独立观测可加性；模型层把链路扩展为“链路＋功率＋积累次数＋DD 精化”的主动观测，并用路径相关方位场景替代统一缩放的 RCS 区间；算法层实现带合法上界和证书间隙的分支定界定价器。

本轮已经闭合此前缺失的四条链：mixed-mode exact LLR、场景互补候选池与完整池证书、全局能量/CPU/收发资源 master，以及多种子配对统计。新的证据支持“平均信息增益伴随正的平均 worst-scenario `P_D` 增益”，但没有声称每个目标都单调改善。系统级结果仍限于九个小规模实例；下一步应扩大 promotion 样本，而不是增加 trajectory、DRL、CVaR 或更多 sensing modes。
