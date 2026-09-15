# V1.5 Robust Information-Aware Active Observation Design

## One-sentence argument

Jointly choosing bistatic paths and discrete acquisition modes by their worst-case received hypothesis-separation information converts the scheduler from passive fusion ranking into an auditable active-observation design problem, while leaving end-to-end detection claims to the exact-LLR Monte Carlo layer.

## Terminology ledger

| Canonical term | Meaning | Avoided ambiguity |
|---|---|---|
| received KL information | KL divergence retained after an observed packet-erasure channel | not “effective SNR” |
| Jeffreys divergence | sum of the two directed KL divergences | not a new detector statistic |
| active observation | target, bistatic link, and acquisition-mode tuple `(q,i,j,m)` | not merely a selected link |
| aspect scenario | one path-dependent finite RCS-factor realization | not a scalar RCS interval |
| information-certified pricing | branch-and-bound pricing with a valid information upper bound | not a certificate for final `P_D` |

## Scope and system boundary

V1.5 adds an opt-in research path and does not mutate the frozen V1.4 detector or its published result files. The new layer designs one coherent processing interval using discrete power, look-count, and delay–Doppler-refinement modes. It assumes that the fusion destination observes whether each report arrived and that packet success is independent of the target hypothesis. The present aspect law is a transparent finite-scenario abstraction, not a calibrated target signature. Geometry, communication reliability, refined sensing tables, and fusion assignments still come from the existing simulator.

The implementation boundary is deliberate:

1. received information is the pricing objective and the bounding quantity;
2. worst-case finite scenarios are enforced during bundle generation;
3. V1.4 exact-LLR Monte Carlo remains the authority for end-to-end `P_D` and `P_FA` promotion;
4. trajectory control, learned RCS signatures, and joint column-and-scenario generation remain outside this single-CPI module.

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

The oracle first shortlists physical links by their best robust mode and then performs an exact depth-first branch-and-bound search over mode choices. At a partial node, its optimistic bound adds, separately for each scenario, the largest remaining per-link information increments that could occupy the unfilled slots. It ignores future non-negative energy and reporting costs. Every feasible continuation is componentwise no larger than this relaxation, so the minimum of the relaxed scenario totals is a valid upper bound. A node is safely pruned when this bound cannot exceed the incumbent.

If the declared node limit is not reached, the returned solution is exact over the shortlisted candidates and its certificate gap is zero. If the limit is reached, the root relaxation is returned as a conservative upper bound and the nonzero incumbent gap is reported. The shortlist itself is an explicit scope boundary: exactness applies to the retained candidate set, not to paths removed before enumeration.

## Reproducible experiment and current evidence

`tools/run_active_information_v15.py` compares joint link-and-mode choice (`active_modes`) with robust link choice restricted to the nominal mode (`fixed_nominal`). Both methods share the same 100 target instances, fusion decisions, four aspect scenarios, four-observation cap, and normalized energy budget of 64. The configuration and command arguments are written to an artifact-specific immutable manifest.

The 10-trial run in `results_active_information_v15/run_0c4d9ba05082` produced:

| Method | Mean worst-scenario information | Median | Mean energy | Mean observations | Exact rate | Maximum certificate gap |
|---|---:|---:|---:|---:|---:|---:|
| fixed nominal | 43.3951 | 1.8405 | 64.0 | 4.0 | 1.00 | 0 |
| active modes | 49.2893 | 2.1570 | 64.0 | 4.0 | 1.00 | 0 |

The paired mean gain is 5.8942, or 13.58%, and `active_modes` is not worse in all 100 paired target instances. The large mean–median separation shows strong geometry heterogeneity; therefore this result supports the algorithmic value of mode choice but is not yet a population-level detection claim. No `P_D` promotion is made from this information-only experiment.

## Claim–evidence map

| Claim | Evidence | Status |
|---|---|---|
| Jeffreys divergence equals the exact-LLR mean gap | analytic identity plus unit test | established under the local Gamma model |
| observed independent erasure scales KL by `chi` | analytic decomposition plus unit test | established under true erasure |
| aspect scenarios create path-dependent rankings | controlled-azimuth unit test | established for the declared abstraction |
| pricing is exact on the shortlist when untruncated | brute-force parity test and zero run-time gaps | established for tested finite instances |
| active modes improve robust received information | 100 paired target instances | supported for the declared preset and seed |
| active modes improve final detection probability | requires exact-LLR Monte Carlo integration | not yet claimed |

## Assumptions and missing evidence

- Conditional independence is used for information additivity; a future latent physical covariance model should replace hand-set correlation sensitivities before correlated-information claims are made.
- Aspect scenarios are synthetic. Measured or electromagnetically simulated signatures are required for target-specific claims.
- Mode energy is normalized look-power energy. A hardware timing and Joule model is required before platform-energy claims are made.
- The current solver prices each target–fusion pair independently. A master problem must enforce fleet-wide CPU, report, and energy coupling for system-level scheduling claims.
- Larger Monte Carlo campaigns, seed sweeps, `P_D/P_FA` calibration, and promotion gates remain required before V1.5 replaces V1.4 results.

## Recommended manuscript outline

1. **Problem:** passive fusion ranking cannot decide how evidence should be acquired.
2. **Information bridge:** derive KL, Jeffreys, erasure scaling, and additivity from the exact likelihood model.
3. **Active model:** define link–mode observations, energy, refinement, and path-dependent aspect scenarios.
4. **Robust pricing:** state the max–min problem, branch-and-bound relaxation, and certificate boundary.
5. **Evidence:** report identities, brute-force parity, certificates, and paired information results before any detection claim.
6. **Discussion:** separate current finite-scenario single-CPI evidence from future correlated, trajectory-aware, and hardware-calibrated extensions.

## 中文结构说明

本次升级把论证主线从“融合端如何排序已有观测”推进到“系统应主动采集哪些观测”。理论层以现有精确 LLR 为起点，证明 KL、Jeffreys 散度、真实擦除缩放和独立观测可加性；模型层把链路扩展为“链路＋功率＋积累次数＋DD 精化”的主动观测，并用路径相关方位场景替代统一缩放的 RCS 区间；算法层实现带合法上界和证书间隙的分支定界定价器。

当前证据足以支持“主动模式提高最坏场景接收信息”这一模块级结论，但不足以声称最终检测概率已经提升。后续应按顺序完成混合模式精确 LLR 检测、全局资源 master、相关物理模型与多种子大样本 promotion；这样可以保持每一步可核验，也不会把信息代理指标误写成最终性能指标。
