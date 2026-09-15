# Active Evidence Acquisition for Resource-Constrained Multi-UAV OTFS-ISAC

## Draft

### Recommended title

**Active Evidence Acquisition for Resource-Constrained Multi-UAV OTFS-ISAC Cooperative Detection**

Alternatives:

1. Detector-Consistent Active Observation Design for Multi-UAV OTFS-ISAC
2. Robust Multi-View Evidence Acquisition under Sensing–Communication–Computation Coupling
3. From Passive Fusion to Active Evidence Acquisition in Cooperative OTFS-ISAC Networks

### One-sentence argument

In resource-constrained multi-UAV OTFS-ISAC, we formulate cooperative sensing as active evidence acquisition, derive detector-consistent information values from the exact likelihood model, construct robust mixed-mode observations, and schedule them globally under sensing, reporting, computation, and energy constraints, supported by paired calibrated detection experiments and bounded by finite aspect scenarios, a single sensing interval, and a conservative interference envelope.

### Abstract

Cooperative sensing is commonly formulated as selecting and fusing observations that already exist, although a multi-UAV integrated sensing and communication network can also control how those observations are acquired. This separation obscures the coupling between sensing power, delay–Doppler refinement, independent looks, report reliability, fusion placement, and shared computation and energy resources. We formulate the system as **active evidence acquisition**: the network chooses which bistatic view to create, how to acquire it, where to fuse it, and whether its resource cost is justified by its detection value. Starting from the exact local log-likelihood ratio (LLR), we derive forward Kullback–Leibler (KL) and Jeffreys information, prove their scaling under observed hypothesis-independent report erasures, and exploit their additivity under conditionally independent observations. Path-dependent aspect scenarios convert robust sensing from uniform signal attenuation into multi-view information complementarity. A certified branch-and-bound oracle prices link–mode bundles, while a fleet-level mixed-integer master selects active columns subject to transmitter energy and load, receiver capacity, report, fusion, and computation constraints. Information guides candidate generation, whereas calibrated worst-scenario detection probability at the target false-alarm probability, (P_D@P_{FA}), determines the operational schedule. Across 90 paired target instances from nine geometry clusters, active modes increased worst-scenario (P_D) by 0.03025 on average (95% cluster-bootstrap CI, 0.01998–0.04006). In nine small fleet instances, the held-out mean-target gain was 0.01717 (0.00360–0.03571), while the held-out worst-target interval included zero. These results close the information-to-detection loop for the tested single-interval finite-scenario setting, but do not yet establish a full fleet-wide certificate, sequential belief adaptation, or trajectory control.

## Unified problem formulation

The research question is no longer “which observations should be selected?” It is:

> Under sensing, communication, computation, and energy constraints, how should a multi-UAV OTFS-ISAC network actively acquire, transport, and fuse the evidence that is most valuable for detection?

For target (q), an action is

\[
a=(i,j,q,m,f_q),
\]

where (i) transmits the sensing waveform, (j) receives the echo, (m) selects sensing power, independent looks, and delay–Doppler refinement, and (f_q) is the fusion UAV. A system decision (\mathbf{x}) chooses a feasible set of actions for all targets. The conceptual objective is

\[
\max_{\mathbf{x}\in\mathcal X}
\Phi\!\left(
\{P_{D,q}(\mathbf{x};s)\}_{q,s}
\right)
\quad\text{s.t.}\quad
P_{FA,q}(\mathbf{x};s)\le \alpha,
\]

with shared sensing, reporting, computation, receiver, and energy constraints in (\mathcal X). The implementation uses lexicographic fairness: minimize the worst target detection deficit, then the total deficit, energy, remote reports, and CPU cycles.

## Theory: one ladder, not a collection of metrics

### T1 — Signal quality becomes likelihood separability

For (L) independent complex samples with sensing SINR (\gamma),

\[
z_\ell|H_0\sim\mathcal{CN}(0,\sigma^2),\qquad
z_\ell|H_1\sim\mathcal{CN}(0,(1+\gamma)\sigma^2).
\]

The exact LLR provides the bridge from a physical link budget to a detection statistic. Deflection and the centered LLR mean gap describe whether one fixed observation is useful, but do not yet decide how to create observations.

### T2 — Likelihood separability becomes detector-consistent information

The forward and reverse divergences are

\[
D_{\mathrm{KL}}(P_1\Vert P_0)=L[\gamma-\log(1+\gamma)],
\]

\[
D_{\mathrm{KL}}(P_0\Vert P_1)=L\left[\log(1+\gamma)-\frac{\gamma}{1+\gamma}\right].
\]

Their sum is

\[
J(P_1,P_0)=\frac{L\gamma^2}{1+\gamma},
\]

which equals the centered exact-LLR mean gap. Information is therefore not an unrelated surrogate; it is an optimization layer derived from the implemented detector. If report arrival (E\sim\mathrm{Bernoulli}(\chi)) is observed and independent of the hypothesis, then

\[
D_{\mathrm{KL}}(P_1^{\mathrm{rx}}\Vert P_0^{\mathrm{rx}})
=\chi D_{\mathrm{KL}}(P_1\Vert P_0).
\]

Under conditional independence, received information is additive across observations. These two identities justify modular pricing but not the final detection claim.

### T3 — Additive information becomes robust multi-view complementarity

For aspect scenario (s\),

\[
I_s(B)=\sum_{a\in B}I_{a,s},\qquad
I^{\mathrm{rob}}(B)=\min_{s\in\mathcal S}I_s(B).
\]

Path-dependent scenarios may reverse the ranking of two views. Robust scheduling must therefore preserve complementary pairs such as ((10,0)) and ((0,10)), even though both have zero singleton worst-case value. This is **multi-view information complementarity**, not merely an SNR diversity gain.

### T4 — Exogenous information becomes decision-dependent information

Active sensing power changes both the desired echo and interference received elsewhere, so the full problem has

\[
I_{a,s}=I_{a,s}(\mathbf{x}),
\qquad
I_s(B)\ne\sum_{a\in B}\bar I_{a,s}
\]

in general. The current implementation exposes this externality but retains a linear master by evaluating every column under a conservative full-load interference envelope. This establishes a safe, column-separable lower-fidelity layer; it is not equivalent to solving the selected-load fixed point.

### T5 — Decision-dependent information becomes sequential information-to-go

The next theoretical boundary is a belief state (b_t\) updated after each observation:

\[
b_{t+1}=\mathcal B(b_t,a_t,y_t),
\]

with expected information gain

\[
\operatorname{EIG}(a_t|b_t)=
\mathbb E_y D_{\mathrm{KL}}
\left[p(\theta|y,a_t,b_t)\Vert p(\theta|b_t)\right].
\]

This layer is a future extension. It must not be claimed from the present single-interval experiments.

## Model: four coupled layers

| Layer | Input | Mechanism | Output | Current boundary |
|---|---|---|---|---|
| Physical acquisition | Geometry, waveform, mode (m), target aspect | Desired echo and cross-UAV leakage; coarse or refined DD response | Scenario-wise (\gamma_{a,s}) | Synthetic aspect law; normalized energy |
| Information | (\gamma_{a,s},L_m,\chi_a) | Exact-LLR KL/Jeffreys and erasure scaling | (I_{a,s}) | Conditional independence |
| Operational detector | Mixed-mode observations and true erasures | Per-mode Gamma LLR, independent calibration and evaluation | (P_D@P_{FA}) | Finite Monte Carlo accuracy |
| Network allocation | Active columns and shared budgets | Lexicographic binary master | One column per target | Exact only over generated columns |

The mode tuple is (m=(p_m,L_m,r_m)), comprising sensing-power scale, independent-look count, and DD-refinement flag. Its explicit compute cost is

\[
C_a=L_m(C_{\mathrm{MF}}+C_{\mathrm{LLR}})
+\mathbf 1_{r_m}C_{\mathrm{DD}}.
\]

The active column additionally records per-transmitter energy and load, receiver load, fusion load, remote reports, and fusion computation. This representation is the contract between physical sensing and network optimization.

## Algorithm: evolution and integrated implementation

The algorithmic tree is

\[
\text{marginal greedy}
\rightarrow\text{bundle MILP}
\rightarrow\text{column generation}
\rightarrow\text{robust pricing}
\rightarrow\text{certified B\&B}
\rightarrow\text{branch-price-and-cut}
\rightarrow\text{sequential planning}.
\]

The implemented integrated pipeline is:

1. Enumerate feasible bistatic links and discrete acquisition modes.
2. Compute scenario-wise detector-consistent information.
3. Retain robust singletons, scenario leaders, and complementary pairs, or use the complete physical pool when it is below the declared limit.
4. Solve each target–fusion pricing problem by branch-and-bound with an optimistic information bound.
5. Expand retained links into feasible mixed-mode active columns.
6. Re-evaluate each column using the exact mixed-mode LLR, observed true erasures, and the full-load interference envelope.
7. Solve the fleet-level lexicographic master under all declared shared budgets.
8. Re-evaluate the selected schedule with independent hold-out detector samples before reporting (P_D@P_{FA}).

The public implementation entry point is `solve_active_evidence_acquisition`. Its result includes a machine-readable scope declaration. The current certificate hierarchy is:

| Level | Statement that may be made |
|---|---|
| Local B&B, untruncated | Exact over the declared candidate pool |
| Complete-pool local mode | Exact over all feasible physical links within the safety limit |
| Global MILP | Exact over generated active columns |
| Full fleet action universe | **Not yet certified** |

The next optimization step should combine column generation with worst-scenario separation and integer branching. A branch-price-and-cut or CG+CCG solver should replace fixed candidate enumeration only after it provides: (i) valid reduced-cost bounds, (ii) an explicit uncertainty separation oracle, (iii) branch-compatible columns, and (iv) a global optimality or declared-gap certificate.

## Current evidence

### Target–fusion pair level

The formal paired experiment used 90 target instances from nine geometry clusters, 2,048 (H_0) calibration samples, and 4,096 independent evaluation samples per scenario.

| Active minus fixed nominal | Estimate | 95% cluster-bootstrap CI |
|---|---:|---:|
| Robust received-information mean | 4.6456 | [1.8370, 8.8870] |
| Robust received-information median | 0.5389 | [0.0885, 0.9434] |
| Worst-scenario (P_D) mean | 0.03025 | [0.01998, 0.04006] |
| Worst-scenario (P_D) median | 0.00439 | [0.00073, 0.01648] |

Mean scenario (P_{FA}) was 0.05024 for active modes and 0.05000 for fixed nominal. Active information improved in 80% of instances, whereas worst-scenario (P_D) improved in 61.1% and was non-worse within (10^{-6}) in 94.4%. The supported claim is a positive average detection consequence, not pointwise monotonic improvement.

### Fleet level

The fleet diagnostic used nine systems with three UAVs, two targets, and 234 generated columns per instance. Selection used 16,384 detector samples per scenario and reporting used an independent 32,768-sample hold-out replay.

| Held-out endpoint | Fixed nominal | Active modes | Paired gain | 95% cluster-bootstrap CI |
|---|---:|---:|---:|---:|
| Mean-target (P_D) | 0.24779 | 0.26496 | 0.01717 | [0.00360, 0.03571] |
| Worst-target (P_D) | 0.13877 | 0.15158 | 0.01282 | [−0.00066, 0.03345] |

Mean scenario (P_{FA}) remained near 0.05. The first endpoint supports a preliminary system-level mean-detection gain; the second is inconclusive.

## Claim–evidence map

| Claim | Evidence | Status |
|---|---|---|
| KL/Jeffreys information is detector-consistent | Algebraic identity and unit tests | Supported under the local Gamma model |
| True observed erasure scales information by (\chi) | Analytic decomposition and unit test | Supported under hypothesis-independent erasure |
| Candidate generation preserves basic cross-scenario complementarity | Adversarial complementary-pair test | Supported for the declared union strategy |
| Mixed modes improve average robust information and average worst-scenario (P_D) | 90 paired targets, nine clusters | Supported for the tested preset |
| Global active scheduling improves held-out mean-target (P_D) | Nine paired small-system instances | Preliminary support |
| Global active scheduling improves held-out worst-target (P_D) | CI includes zero | Inconclusive |
| The solver is globally exact over every feasible network action | No full action-space oracle | Unsupported; do not claim |
| The method adapts beliefs or trajectories over time | No sequential experiment | Future work |

## Research evolution without version-number narration

| Research question | Theory object | Decision object | Algorithm | Status |
|---|---|---|---|---|
| Which available evidence should be fused? | Deflection / detection surrogate | Observation | Marginal greedy | Historical foundation |
| How should fusion and resources be allocated jointly? | Detector-aligned bundle utility | Fusion and bundle | Bundle MILP / CG | Implemented foundation |
| How should state uncertainty be handled? | Worst-case utility | Robust bundle | Robust pricing | Implemented foundation |
| How should observations themselves be created? | KL/Jeffreys information | Link–mode action | Certified B&B and global column master | Current validated focus |
| How should network decisions change information values? | (I_{a,s}(\mathbf{x})) | Global active bundle | Branch-price-and-cut / CG+CCG | Next optimization target |
| How should new observations update later sensing? | EIG / information-to-go | Sequential action | Bayesian design / MPC | Future |
| How should UAV motion change future views? | Information geometry | Trajectory and sensing | Two-timescale control | Long term |

## Assumptions and missing inputs

- The aspect response is synthetic rather than measured or electromagnetically simulated.
- Information additivity assumes conditional independence; correlated joint likelihoods require a covariance-generating physical model.
- Mode energy is normalized, not hardware-calibrated in joules or timing.
- The full-load interference envelope is conservative and does not recover selected-load decision dependence exactly.
- The global experiment contains only nine small systems and is insufficient for a promotion-level worst-target claim.
- The current system is single-interval and does not update a posterior belief or optimize trajectory.
- A target venue, word limit, and required section format have not yet been specified.

## Section outline

1. Introduction: from passive evidence selection to active evidence acquisition.
2. System model: physical acquisition, reporting, exact LLR, uncertainty, and resource budgets.
3. Detector-consistent information: KL/Jeffreys identities, erasure scaling, and complementarity.
4. Active observation model: link–mode–fusion actions and decision-dependent interference.
5. Integrated solver: certified local pricing, active-column construction, and global master.
6. Experiments: identity checks, oracle parity, pair-level detection consequence, and fleet-level hold-out evaluation.
7. Discussion: certificate boundary, conditional independence, conservative interference, and scale.
8. Conclusion: established single-interval closure and the path to robust sequential information control.

## 中文结构说明

本稿把所有历史版本压缩为一条“问题深化链”，正文不再按版本号罗列功能。理论层从物理 SINR 依次推进到 LLR、KL/Jeffreys、多视角互补、决策相关信息和未来的 EIG；模型层用统一的主动动作 (a=(i,j,q,m,f_q)) 串起感知、上报、融合和资源约束；算法层严格区分局部候选池证书、生成列上的全局证书与尚未完成的全动作空间证书。

当前可以稳定主张的是：已在有限方位场景、单个感知区间和保守满载干扰包络下闭合“信息增益 → 混合模式似然 → 全局调度 → (P_D/P_{FA})”链条。不能提前主张完整 branch-price-and-cut、贝叶斯序贯感知、轨迹控制或 worst-target 系统级提升。后续实验应优先扩大系统级样本、补齐强基线与模块消融，再扩展新理论。

