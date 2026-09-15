# Low-RCS Evidence Rescue in Resource-Constrained Multi-UAV OTFS-ISAC

## Integrated theory, model, and algorithm draft

### Recommended title

**Low-RCS Evidence Rescue through Acquisition–Transport–Fusion Co-Design in Multi-UAV OTFS-ISAC Networks**

### Central argument

A cooperative network should not be credited merely for illuminating a target
from more locations. Its value must be measured by whether it creates enough
hypothesis-separating evidence for a low-RCS target, preserves that evidence
during reporting, and crosses a declared detection requirement under shared
sensing, communication, computation, and energy limits.

### Abstract

Detecting small unmanned aerial vehicles is limited not only by weak echoes but
also by the loss of already scarce evidence while observations are transported
and processed across a network. We formulate cooperative OTFS-ISAC sensing as
**low-RCS evidence rescue**, an acquisition–transport–fusion co-design problem.
The physical layer creates target-dependent bistatic observations by selecting
transmitter–receiver pairs, sensing modes, independent looks, and delay–Doppler
refinement. The transport layer maps locally generated likelihood information
to the evidence received by one final fusion UAV under report erasures. The
fusion layer performs calibrated exact-likelihood detection while accounting
for receiver-side and fusion-side computation at their actual execution nodes.
We introduce a conservative minimum-detectable-RCS bracket at a prescribed
worst-target detection probability and false-alarm limit, together with network
evidence loss and robust evidence retention as diagnostic quantities. A local
branch-and-bound pricing oracle constructs aspect-robust active observations;
an analytical fusion screen retains high-information destinations and a
locality anchor; and a fleet-level lexicographic MILP first protects detection
fairness, then minimizes evidence loss and resource cost. Current evidence
supports the exact-LLR information identities, mixed-mode target-level gains,
and the corrected single-fusion execution path. A corrected 15-UAV/10-target
instance improves mean target detection from 0.77596 to 0.78951 but leaves the
worst target unchanged at 0.20776. This is a diagnostic, not a population-level
performance claim. The framework therefore separates formation-limited from
transport-limited failure and states precisely what must be demonstrated before
claiming that cooperation rescues low-RCS targets.

## 1. Why cooperative illumination is not the novelty

The literature already contains multi-static transmit/receive beamforming,
multi-cell anti-UAV cooperative beamforming, OTFS cell-free detection with
power allocation, and distributed grouping. Representative primary sources
include:

- [Joint transmit beamforming and receive filtering for cooperative multi-static ISAC](https://ieeexplore.ieee.org/document/10496487), which jointly maximizes radar SINR under communication and power constraints.
- [Cooperative beamforming design for anti-UAV ISAC](https://ieeexplore.ieee.org/document/10815056), which jointly designs multi-cell transmit and receive beamformers and also develops a distributed method.
- [Target detection for OTFS-aided cell-free MIMO ISAC](https://ieeexplore.ieee.org/document/10922196), which combines OTFS, distributed sensing, detection, and sensing-centric power allocation.
- [Distributed multinode cooperative ISAC](https://ieeexplore.ieee.org/document/10892128), which jointly addresses node grouping and beamforming.
- [Robust coherent distributed ISAC under RCS and phase uncertainty](https://arxiv.org/abs/2604.02634), which optimizes expected Kullback–Leibler divergence under statistical RCS variation and synchronization uncertainty.

Accordingly, this work does **not** claim novelty for cooperative illumination,
OTFS cooperative sensing, power allocation, final-fusion selection, or KL-based
beam design in isolation. Its defensible research gap is the missing end-to-end
accounting of low-RCS evidence:

\[
\text{weak-echo deficit}
\rightarrow \text{evidence formation}
\rightarrow \text{evidence transport}
\rightarrow \text{likelihood fusion}
\rightarrow P_D@P_{FA}.
\]

## 2. Primary performance object: minimum detectable RCS

Let \(\mathcal X\) contain every admissible sensing, reporting, fusion, and
resource decision. For required worst-target detection probability
\(P_D^{\rm req}\) and false-alarm limit \(\alpha\), define

\[
\sigma_{\rm det}^{\star}
=\inf\left\{\sigma>0:\exists\mathbf{x}\in\mathcal X,
\min_q P_{D,q}(\sigma;\mathbf{x})\ge P_D^{\rm req},\quad
\max_q P_{FA,q}(\sigma;\mathbf{x})\le\alpha\right\}.
\]

This quantity changes the question from “does cooperation improve average
detection?” to “how small a target can the complete constrained system
reliably detect?” A finite experiment must report a bracket, not an interpolated
point estimate. If \(\sigma^-\) is the largest tested failure and \(\sigma^+\)
the first tested pass, the supported statement is

\[
\sigma_{\rm det}^{\star}\in(\sigma^-,\sigma^+].
\]

The new `bracket_minimum_detectable_rcs` implementation also refuses to hide a
non-monotone Monte Carlo sweep. In the current single-instance diagnostic,
0.05, 0.1, and 0.2 m\(^2\) fail the 0.95 worst-target requirement whereas
0.5 m\(^2\) passes, yielding only the provisional bracket
\((0.2,0.5]\) m\(^2\). Multi-seed confirmation is still required.

## 3. Three-layer evidence model

### 3.1 Evidence formation

An active observation is

\[
a=(i,j,q,m),
\]

where UAV \(i\) illuminates target \(q\), UAV \(j\) receives the echo, and
mode \(m=(p_m,L_m,r_m)\) selects sensing-power scale, independent looks, and
whether delay–Doppler refinement is used. For scenario \(s\), its sensing SINR
\(\gamma_{a,s}\) induces exact-LLR information

\[
D_{a,s}^{10}=L_m[\gamma_{a,s}-\log(1+\gamma_{a,s})],
\qquad
J_{a,s}=\frac{L_m\gamma_{a,s}^2}{1+\gamma_{a,s}}.
\]

The generated evidence of a bundle \(B_q\) is

\[
I_{q,s}^{\rm gen}(B_q)=\sum_{a\in B_q}D_{a,s}.
\]

Path-dependent aspect scenarios make different bistatic views complementary;
this is why robust candidate construction preserves scenario leaders and
complementary pairs rather than only the best nominal singleton.

### 3.2 Evidence transport

One final fusion UAV \(f_q\) remains the canonical architecture. If report
arrival from receiver \(j(a)\) to \(f_q\) has observed, hypothesis-independent
success probability \(\chi_{j(a)f_q}\), then

\[
I_{q,s}^{\rm rx}(B_q,f_q)
=\sum_{a\in B_q}\chi_{j(a)f_q}D_{a,s},
\]

\[
\mathcal L_{q,s}^{\rm net}
=I_{q,s}^{\rm gen}-I_{q,s}^{\rm rx}
=\sum_{a\in B_q}(1-\chi_{j(a)f_q})D_{a,s},
\]

\[
\rho_q^{\rm net}
=\min_s\frac{I_{q,s}^{\rm rx}}{I_{q,s}^{\rm gen}+\epsilon}.
\]

These quantities do not replace \(P_D\). They diagnose whether a weak target
fails because insufficient evidence was formed or because useful local
evidence did not survive the reporting network. A lossless-report oracle sets
\(\chi=1\) while holding the sensing bundle and modes fixed. Small oracle
headroom indicates formation limitation; large headroom indicates transport
limitation.

### 3.3 Likelihood fusion and computation placement

Received exact LLRs are summed at a single final fusion UAV. Under the current
conditional-independence model this is the correct detector, but the model does
not yet support claims about coherent cross-node phase combining or correlated
joint likelihoods.

Compute is now attributed to the node that executes it. Receiver \(j\) incurs

\[
C_j^{\rm rx}=\sum_{a:j(a)=j}
\left[L_m(C_{\rm MF}+C_{\rm LLR})+
\mathbf 1_{r_m}C_{\rm DD}\right],
\]

while fusion UAV \(f\) incurs

\[
C_f^{\rm fus}=C_0+C_1|B_f|+C_3|B_f|^3.
\]

The per-UAV constraint is therefore

\[
C_u^{\rm rx}+C_u^{\rm fus}\le \bar C_u,
\]

instead of charging all observation processing to the selected fusion UAV.

## 4. Integrated optimization model

For each target, an active column contains

\[
c=(q,f_q,B_q,\mathbf P_D,\mathbf P_{FA},
\mathbf I^{\rm gen},\mathbf I^{\rm rx},\mathbf L^{\rm net},
\mathbf E,\mathbf C^{\rm rx},C^{\rm fus}).
\]

The fleet master chooses exactly one column per target. Its implemented
lexicographic objective is

\[
\operatorname{lexmin}
\left(
d_{\max},
\sum_q d_q,
\sum_q\max_s\mathcal L_{q,s}^{\rm net},
E,
N_{\rm remote},
C
\right),
\]

where \(d_q=[P_D^{\rm req}-\min_sP_{D,q,s}]_+\). Thus detection fairness
cannot be traded for a cheaper network, but statistically equivalent columns
prefer the route that discards less evidence before energy and compute
tie-breaks.

## 5. Algorithmic pipeline

The integrated implementation proceeds as follows.

1. Build feasible bistatic links and discrete acquisition modes.
2. Compute scenario-wise exact-LLR KL or Jeffreys information.
3. Preserve robust singletons, scenario leaders, and complementary pairs.
4. Solve each target–fusion information-pricing problem by branch-and-bound.
5. Optionally screen fusion destinations analytically: retain the top \(K_f\)
   by robust received information plus the destination with the fewest remote
   reports as a locality anchor.
6. Expand retained links into mixed-mode columns and evaluate each with common
   physical-statistic random numbers, fusion-dependent erasure streams, and a
   conservative full-load interference envelope.
7. Record generated evidence, received evidence, network loss, robust
   retention, receiver CPU, and final-fusion CPU for every column.
8. Solve the lexicographic fleet MILP under transmitter energy/load, receiver
   capacity, report, link, final-fusion, and per-UAV CPU constraints.
9. Re-evaluate the selected schedule with independent held-out detector samples
   and report \(P_D@P_{FA}\), not information alone.
10. Sweep RCS and return the first supported pass bracket; flag non-monotone
    outcomes for additional Monte Carlo replication.

The analytical fusion screen reduces expensive detector evaluation but is not
a global certificate over removed destinations. Local pricing may be exact over
its declared pool, and the MILP is exact over generated columns; neither proves
optimality over the complete active-action universe.

## 6. What the current 15-UAV/10-target result says

After decoupling physical evidence from fusion-dependent random streams, the
corrected single-instance run generated 5,850 columns. Active acquisition
obtained mean/worst target \(P_D=0.78951/0.20776\), versus
\(0.77596/0.20776\) for fixed nominal modes, with mean \(P_{FA}\) near 0.05.
The result supports three narrow conclusions:

1. the corrected execution path is stable across fusion destinations when
   report delivery is lossless;
2. active modes can improve mean detection in this instance;
3. fusion and additional candidates do not rescue the bottleneck target here.

It does **not** establish a worst-target advantage, a fleet-population effect,
or a minimum-detectable-RCS improvement. The unchanged worst target is evidence
that the present failure may be formation-limited, but that diagnosis must be
confirmed using the lossless-report oracle with the selected sensing bundle
held fixed.

## 7. Required evidence before promotion

The next formal experiment should use 15 UAVs and 10 targets with paired common
random numbers and at least the RCS grid
\(\{0.05,0.1,0.2,0.5\}\) m\(^2\). At each RCS, compare:

- fixed nominal acquisition with the current single fusion;
- active acquisition with the current transport model;
- the same selected sensing bundle under lossless reporting;
- candidate limits \(K\in\{2,3,4,5,6\}\);
- analytical fusion screening versus all fusion destinations.

Report worst/mean \(P_D\), max/mean \(P_{FA}\), generated and received
information, worst retention, network evidence loss, energy, remote reports,
per-UAV CPU maximum, column count, and wall time. Use paired cluster/bootstrap
intervals across geometry seeds. Promote a low-RCS rescue claim only if the
minimum-detectable-RCS bracket moves downward without violating false-alarm or
resource constraints.

Receiver-local micro-fusion is the next transport extension, not a current
claim. Summing independent exact LLRs locally is algebraically lossless, but an
aggregate packet creates correlated all-or-nothing loss. It must therefore be
paired with an explicit packet-length/reliability or protection model before
being presented as a performance improvement. Two-tier or multi-fusion
architectures should be attempted only if the lossless-report oracle reveals
substantial transport headroom.

## 8. Claim–evidence boundary

| Claim | Present status |
|---|---|
| KL/Jeffreys values follow from the implemented exact LLR | Supported analytically and by tests |
| Observed hypothesis-independent erasure scales information by \(\chi\) | Supported under the declared erasure model |
| CPU is constrained at the actual receiver and final-fusion UAV | Implemented and regression-tested |
| Network evidence loss precedes resource tie-breaks | Implemented and regression-tested |
| Minimum detectable RCS is reported conservatively | Implemented as a bracket diagnostic |
| Active modes improve target-level detection on average | Supported in the existing paired target experiment |
| Active scheduling improves fleet worst-target detection | Not supported by the current corrected instance |
| Fusion rescues the current low-RCS bottleneck | Not yet established; run lossless-report headroom |
| Receiver-local or multi-tier fusion improves performance | Future work |
| Complete fleet action-space optimality | Not supported |

## 中文整合说明

这次整合把论文主线从“协同照射能提升感知”改成“低 RCS 证据救援”。核心指标不再只是平均 \(P_D\)，而是满足指定 \(P_D@P_{FA}\) 时的最小可检测 RCS；模型按证据生成、证据传输、最终融合三层组织；算法仍保留单 fusion 根节点，但会显式计算生成证据、接收证据、网络证据损失和保留率。

代码已修正 CPU 归属：匹配滤波、LLR、DD 精化记在接收 UAV，聚合开销才记在 fusion UAV。全局主问题在检测缺口之后优先最小化证据损失，再比较能量、上报数和计算量。新增的 RCS 工具只给保守区间，不对稀疏测试点做虚假插值。

当前最重要的实验判断不是继续盲目增加候选链路，而是先做 lossless-report oracle。若理想上报仍不能提高最差目标，则问题主要在物理证据生成，应优化 Tx/Rx 组合、模式和功率；若理想上报显著提高，则才有充分理由继续做 receiver-local micro-fusion、保护上报或两层聚合。这样可以避免把 fusion 当成能够凭空补偿小型 UAV 弱回波的模块。
