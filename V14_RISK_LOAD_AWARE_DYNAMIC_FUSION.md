# V1.4：RCS 风险与 CPU 负载感知的动态观测束融合

## 1. 核心结论

V1.4 不再把性能问题归结为单独的 fusion-node ranking，而是统一建模

\[
\boxed{\text{RCS risk}+\text{dynamic bundle}+\text{CPU load}+\text{true-erasure detection}}.
\]

诊断表明，当前主要优化空间来自把每目标观测数从 2 动态放宽到安全上限 6，而不是继续微调融合节点或通信可靠度。

## 2. 系统与观测模型

系统含 $M$ 架 UAV 和 $Q$ 个目标。链路 $k=(i,j,q)$ 表示 UAV $i$ 照射目标 $q$，UAV $j$ 接收回波。统计量送往目标特定融合节点 $f_q$：若 $j=f_q$，它是本地观测；否则经 $j\to f_q$ 报告。

在 $L=K_{\rm look}$ 个独立 look 的 Swerling-II-like 模型下，

\[
X_k|\mathcal H_0\sim\Gamma(L,1),\qquad
X_k|\mathcal H_1\sim\Gamma(L,1+\gamma_k).
\]

完整局部 LLR 为

\[
\ell_k=-L\log(1+\gamma_k)+\frac{\gamma_k}{1+\gamma_k}X_k.
\]

代码原有统计量是其 $H_0$ 中心化形式

\[
\widetilde\ell_k=\frac{\gamma_k}{1+\gamma_k}(X_k-L).
\]

V1.4 显式恢复被中心化移除的常数：

\[
c_k=L\left[\frac{\gamma_k}{1+\gamma_k}-\log(1+\gamma_k)\right],
\qquad
\ell_k=\widetilde\ell_k+c_k.
\]

因此单链路 deflection 为

\[
D_k=L\gamma_k^2.
\]

低 RCS 下 $\gamma_k\propto\sigma_q$，故 $D_k=O(\sigma_q^2)$。融合只能利用已有 sensing information，不能创造信息。

## 3. 真擦除与融合检测器

令 $m_k\in\{0,1\}$ 表示报告成功，且通信事件与目标假设独立：

\[
p(m_k|\mathcal H_1)=p(m_k|\mathcal H_0).
\]

当前主检测器使用中心化 LLR 的 deflection-optimal 线性融合：

\[
F_q=\sum_{k\in b_q}w_km_k\widetilde\ell_k,\qquad
w\propto\Sigma_0^{-1}\delta.
\]

独立模型下 $w_k\propto\delta_k/v_{0,k}$。阈值在真实擦除混合分布下校准。

V1.4 另实现严格 exact-LLR 基线。条件独立时，全局 NP 统计量是

\[
\boxed{\Lambda_q=\sum_{k\in b_q}m_k\ell_k}.
\]

不能仅把中心化统计量的权重设为 1：成功报告集合是随机的，因此 $c_k$ 必须随实际成功报告逐项加入。新路径实现了完整 LLR 抽样及其 $H_0$ 混合阈值校准。

该基线只在 LLR、true erasure、独立观测条件下启用。相关观测需要联合似然模型，不能把 covariance-aware 线性融合称为 exact LLR。

## 4. RCS 区间风险

\[
\sigma_q\in[\underline\sigma_q,\bar\sigma_q],
\qquad
\underline\sigma_q=\eta_\sigma\bar\sigma_q.
\]

固定 bundle 的 sensing SINR 随 RCS 单调增加，因此区间最坏情况位于下端点：

\[
\widehat P^{\rm rob}_{D,qfb}=P_D(q,f,b;\underline\sigma_q).
\]

这是一致的 interval-robust 模型，不等同于 target-specific CVaR。只有在得到可辨识的目标类别或 aspect 先验后，才应以 $p_q(\sigma)$ 升级到 CVaR，避免伪造风险分布。

## 5. 动态 bundle 与 CPU 负载

二元变量 $u_{qfb}$ 表示目标 $q$ 选择融合节点 $f$ 和观测束 $b$：

\[
\sum_{f,b}u_{qfb}=1,\qquad u_{qfb}\in\{0,1\}.
\]

V1.4 关闭人为的“每融合节点最多一个目标”和“每融合节点最多两条观测”限制。bundle 只受安全上限 $|b|\le K_{\rm safe}=6$ 及真实资源约束限制。

非空 bundle 的 CPU 工作量为

\[
C^{\rm cpu}_{qfb}=c_0+c_1|b|+c_2|b|^3,
\]

且每个融合 UAV 满足

\[
\boxed{\sum_{q,b}C^{\rm cpu}_{qfb}u_{qfb}\le F_fT_{\rm proc}}.
\]

当前筛查使用同构 CPU：$F_f=10^9$ cycles/s、$T_{\rm proc}=20$ ms、$c_0=10^6$、$c_1=2\times10^6$、$c_2=0$。这些是可审计的实验配置，不是目标函数中的拟合权重。

## 6. 词典序主问题

检测缺口为

\[
d_{qfb}=[P_D^{\rm req}-\widehat P^{\rm rob}_{D,qfb}]_+.
\]

主问题严格按顺序最小化

\[
\operatorname{lexmin}\left(
\max_qd_q,
\sum_qd_q,
\sum_{qfb}r_{qfb}u_{qfb},
\sum_{qfb}C^{\rm cpu}_{qfb}u_{qfb}
\right).
\]

前两层优化 worst-target 与总检测缺口；后两层仅在检测性能不变时减少报告和计算。这样不会用手调标量把可靠性换成开销。

## 7. 列生成算法

1. 对每个 $(q,f)$ 分别保留高价值 local/remote 候选，避免本地证据族被远端候选淹没。
2. 以空列、单例及 detector-marginal greedy 前缀初始化 restricted master。
3. 先求 worst-deficit LP，再锁定该层并求 total-deficit LP。
4. 从 target、receiver、fusion、report、total-link 与 CPU 约束取得 dual price。
5. pricing 的 CPU reduced-cost 项为 $\pi_f^{\rm cpu}C^{\rm cpu}_{qfb}$。
6. 无负 reduced-cost 列或达到迭代上限后，求四阶段整数 restricted master。

繁忙融合节点因 CPU dual price 自然变贵，不再依赖“每节点最多一个目标”的硬编码。

## 8. Fusion-headroom 诊断

每目标计算：

- $P_D^{\rm local}$：最佳单一本地观测；
- $P_D^{(2)}$：任意融合节点、shortlist 内精确枚举至多两条观测；
- $P_D^{(6)}$：放松共享容量后，以 exact-detector marginal greedy 增长至安全上限 6；
- $P_D^{\rm pc,(6)}$：同一口径下令所有可行远端报告 $\chi=1$；
- $P_D^{\rm fixed,(2)}$：固定融合节点、至多两条观测。

定义

\[
G^{\rm obs}=P_D^{(6)}-P_D^{(2)},\quad
G^{\rm comm}=P_D^{\rm pc,(6)}-P_D^{(6)},\quad
G^f=P_D^{(2)}-P_D^{\rm fixed,(2)}.
\]

$K=2$ 只在声明的 shortlist 上精确枚举；$K=6$ 是受控 greedy 诊断，不宣称全组合数学 oracle。

## 9. 实验结果

### 9.1 Headroom 冒烟诊断

3 个几何、共 30 个目标：

| 指标 | 均值 | 中位数 |
|---|---:|---:|
| $P_D^{\rm local}$ | 0.8439 | 0.9647 |
| $P_D^{(2)}$ | 0.9052 | 0.9966 |
| $P_D^{(6)}$ | 0.9528 | 0.9999 |
| $G^{\rm obs}$ | 0.0476 | 0.00334 |
| $G^{\rm comm}$ | 0.00040 | 0 |
| $G^f$ | 0.00021 | 0 |

平均 observation-count headroom 约为 fusion-location headroom 的 227 倍。因此当前重点仍应是动态互补观测，不是继续细调 fusion ranking。以上数值使用粗粒度调度视图分配固定融合节点，并以配置声明的细化接收机统计量计算 detector headroom。

### 9.2 Exact LLR 消融

MC=20、固定 $K=2$，两种检测器共享完全相同的 bundle 和融合节点决策：

| 检测器 | $P_D$ | $P_{FA}$ |
|---|---:|---:|
| deflection-weighted centred LLR | 0.570 | 0.0531 |
| exact LLR sum | 0.575 | 0.0534 |

exact-minus-deflection 为 $+0.005$，95% 配对区间约为 $[-0.0048,0.0148]$。exact LLR 是理论正确性基线，但当前没有证据把它提升为性能 headline。

### 9.3 RCS-robust + CPU dynamic bundle

冻结配置的独立 MC=200，混合阈值校准样本数 16384：

| 方法 | $P_D$ | trial-cluster 95% CI | $P_{FA}$ | 平均观测数 | CPU 最大利用率均值 |
|---|---:|---:|---:|---:|---:|
| RCS-robust joint bundle CG | 0.6845 | [0.6645, 0.7045] | 0.0513 | 42.91 | 0.9335 |
| nominal joint bundle CG | 0.6285 | [0.6075, 0.6495] | 0.0524 | 35.72 | 0.8970 |
| fixed fusion + joint bundle | 0.6010 | [0.5790, 0.6225] | 0.0519 | 26.82 | 0.9690 |
| local-only joint bundle | 0.5765 | [0.5545, 0.5975] | 0.0510 | 22.03 | 0.8158 |

robust-minus-nominal 为 $+0.0560$，配对 95% CI 为 $[0.0420,0.0700]$；robust-minus-fixed 为 $+0.0835$，区间为 $[0.0649,0.1021]$；robust-minus-local 为 $+0.1080$，区间为 $[0.0898,0.1262]$。CPU 约束在 58.5% 的 robust trials 中达到 99% 利用率，说明负载预算实际参与决策。

robust 的 trial-cluster $P_{FA}$ 区间为 $[0.0495,0.0532]$，覆盖设计值 0.05，correctness gate 通过。弱目标 $P_D$ 为 0.605，与 nominal 相同；因此证据支持总体检测和稳健性提升，但不声称弱目标相对 nominal 已提升。

## 10. 边界与下一步

- 相关性 MC=20 筛查已完成；covariance-aware detector marginal 已进入 pricing，因此暂不叠加人为 diversity penalty。
- 统一 RCS 下界不是 target-specific CVaR；没有可靠先验前不伪造分布。
- headroom 的 $K=6$ 路径不是无限观测的全局组合最优。
- MC=200 支持把 V1.4 作为后续候选版本；headline 身份仍应与论文主结果迁移分开管理。
- 细化表触发条件修复后，MC=200 主结果与相关性筛查逐字段复核一致；headroom 与 detector 消融已按统一细化口径更新。
- 若 MC=200 仍显示 $P_D^{(6)}\approx P_D^{\rm local}$，则系统属于 sensing-limited，应转向更多 look、sensing power 或几何设计。

### 10.1 相关性筛查结果

| 总相关强度 | $P_D$ | 相对 $\rho=0$ 的配对差 | 95% CI | 中位 deflection | 平均观测数 |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.685 | 0 | [0, 0] | 11.19 | 42.35 |
| 0.3 | 0.660 | -0.025 | [-0.0697, 0.0197] | 8.30 | 40.55 |
| 0.6 | 0.670 | -0.015 | [-0.0583, 0.0283] | 7.41 | 39.70 |

相关性显著降低可加信息量，但 MC=20 尚未证明总体 $P_D$ 显著下降。观测数随相关强度增加而减少，说明 covariance-aware bundle value 正在主动拒绝冗余观测。下一步若扩展，应提高相关性实验样本量并标定相关系数，而不是再引入一个手调 diversity reward。

## 11. 复现

    python -m pytest -q
    python tools\run_fusion_headroom_v14.py --mode headroom --headroom-trials 3 --calibration-samples 16384 --out results_fusion_headroom_v14_refine_fix
    python tools\run_fusion_headroom_v14.py --mode detector --mc 20 --workers 4 --detector-k 2 --calibration-samples 16384 --out results_fusion_headroom_v14_refine_fix
    python tools\run_fusion_headroom_v14.py --mode correlation --corr-mc 20 --workers 4 --calibration-samples 16384 --out results_fusion_headroom_v14_refine_fix
    python tools\run_fusion_headroom_v14.py --mode main --mc 200 --workers 4 --calibration-samples 16384 --out results_fusion_headroom_v14_refine_fix

实现入口：isac_sim/llr.py、isac_sim/soft_channel.py、isac_sim/fusion.py、isac_sim/fusion_headroom.py、isac_sim/bundle_master.py 与 tools/run_fusion_headroom_v14.py。

每个结果文件都有同名的独立有效配置清单（`*.config.json`）；后续以不同 mode 写入同一目录时，不再依赖会被覆盖的通用 `config.json`。如只需补建配置清单，可在相同命令末尾添加 `--manifest-only`，不会执行仿真或改写 CSV。
