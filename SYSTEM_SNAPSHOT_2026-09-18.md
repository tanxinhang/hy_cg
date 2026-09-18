> **归档提示（2026-09-18）**：本文引用的部分 `results_*` 产物已移入 `_archive/2026-09-18/`；正文中的路径引用已同步更新为归档位置，命令行示例里的 `--out` 目录仍写作历史原名（重跑时依旧输出到该名）。

# 系统性能与环境快照

> 生成时间：2026-09-18 01:30
> 定位：把散落在 `SYSTEM_PERFORMANCE_STATUS.md`（2026-09-16，**不含协调线**）、
> `PERFORMANCE_OPTIMIZATION_ROADMAP.md`、`COORDINATION_INTERFERENCE_ROUTE.md` §11+§13、
> `results/coordination_experiment*/` 的数字收敛到一页，便于对外引用。
> **数值权威源**：论文工作点 → `_archive/2026-09-18/results_target_local_v1/main/main.csv`；
> 协调前沿 → `results/coordination_experiment_precision/*.csv`（九档 800-trial）；
> κ 扫描 → `results/coordination_experiment_kappa/*.csv`（四档 800-trial）。
> 复算哈希见 `results/coordination_experiment/RUN_LOG.md` §5.1 与 §6.7。

---

## 0. 一句话

**论文工作点（4 km / RCS ≥ 20 m²）达标且通信极省（P_D 0.9764，报告 0.751 条、0.80 ms，是最强
启发式的 1/6 代价）；但换到小目标（RCS ≤ 0.2 m²）全面不达标。** 有效的补偿只有两条，
且**互为替代品、不可叠加计功**：① 机间协调（把不参与观测的照射机静默）——把达标所需积分时间
从 16× 压到 3×；② 加深直射对消（κ_dc 40 → 60 dB）——1× 时间下单臂即可达标。

---

## 1. 三个工作点必须分开说（混用即不可比）

| 工作点 | 场地 | RCS | 用途 | 状态 |
|---|---|---|---|---|
| **论文主口径** `target-local-v1` | 4 km | 50 m² | 论文主结果 | ✅ 达标 |
| **协调实验工作点** | 500 m | 0.2 m² | 协调/对消实验 | ⚠️ L=16 下不达标，需协调或加深对消 |
| **小目标缺口工作点** | 400 / 600 m | 0.05–0.2 m² | 链路预算诊断 | ❌ 任何软件旋钮都不够 |

⚠️ **「干扰受限」是几何条件量，不是系统属性。** `rinr = residual/n0` 中位数随部署尺度翻转：
4000 m **−2.8 dB**（噪声受限）／2000 +0.6／1000 +5.0／800 +6.3／400 **+10.6 dB**（干扰受限）。
⇒ 论文工作点（4 km）其实**噪声受限**，ROADMAP 里 400 m 的分母类饱和结论**不能搬过去**。
**任何「某旋钮无效」的结论，先声明 r。**

---

## 2. 论文工作点性能（MC=1000，seed=2026，4 km / RCS 50 m² / 15 UAV / 10 目标）

| 方法 | P_D | worst-target | weak-target | P_FA | 报告数 | 时延 (ms) | 比特 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **proposed_c2f_adaptive_pd** | **0.9764** | **0.966** | 0.942 | 0.0487 | **0.751** | **0.80** | 481 |
| exact_marginal_greedy | 0.9408 | 0.931 | 0.897 | 0.0493 | 5.131 | 5.47 | 3284 |
| sense_sinr | 0.9784 | 0.972 | 0.925 | 0.0494 | 4.595 | 4.90 | 2941 |

列名对应（`_archive/2026-09-18/results_target_local_v1/main/main.csv`）：`P_D` = 跨 trial 平均 = `actual_mean_target_P_D`；
**worst-target** = `actual_worst_target_P_D`；**weak-target** = `P_D_weak`。
⚠️ **`P_D_weak` ≠ worst-target**，是两个量（前者是弱目标跨 trial 均值，后者是跨目标取最坏后再跨 trial 平均），
引用时必须说清，别用同一个词指两者。

- vs `exact_marginal_greedy`：**Δ = +0.0356**，CI [0.0311, 0.0401] —— 不含 0，显著。
- vs `sense_sinr`：**Δ = −0.0020**，CI [−0.0050, +0.0010] —— 含 0，**统计上不可分**。
- 其他：`all_targets_satisfied_prob` = 0.726（proposed）/ 0.388（greedy）；`worst_target_satisfied_prob` = 0.955。

⚠️ **可追溯性备注（本次新查）**：`main.csv` 的 `paired_proposed_delta_P_D` 等 **7 个配对列全为空**
（`full-refinement-pd` 与 `prediction-stress` 亦同）⇒ 论文头条的配对差值**不在预计算产物里**。
但**可从 `trials.csv` 逐 trial 独立复算**（1000 trials × 3 方法），实测：
`+0.0356`，trial 级 bootstrap 95% CI **[+0.0310, +0.0402]**；`−0.0020`，CI **[−0.0050, +0.0010]**
⇒ **与论文值一致**（前者差异在 bootstrap 噪声内）。复现命令见 §9。
**投稿前建议**：把配对列回填进 `main.csv`，否则审稿人从冻结产物里看不到那个 CI。

**⇒ 唯一正确的卖点是通信效率（代价降到 1/6），不是检测增益。**
说「提升了检测性能」会被 `sense_sinr` 那一行反驳。

**RCS 轴（4 km，K=0）**：0.05→0.083｜0.1→0.113｜0.2→0.167｜1→0.434｜5→0.769｜20→0.940｜50→0.981
⇒ 达标（≥0.95）需 RCS ∈ (20, 50] m²。协同增益呈**倒 U 形**（峰值在 1 m²，+0.037）。

---

## 3. 协调前沿：九档 800-trial（500 m / RCS 0.2 / L=16 为 1×）

统计量 = **跨 4 个部署种子的 worst-target P_D 的 MIN**（本库全部门槛/增益都是 MIN 口径；
同一文件里的 MEAN 是另一个数，别混用）。每格 800 trials，**全部 800/800 收敛**。

| L（时间） | uncoordinated | mask_only | sparse | P_FA 范围 |
|---|---:|---:|---:|---|
| 16 (1×) | 0.7800 (0/4) | 0.8150 (0/4) | 0.8950 (0/4) | 0.0481–0.0504 |
| 32 (2×) | 0.8300 (0/4) | 0.8600 (0/4) | 0.9350 (1/4) | 0.0485–0.0503 |
| 40 (2.5×) | 0.8350 (0/4) | 0.8850 (0/4) | 0.9450 (2/4) | 0.0479–0.0510 |
| **48 (3×)** | 0.8450 (0/4) | 0.9150 (0/4) | **0.9550 (4/4)** ✅ | 0.0479–0.0511 |
| 56 | 0.8700 (0/4) | 0.9250 (0/4) | 0.9550 (4/4) | 0.0480–0.0508 |
| 64 (4×) | 0.8850 (0/4) | 0.9300 (0/4) | 0.9700 (4/4) | 0.0484–0.0510 |
| **96 (6×)** | 0.9200 (0/4) | **0.9500 (4/4)** ✅ | 0.9650 (4/4) | 0.0486–0.0508 |
| 144 (9×) | 0.9400 (3/4) | 0.9600 (4/4) | 0.9750 (4/4) | 0.0484–0.0507 |
| **256 (16×)** | **0.9550 (4/4)** ✅ | 0.9750 (4/4) | 0.9850 (4/4) | 0.0486–0.0509 |

**三条达标路径**：`sparse` **L=48（3×）**｜`mask_only` **L=96（6×）**｜`uncoordinated` **L=256（16×）**
⇒ **协调 = 5.3 倍积分时间节省**；门槛区间 **(40, 48]**；P_FA 27 臂全在 0.0479–0.0511（标称 0.05）。

**协调增益序列**（sparse − uncoordinated）：
`+0.1150 / +0.1050 / +0.1100 / +0.1100 / +0.0850 / +0.0850 / +0.0450 / +0.0350 / +0.0300`
—— L≤48 基本平（跨度 0.010 = 2 格），**L≥48 后单调不增（仅 56/64 持平）**。

**机制权重翻转**：L≤40 重调度 ≥ 静默；**L≥48 静默主导**（翻转点 L=40→48）。
**效率陈述**：协调 1×（0.8950）≈ 不协调 **4.4×**（0.8850）——略高于 4×，且它本身不花积分时间。
（旧「≈3.3×、不及 4×」是 200-trial 低估造成的**假象，已反转**。）
⚠️ **5.3× 与 4.4× 不矛盾**：前者是**门槛比值**（16/3），后者是**等值点**；不协调臂在高时间档饱和更快，
故比值随目标水平上升。**引用时必须点明用的是哪一个。**

---

## 4. 协调 × 对消深度：替代品（判定预注册，2026-09-18 定案）

| κ_dc (dB) | uncoord | mask_only | sparse | **协调增益 g** | uncoord 达标 | 中位 rinr |
|---|---:|---:|---:|---:|---|---:|
| **40（发布值）** | 0.7800 | 0.8150 | 0.8950 | **+0.1150** | 0/4 | +10.87 dB |
| 50 | 0.9400 | 0.9550 | 0.9500 | +0.0100 | 1/4 | +1.64 dB |
| 60 | 0.9500 | 0.9550 | 0.9500 | **+0.0000** | **4/4** | **−4.19 dB** |
| 80 | 0.9500 | 0.9500 | 0.9500 | **+0.0000** | **4/4** | −5.82 dB |

- **VERDICT: SUBSTITUTES**。塌缩 `g(40)−g(80)` = **+0.1150** ≫ 阈值 0.05；跨界在 **κ=50→60**
  （**四个种子全部一致**，κ=50 全正、κ=60 全负）。
- 交互量 = (+0.1700) − (+0.0550) = **+0.1150** ⇒ **不可叠加计功**。
- **硬证据**：κ=80 时 `mask_only` 与 `uncoordinated` 在 **4/4 种子上 `worst_pd`/`mean_pd` 逐位相同**，
  而 `#TX` 差 **+10.9**（15.0 → 4.1）—— 关掉 15 台里的 11 台辐射机，检测统计一字不变。
  且**不靠 0.95 天花板**（12 格无 `[[CEILING]]`）。
- 交叉校验：κ=40 档与九档前沿 **L=16 行逐格重合**（`rounds 3.3925` vs 3.39）⇒ 两批同口径。
- 副产物：**κ≥50 时 `mask_only` ≥ `sparse`**（对消压低残余后「重新选择照射机」归零）；
  **κ=60 时 1× CPI 的不协调单臂即 4/4 达标**（种子 303 恰好 0.9500、零余量）。

---

## 5. 为什么还差：杠杆的饱和性（`gap` = 400 m / RCS 0.05 最弱目标抬到 0.80 还差多少 dB）

| 族 | 成员 | 买到 dB（base gap 7.90） |
|---|---|---:|
| **分子类**（精确线性，不饱和） | RCS、几何、`G_proc = N·L`、`G_hw` | `G_proc`×4→3.82；`G_hw`+10→3.23；RCS×10→5.85 |
| **分母类**（饱和，**别写**） | `P_default`、`noise_figure_db` | P×100→**0.31**；NF 7→0→**0.25** |
| **减分母类** | `kappa_dc`、照明机调度 | κ40→50→4.27；40→60→**5.47**（80 只多 0.17） |
| **样本数类**（√L） | `detect.n_looks` | looks 16→64→**2.16**；16→256→4.05 |

- **优化是有序的**：κ 40→60 把 r 压到 0.35 后，`P×100` 才从 0.31 变成 **1.66** dB。**先对消，再提功率。**
- **最小闭合组合**：`G_proc×4 + κ60 + looks64` → gap **0.37**（几乎闭合）。
- **算法侧已到顶**：`maxmin` +0.206 是唯一有效旋钮；budget 轴用尽；`corr` 负收益；
  **lossless oracle headroom ≡ 0** ⇒ 算法只能**重分配**预算，造不出预算。
- ⚠️ 第 5 节全部是 **LLR 矩的解析界**，不是 MC 的 P_D，**只可用于排序与定量缺口**。

---

## 6. 环境设置

### 6.1 运行环境

| 项 | 值 | 备注 |
|---|---|---|
| OS / 平台 | Windows 10 (10.0.19041) | Bash 走 PortableGit 1.2.0 |
| Python | **3.11.0**（`E:\anaconda\3_11_python\python.exe`） | 托管版 3.13.12 备用；Windows Python **不认 MSYS 路径** |
| numpy | 2.2.6 | |
| CPU | **16 核** | 单档独占跑用 `OMP_NUM_THREADS=8`；四档并行各 4 线程 |
| Node | 22.22.2（托管） | |
| `OMP_NUM_THREADS` | **未设**（默认） | 逐次命令里显式指定；实验对线程数**不敏感**（RUN_LOG §6.7） |
| 系统代理 | 当前 `127.0.0.1:9504` | ⚠️ **端口每会话变**（曾见 24638）。对 github 有害（502）⇒ push 前 unset 并改指 `127.0.0.1:7890`（该通道仍在监听） |
| 长跑纪律 | `python -u` + `>> log 2>&1` | **绝不用 `\| tee`**（块缓冲 ⇒ 被杀时日志 0 字节）；MC ≥ 100 必须后台 |

### 6.2 配置（论文口径 vs 协调实验，逐字段实测导出）

| 字段 | 论文口径 | 协调实验 uncoord | 协调实验 sparse |
|---|---|---|---|
| `geometry.area_xy` (m) | 4000 | **500** | **500** |
| `detect.target_rcs` (m²) | 50 | **0.2** | **0.2** |
| `detect.n_looks` | 16 | 16 | 16 |
| `interference.direct_cancellation_db` | 40 | 40 | 40 |
| `interference.sense_gate_by_active_tx` | **False** | **True** | **True** |
| `selector.tx_penalty` | 0.0 | 0.0 | **0.1** |
| `selector.max_tx_nodes` | None | None | None |
| `comm.interference_model` | orthogonal | orthogonal | orthogonal |
| `radio.rho` | 0.8 | 0.8 | 0.8 |
| `run.num_mc` | 200（主结果用 1000） | 200 | 200 |

**sparse 与 uncoord 的唯一差异字段 = `selector.tx_penalty`**（实测核对），
其余物理量逐位相同 ⇒ 增益可归因于**协调代价项**，不是别的旋钮被动了。

两套工作点**共有**的部分：`scale.M=15`（无人机）、`scale.Q=10`（目标）、
`waveform.N = waveform.L = 64`、`delta_f = 30 kHz`、`fc = 5.9 GHz`、`T = 33.3 µs`、
`radio.noise_figure_db = 7`、`noise_psd_dbm_hz = −174`、`radio.P_default = 1.0`、
`radio.eps_mode = noise_relative`、`detect.pd_required = 0.95`、`detect.weak_pd_required = 0.80`、
`fusion.mode = explicit`、`fusion.rule = nearest_target`、`radio.radar_net_gain_db = None`。

**发布门禁**：`tools/check_release_identity.py --check` ⇒ **CLEAN**（94 冻结键 0 违反）。
⚠️ `interference.direct_cancellation_db` 在**冻结键**内、**不在** 16 条可标定键里
⇒ **κ=40 是发布值本身**，κ=50/60/80 是**带标签的变体**（走 override，默认路径逐位不变）。

---

## 7. 引用前必读的四条口径纪律

1. **统计量**：本库门槛/增益一律是「跨部署种子的 worst-target P_D 的 **MIN**」。
   同一脚本可能对同一列名给出两种统计量（`run_coordination_experiment.py`：表列 `mean`、
   头条 `min`）——**跨节引用前先钉死口径**，否则会看起来像矛盾。
2. **`P_D` 与 `gap` 不同构**：`gamma` 买了多少 dB ≠ 缺口降了多少 dB（`d'` 对 γ 是凹的）。
3. **解析界 ≠ MC**：第 5 节只能排序，不能当检测性能宣称值。
4. **口径 = 物理假设**：`orthogonal`（论文主结果，比 `active_set` 高 ~0.038 P_D）
   / `full_concurrent`（最坏） / `active_set`（消融）。**凡功率结论必须声明口径。**
   另有枚举值静默退化陷阱：传 `"Orthogonal"` 会退化成 `full_concurrent`。

---

## 8. 未收口（不影响本页结论）

- **协调信令开销 / 失效机制未建模** ⇒ 论文必须声明为假设。
- **门槛若要写到单档**需补 L=44（40→48 是 2/4→4/4 跃迁，边际价值低）。
- **融合栈三处逻辑不自洽**、系统单圈闭环缺时间划分 τ、`distributed_bids` 零调用、
  `ABLATION_VARIANTS["full"]` 是空覆盖 ⇒ 详见 `SYSTEM_INTEGRITY_AUDIT_2026-09-16.md`。
- `kappa_dc` 的标定张力（注释称对消到回波量级，实测残留直射比 raw 回波高约 47 dB）
  未追到底 ⇒ 引用 κ 前应先解决。
- **全部协调工作尚未提交**（HEAD `acdcacc`）。

---

## 9. 复现

```bash
export PATH="/c/Users/Administrator/.workbuddy/binaries/PortableGit/versions/1.2.0/usr/bin:/c/Users/Administrator/.workbuddy/binaries/PortableGit/versions/1.2.0/mingw64/bin:$PATH"

# 论文工作点主结果（MC=1000，数小时，务必 run_in_background）
E:/anaconda/3_11_python/python.exe tools/rerun_target_local_v1.py --out results_target_local_v1

# 协调前沿九档（每档 800 trials；单档独占用 OMP_NUM_THREADS=8）
E:/anaconda/3_11_python/python.exe -u tools/run_coordination_experiment.py \
  --mc 200 --seeds 4 --looks 48 --penalty 0.1 --rounds 12 --out results/coordination_experiment_precision

# 对消深度扫描四档 + 事后合并判定（不重跑 MC）
E:/anaconda/3_11_python/python.exe -u tools/probe_coordination_kappa.py \
  --kappa 40 50 60 80 --mc 200 --seeds 4 --penalty 0.1 --looks 16 --rounds 12 --no-summary \
  --out results/coordination_experiment_kappa
E:/anaconda/3_11_python/python.exe -u tools/probe_coordination_kappa.py \
  --report-only --kappa 40 50 60 80 --out results/coordination_experiment_kappa

# 环境与配置自检
E:/anaconda/3_11_python/python.exe tools/check_release_identity.py --check      # 期望 RESULT: CLEAN
```

**从 `trials.csv` 复算配对差值**（无需重跑 MC）：

```python
import csv, numpy as np
rows = list(csv.DictReader(open('_archive/2026-09-18/results_target_local_v1/main/trials.csv')))
d = {}
for r in rows:
    d.setdefault(r['method'], {})[int(r['trial'])] = float(r['detected_fraction'])
t = sorted(d['proposed_c2f_adaptive_pd'])
p = np.array([d['proposed_c2f_adaptive_pd'][i] for i in t])
b = np.array([d['exact_marginal_greedy'][i] for i in t])
x = p - b
rng = np.random.default_rng(0)
idx = rng.integers(0, len(x), (10000, len(x)))
lo, hi = np.percentile(x[idx].mean(axis=1), [2.5, 97.5])
print(f'{x.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]')   # 期望 +0.0356 [+0.0310, +0.0402]
```

长跑纪律：`python -u` + 直接 `>> log 2>&1`（**不用 `| tee`**，块缓冲会让日志在被杀时留 0 字节）；
MC ≥ 100 必须后台；并发批次按**最慢档**算墙钟。

