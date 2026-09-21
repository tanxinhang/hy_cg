# 版本迭代开发规范（DEV_STANDARD）

> 适用：`isac_sim` / `experiments` 仓库（`D:\Desktop\conference`，ICC 2027 投稿线）。
> 生效：V1（release `V1_STABLE_MANIFEST.json`，**95** 冻结键 —— κ 删除后由 96 减一）之后的所有迭代。
> 用途：任何改动在动手前先按 §1 分诊，按 §9 checklist 逐卡通过，再提交。
> 配套：**`docs/TEST_STANDARD.md` 是测试书写规范的唯一细节口径**（§7 只给指针）。

---

## 0. 三条不可协商的口径

| 编号 | 口径 | 理由 |
|---|---|---|
| **R1** | **默认发布路径必须逐位不变**。新能力一律"未登记新键 + 默认关闭"，改已有冻结键的**值** = 宣布新假设。 | 论文数字与代码一旦错位，全部结果作废 |
| **R2** | **先定判据，再跑实验**。判据写进脚本 docstring，跑前做常数一致性自查。 | 本轮两次结论反转都源于"跑完再解释" |
| **R3** | **算法不改物理量**。能量、噪声、距离、门限属于模型；调度、排序、搜索属于算法。 | 用算法掩盖物理缺口 = 虚假增益 |

---

## 1. 变更分诊（动手前必做）

任何改动先归到下面四类之一，然后**只走该类对应的流程**。

| 类别 | 判据 | 必经流程 | 示例 |
|---|---|---|---|
| **A · 可标定** | 键已在 `calibratable`（18 项） | ① 跑门禁 ② 记录数字 | `run.num_mc`、`geometry.area_xy`、`target_rcs` |
| **B · 门控新功能** | **新增未登记键**，默认关闭 | ① 默认路径逐位不变证明 ② `check_release_identity` CLEAN ③ 单测覆盖开关两侧 | `selector.maxmin_objective`、`leximin_rho` |
| **C · 假设变更** | 改 FROZEN 96 键的值 / 换目标函数形状 / 换检测口径 | ① 显式声明"新假设" ② 重跑全部受影响结果 ③ 更新 manifest + 打 tag | `softmin_tau`、`lambda_c`、`score_mode`、κ |
| **D · 架构重构** | 不动数值，只动结构 | ① 前后数值逐位快照 ② 模块行数门禁 ③ 分层边界测试 | 大文件拆同名包、注释中文化 |

⚠️ **"改目标函数形状"与换 κ 同级，属于 C 类，不是 B 类**。
⚠️ 分诊不清时按**更高**类别处理（宁可多跑门禁）。

### 1.1 分诊之前：先过"四问"（2026-09-21 定，两个方向都栽过）

分诊只决定**流程**，不保证这个改动**有东西可改**。动手前先回答：

| # | 问 | 典型坑与后果 |
|---|---|---|
| **Q1** | **preset 已经把这个开关打开了吗？** | `paper-canonical` 已开 `soft_stat_model=llr`、`rcs_model=mean`、`score_mode=exact_utility`、`stop_at_D_min=False`、`fusion.mode=explicit` ⇒ 再"打开"是**空操作**（方向 2 一次砍掉 4 个臂） |
| **Q2** | **它是死参数吗？（grep 有没有被读 / 被传）** | `sensing_power_scale_by_uav`（零处传非 None）、`cancellation.search_half_width`、`cancellation.n_cpi`（接不到对消臂） |
| **Q3** | **单世界还是双世界？** | `belief_mode=True` 下生产是**两套表**（belief 给调度、truth 给检测 + `truth_captured_links`）。只建一套 = 静默跑在完美先验世界 ⇒ "加链路"机制上必然 **0.0000**，而双世界下是 **+0.042** |
| **Q4** | **分辨力够吗？** | 配对差 sd ≈ **0.122** ⇒ MC=20 只能分辨 ≥0.10，分辨 0.05 需 **n≥53**。n≤3 的"强信号"基本是噪声（实测 n=3 报 +0.2222、n=20 只有 +0.050） |

任一问答不上来，先补诊断，**不要直接开跑**。

---

## 2. 发布身份与门禁

### 2.1 冻结资产

| 资产 | 位置 | 规模 |
|---|---|---|
| 发布清单 | `release/V1_STABLE_MANIFEST.json` | frozen **95** / calibratable **18** / rng_offsets 28 / preset_registry 13（κ 退役后 96→95） |
| 头条预设 | `target-local-v1` | — |
| 基线 CSV | md5 两枚（`baseline_csv_md5` / `release_baseline_csv_md5`） | — |

### 2.2 门禁（每次提交前全跑）

```bash
PY=E:/anaconda/3_11_python/python.exe          # 逐位门禁只在这个环境下有结论
$PY -m pytest -q                                # ① 单元/契约（含 ⑤⑥ 两道自研门禁）
$PY tools/check_release_identity.py --check     # ② 95 冻结键，必须 CLEAN 95/95
$PY tools/check_contract_refs.py                # ③ 契约引用区间，必须 32/32
$PY tools/parity_check.py                       # ④ 重构逐位一致性（D 类必跑）
```

① 里除了常规契约，还含两道**自研门禁**（2026-09-21 立）：

| 编号 | 门禁 | 钉什么 |
|---|---|---|
| **⑤** | `tests/test_test_inventory.py` + `tests/_test_inventory.py` | **防丢**：拆大文件时不得静默丢 `def test_*`（AST 比对 155 个基线名） |
| **⑥** | `tests/test_study_docs_hygiene.py` | **归档卫生**：不得新增乱码文档、md 引用的 `data/<dir>` 必须存在、`scripts/` 每脚本须被 README 索引 |
| **⑦** | `tests/test_test_file_hygiene.py` | **测试自身的结构卫生**：模块 docstring、ASCII 命名、`__main__` 守卫、脚手架 `_` 前缀、禁止从 `test_*` import（`docs/TEST_STANDARD.md` §1） |

**当前基线（2026-09-21）**：`pytest -q` ⇒ **716 passed / 7 xfailed / 6 subtests**，≈162 s；
`tests/` 47 个文件、**0 个超 350 行**；`check_release_identity` CLEAN（95/95）；
`check_contract_refs` CLEAN（32/32）。
跑出偏离先做环境与口径核对（见下），别直接归因于代码。

⚠️ **逐位门禁是环境绑定的**：只在 **python 3.11.0 + numpy 2.2.6** 下有结论。换环境跑出的
"不一致"先做环境核对，别直接归因于代码。
⚠️ CI 跑不了逐位门禁（基线在 `.gitignore` 排除的 `.workbuddy/`）。CI 现为三道环境无关门禁
（发布身份 / 契约引用 / TP-UIC 契约）+ 显式声明未覆盖项。
⚠️ `check_contract_refs.py` 的**区间必须手改**（符号出现在别处不算契约），报错会给"符号实际行号"。

### 2.3 判定"推送成功"

```bash
git -c http.proxy=http://127.0.0.1:<PORT> -c https.proxy=http://127.0.0.1:<PORT> push origin main
git ls-remote origin refs/heads/main     # 必须实查
```

- 端口**每次先探**（`git ls-remote` 试）；历史可用过 `7890`，`38604`/`5639` 已失效。
- **别信 `git status -sb`**：本工作区不落盘 remote-tracking 引用，推完也会显示
  `[origin/main: gone]`——那是环境现象，不是推送失败。

---

## 3. 代码结构规范（积木化）

| 编号 | 规则 | 门禁 |
|---|---|---|
| **S1** | `isac_sim/` 单文件 **≤150 行** | `tests/test_module_size.py::MAX_LINES` |
| **S2** | 拆大文件优先做**同名包**（`x.py` → `x/__init__.py` + 子模块），调用点不动 | — |
| **S3** | 注释与 docstring **中文化** | 人工复查 |
| **S4** | **禁止** `sys.modules` 别名 / 兼容层 / `try: import ... except` 兜底 | `tests/test_layer_boundaries.py` |
| **S5** | 测试文件 **≤350 行**（`tests/` **全目录纯红线，无豁免**），一条链路一个文件 | `MAX_TEST_LINES` + `test_test_modules_are_at_most_350_lines` |
| **S6** | 共用脚手架放 `tests/_<topic>_common.py`（pytest prepend 模式可直接 import）；**禁止从 `test_*.py` import** | `docs/TEST_STANDARD.md` §1 |
| **S7** | 每个优化方向独占 `studies/directionN/`：`README.md`（唯一入口+汇总数字）/ `scripts/`（一次性探针，不进 CI）/ `data/`（实测）/ `docs/`（审计） | `tests/test_study_docs_hygiene.py` |
| **S8** | 含中文的 md **禁止用 PowerShell `Get-Content \| Out-File` 搬运**（按 GB18030 解码 ⇒ 双重编码） | 同 S7 的乱码检测 |

⚠️ **按顶层 class 自动切分会丢顶层 `def`**（切丢过辅助函数 `_state`）——切前先把
顶层函数搬进脚手架或库里。
✅ 历史超限测试文件（6 个，最大 825 行）已按主题拆完 ⇒ **豁免取消**，现在 `tests/` 是纯红线。
拆分的完整流程与防丢纪律见 `docs/TEST_STANDARD.md` §5、`studies/TEST_SPLIT_2026-09-21.md`。

---

## 4. 实验与证据规范

### 4.1 必做清单

| 编号 | 规则 |
|---|---|
| **E1** | **判据写进脚本 docstring**，跑前定死 PASS/FAIL 阈值与成本口径（P_D、P_FA、obs、bits 同列） |
| **E2** | **常数一致性自查**：预注册判据里的常数互相是否自洽（曾出现 `RHO_PASS=0.9` ⟺ 需要 2.0 BW = `BW_FAIL`，规则必然 FAIL） |
| **E3** | **诚实口径**：调度器用 belief 选、评估必须用 truth 评（`simulate.py:988–1012` 用 `eval_base`/`eval_tables`/`selected_eval` **重算** D）。禁止 belief 选 + belief 评当结果 |
| **E4** | **跨 run 一律用同法名边际差**。⚠️ 配对 ref 陷阱：19 方法跑用 `proposed_c2f` 当基准，与 3 方法基线的 `adaptive_pd` 混用会让结论反向 |
| **E5** | 检验统计量分布不确定时，**偏好门限无关的 AUC** 而非固定门限 P_FA |
| **E6** | 每个 feature 干预必须配 **oracle / upper-bound** 对照（同 run 配对） |
| **E7** | **报绝对值必须报跨场景分布**，不能只报 `mean ± se`（见 §4.3） |
| **E8** | **跑前声明分辨力**：按配对 sd≈0.122 估算 MC —— MC=20 只能分辨 ≥0.10，分辨 0.05 需 n≥53。**低于门槛的扫描不得出定论**，只能当诊断（§1.1 Q4） |
| **E9** | **报任何端到端数字必须同时报口径**：κ 取值 / 记账口径（`residual_accounting`）/ δ / `production_wire` 开关。融合类收益**必须先报 κ 区间**（§5.7） |

### 4.3 数字报告口径（方差审计后新增，`TRIAL_VARIANCE_AUDIT.md`）

实测（M=15/Q=10，32 场景）：跨场景 sd **0.127**，而 `mean` 的 naive 标准误只有
**0.023** ⇒ 只报 `± se` **低估不确定性 5.6 倍**，且 MC→∞ 也不收敛（场景运气 +
检测抽样都是结构性的）。

| 报告对象 | 正确报法 |
|---|---|
| **绝对值** P_D | 中位 + 跨场景 [P10, P90] + 场景数 n（例：0.60，[0.50, 0.80]，n=32） |
| **配对差** ΔP_D | 仍报 `mean ± s.e.`（同批场景配对，场景方差已被消掉） |

⚠️ **方差的载体是「摇摆目标」**：固定场景只换检测流，Q=10 里通常 3 个恒检出 /
4 个恒漏检 / **3 个摇摆**，`Var(P_D) ≈ Σ_q p_q(1−p_q)/Q²`。
⇒ 压缩 trial 间离散的手段是**让每目标 D 远离门限**，**不是加 MC**。

### 4.2 性能根因审计的专用口径（本轮定死）

| 项 | 口径 |
|---|---|
| 主工作点 | 600 m / RCS 0.1（`--preset target-local-v1` + `--area 600 --rcs 0.1`） |
| 融合建模 | **只建模相关性**（ρ_tx / ρ_rx / ρ_target / ρ_dd），**不做相干合并** |
| 同步/时钟误差 | **不建模**（避免复杂度膨胀），不进相关性来源 |
| 融合有效链路数 | 分母是**最佳单条 D**，不是被 belief 拖低的 D ⇒ corr=False ≈6.4、corr=True ≈5 |
| 报告字段 | P_D、P_FA、**D 中位**、**obs 均值**、**bits 均值** 必须同表给出 |

⚠️ **"选得多"与"选得准"是替代关系，不是叠加关系**：诚实曲线 k=1→40 的 P_D 是
0.530→0.926，但每一点都在花通信预算。任何"买更多"的方案**必须**与"降 σ"对比后再选。
⚠️ **`top-1 命中率`是错指标**：本轮鲁棒排序把命中率从 24.2% 掉到 9.2%，端到端 P_D 反而更高。

### 4.4 端到端扫描的口径前置条件（2026-09-21 定死）

🔴 **主链路默认没有直连对消**：κ 只来自 `_trial_certificate_scope`，受
`cancellation.production_wire` 门控，**默认 `False` ⇒ `kappa_dc = 1.0`**；40 dB 常数已物理删除、
无替代品 ⇒ 默认 `run_one_trial` 在 600 m / RCS 0.1 下 **D_fuse ≈ 1e-6、P_D = 0.0000**。

⇒ **任何端到端扫描必须先显式开 `production_wire=True`**，否则所有臂躺在地板上、Δ 恒为 0，
会被误读成"融合已饱和 / 干预无效"（方向 2 第一版全 0 就是这么来的）。
测量约 26 s/trial（装配剪枝 + 联合系统缓存后）⇒ 用"每 trial 一次、跨臂共享证书"的缓存，
与生产同语义；**缓存 key 必须含 `residual_accounting` 与 `sigma_*`**（证书随口径变）。

---

## 5. 增益准入制度（防虚假增益）

### 5.1 什么算真实增益 —— 三条同时满足才算

| 编号 | 判据 |
|---|---|
| **G1** | 在**诚实口径**下测得（E3），P_FA 不超基线 +0.01 |
| **G2** | **有实现支撑**：增益来自仓库里真实存在的代码路径，不是注入的常数 |
| **G3** | **成本同列**：obs / bits / 复杂度变化必须与 ΔP_D 一起报告 |

### 5.2 已判定为虚假增益的清单（不得再进归因、不得给改进项）

| 项 | 名义收益 | 判定理由 |
|---|---|---|
| **G_hw（`radio.radar_net_gain_db`）** | +10 dB ⇒ +0.185 | 发布值 **0 dB**；27 dB 只属 `small-uav-link-budget-bridge` 标定桥，无硬件设计支撑 |
| **κ（`interference.direct_cancellation_db`）** | 40 dB ⇒ P_D 0.683 | **已于 2026-09-20 物理删除**。见 5.3 |
| **残余记账"只算结构残差"（δ=0）** | P_D 0.5167 → 0.8833（配对 Δ +0.367 ± 0.048） | ✅ **已落地为门控** `cancellation.residual_accounting="structural"`（默认 `measured` ⇒ 逐位不变）。⏸ 但 **δ=0 档仍只是上界**（字典完备，κ≈68 dB 是构造产物）⇒ 不得单独当可达值引用。**δ>0 档可报，但 δ 未标定**（2026-09-21 再审：3e-3 是扫描档位、已知超标 1.86 倍）⇒ 只能报"若 δ=X 则 P_D=Y"的**敏感性**，不能叫"真实版"。见 §5.6 与 `AUDIT_DIRECTION1_FAILURE_ROOT_CAUSE.md`、`AUDIT_REAL_VS_EXPECTED_GAP.md` |

### 5.3 κ：已物理删除（2026-09-20），删除后默认 = **没有对消**

⛔ **纪律（违反过一次，2026-09-20）**：κ **不得再作为参照臂、基线或讨论对象**。
任何 "κ=40 vs κ=37 差多少" 的表述都违反此条 —— 它把占位常数重新请回了基线位置。

**状态：已删除。** `direct_cancellation_db` 已从 `isac_sim/core/config/interference.py`
移除；`isac_sim/` 内**零处读取**；`fixed_kappa` 对照臂、`experiments/flow/sweeps.py`
的 κ 扫描轴、发布门禁的冻结键（96 → **95**）同步退役。

**为什么删。** 它断言"接收机从聚合直连场消掉固定 40 dB"，但那是需求反推的数字，
生产链路里没有任何接收机实现支撑它。它却撑着 SINR 的整个分母 —— 裸删代价实测
κ=0 ⇒ **P_D 0.050 = P_FA（系统失效）** —— 所以挂在它上面的任何"增益"都是分母上
的记账，不是性能。删它**不是**因为 40 dB 太小，而是因为它根本不该存在。

**删除后的语义（先定死再动手，见 `.workbuddy/KAPPA_DELETION_PREREG.md`）：**

| 情形 | κ_dc | 含义 |
|---|---|---|
| 传入 TP-UIC 实测残余 | 实测值 | **唯一可报的数字来源** |
| `cancellation.mode="predict"` | 解析桥 | 仅用于规划，不得当结果 |
| 都没有（`mode="off"`） | **1.0（0 dB）** | **开放环路声明：没有接收机提供对消** |

⚠️ **后果：默认路径 P_D 会塌到 P_FA 水平。这不是回归，是把假设从分母里拿掉。**
实测（MC=20，600 m / RCS 0.1，seed 2026）：`target-local-v1` **P_D 0.030 / P_FA 0.0557 /
links 0.05**；`paper-canonical` **P_D 0.060 / P_FA 0.0517 / links 0.10**。
注意 links 也几乎归零 —— 每条链路的边际价值 ≈ 0，所以调度器**买不动东西**。
任何要报的数字必须显式接 TP-UIC 实测。

**作废数字清单（不得再引用）：** 0.7000 / 0.6250 / 0.5550 / **0.5150** / 0.5050 /
0.4950 / 0.4750 / 0.4500 —— 全部建立在 κ=40 的占位口径上。其中 **0.5150 还额外
带一个专属缺陷**：它出自"半闭环"接线，选择侧的链路表没拿到证书（见 §5.4），
所以它是**虚高**，不是"接上 TP-UIC 的真实代价"。

**唯一可报：0.4300**（全闭环 TP-UIC 接线，`cancellation.production_wire=true`，
MC=20，600 m / RCS 0.1 / seed 2026 / `target-local-v1`）。

**深度缺口是独立问题：** 需求 52.25 dB vs TP-UIC 实测 37.25 dB，差 **15 dB**。
接线只让数字诚实，不改善它；这 15 dB 才是真正的改进项。

### 5.4 TP-UIC 生产接线（已实现，门控默认关闭）

**状态：已接线。** 门控键 `cancellation.production_wire`（未登记，默认 `False`）
开启后，每个 trial 测量一次接收机，把 **`(残余比例, 回波存活率)`** 注入该 trial
内**所有** `compute_link_tables` 调用点，取代占位常数。

**为什么用 contextvar 而不是逐个传参**：生产链路有六处调用点（粗/细、调度/评估、
真值/信念、active_set 重建）。逐个传参既易漏，又最易造成**两个世界** —— 一处实测、
一处冻结，链路照样跑通、数字照样合理，但接收机模型自相矛盾。contextvar 让六处
自动拿到同一证书，调用点签名不变。显式传参优先于上下文（调用方意图不被广播覆盖）。

| 文件 | 作用 |
|---|---|
| `isac_sim/sensing/model/link_tables/trial_certificate.py` | 证书容器 + contextvar 作用域 |
| `link_tables/compute_link_tables.py` | 入口回填（两参数均省略时才取上下文） |
| `core/config/cancellation.py` | `production_wire` / `_arm` / `_retention` 三个门控键 |
| `experiments/flow/simulate.py::run_one_trial` | 测量 + 作用域包裹（belief 与 default 两分支） |
| `tests/test_tpuic_production_wire.py` | 11 项契约测试（含"粗/细共享同一证书"与"剪枝逐位等价"） |

⚠️ **测量走独立随机流** `[seed, 10**6 + index]`，不消耗主 rng ⇒ 门控关闭时
连一次抽取都没有，**逐位不变**（`check_release_identity` 实测 CLEAN 95/95）。
同一个 trial 的 on/off 因此共享几何与 belief，构成真正的配对比较。

⚠️ **算力约束（2026-09-21 更新，旧结论已作废）**：一次测量曾是 **40 s/trial**，
当时的结论是"禁止进 MC=1000"。经两轮**数值恒等**的提速（装配剪枝 1.31× + 联合
系统缓存 1.74×，见 `.workbuddy/PARALLEL_COST_AUDIT.md`）：单 trial 43.65 → **25.99 s**，
workers=16 吞吐 **3.18 s/trial**，**MC=1000 ≈ 0.88 h** ⇒ **发布扫描已可行**，
不再需要"离线测量 + 查表"这种近似来救算力。

⚠️ **并行会自伤**：单 trial 墙钟 43.65 s（1 进程）→ ~48 s（16 进程），放大约
1.8×；并行效率只有 0.55。**报吞吐（s/trial）与报单 trial 延迟必须分开**，
混着说会误以为单 trial 真的变快了。

**生产链路 20 trial 配对实测（`target-local-v1`，600 m / RCS 0.1，seed 2026）：**

| 门控 | 接收机模型 | P_D | P_FA | 观测数 / 远程上报 / 比特 |
|---|---|---:|---:|---:|
| 关 | **开放环路**（κ 删除后 `kappa_dc=1`，声明"没有接收机"） | **0.030** | 0.0557 | 57.0 / 0.0 / 0 |
| 开 | **TP-UIC 实测（全闭环）** | **0.4300** | 0.0494 | 47.4 / 18.1 / 11584 |

ΔP_D = **+0.4000**（n=20，配对 SE 未计算）。对照"半闭环"时期的 0.5150 ——
**修复证书作用域后掉到 0.4300**，那 0.085 是虚高，不是损失。

### 5.5 残余干扰的记账：恒等式成立，但修正不成立（2026-09-21）

判决脚本用线性性把残差拆开（`f` 确为线性算子）：

    r = y - f(y) = [x - f(x)] + [s - f(s)] + [n - f(n)]      (分解误差 1e-29)

⇒ **`‖f(n)‖²` 不在残差里**，它是被减掉的那部分噪声。现行
`i_res = ‖x−f(x)‖² + ‖f(n)‖²` 把"被减掉的噪声"记成了"没消掉的干扰"，
而这一项占了 `i_res` 的 **99.95%**（多记 33.3 dB）。实测链路表 `n0` 与逐 bin
`sigma2` 是同一个数，于是 `rinr` 中位 **+7.17 dB** —— 分母被抬高 **7.7 dB**。

**但把这一项拿掉是虚假增益。** 拿掉后残余只剩 `‖x−f(x)‖² ≈ 0`，而它之所以为零
是因为 `x_direct = X @ h_true` 落在字典张成空间里。off-grid 探针实测：

| 字典外能量占比 | `kappa_structural` 中位 |
|---:|---:|
| 1e-6 | 59.0 dB |
| **9e-6** | **50.3 dB**（跌破需求 52.25 dB） |
| 1e-4 | 40.0 dB |
| 1e-2 | 20.0 dB |

⇒ 达标要求字典外能量 ≤ 6e-6，折算成**直连路径时延估计精度 ≤ 约 0.2 m（0.7 ns）**
（sinc 近似，只取量级）。
这是一个此前从未声明的强假设，必须与 `G_hw`、DD-only 口径同级登记（§6）。

⚠️ **纪律**：恒等式判决是**无条件成立**的（纯数值），可以引用；
"因此可以删掉 `‖f(n)‖²`"**不成立**。归类错误不等于可以删除 —— 删掉之后
剩下的深度由"字典完备"撑着。方向 1 的正解是**建模直连参数估计误差** +
**用切向阶数/字典加密改善**，见 `studies/direction1/docs/TPUIC_RESIDUAL_ACCOUNTING_AUDIT.md` §7。

⛔⛔ **禁止用 `rinr` 反推 κ**（2026-09-21 复核踩的坑）：
`rinr` 的分子是 `residual_self + residual_direct + residual_multi` 三项之和，
只有 `residual_direct` 被对消器缩放。`residual_self`（自干扰残余）是
**与对消器无关的地板**，实测 `0.261 n0`；`structural_only` 臂的 `rinr = −5.8 dB`
里 **99.1% 就是这个地板**，真实结构残差只有 `0.0023 n0`（比地板低 20.8 dB）。
曾据此算出"全链路 κ_struct = 49 dB"，与直接读字段的 66–68 dB 差 19.6 dB ——
**是换算错误，不是数据矛盾**。反推前必须先扣地板，或直接读
`CancellationResult.i_res_structural`。

~~★ 方向 1 应攻"下尾"而非"中位"~~ —— **2026-09-21 改判，见下方修正。**

原推理（"中位已在地板下 20.8 dB ⇒ 深挖中位零收益、该攻下尾"）**只适用于
"已经用了 structural_only 记账"之后**：实测 6 个接收机逐机 κ_struct =
`68.6 / 72.1 / 67.4 / **59.6** / 72.8 / 67.1` dB，最差机比最好机低 13.2 dB，
把下尾机抬到中位只值 **0.35 dB**（且只影响 6 机中的 1 机）。

🔴 **但它被误用成了"深度没有收益"**。天花板扫描
（`studies/direction1/scripts/diag_direction1_ceiling.py`，固定调度只扫 κ）实测：
κ 36.5（当前）⇒ P_D 0.4667；52.25（需求）⇒ **0.7500**；perfect ⇒ **0.8833**
⇒ 从当前中位走到需求值 **+0.283 P_D**，天花板 **+0.417**。
**该攻的是"把中位从 36.5 抬到 52"，不是"把 68 抬到 72"。**

⇒ ⛔ **"深度无路可达 / 收益 ≈ 0"作废**。无路是因为压噪声增强项
（占 `i_res` **99.8%**）的旋钮没接线，不是物理上不可达。

### 5.6 记账口径已门控化（2026-09-21 落地）

`cancellation.residual_accounting`：

| 取值 | `i_res` | 语义 |
|---|---|---|
| `measured`（**默认**） | `‖x−f(x)‖² + ‖f(n)‖²` | 现行，**逐位不变** |
| `structural` | `‖x−f(x)‖²` | 只记结构残差；**须配 δ>0**，δ=0 只是上界 |

**为什么做成门控而不是直接改掉**：§5.5 已判 δ=0 时 `structural` 是"字典完备"的
构造产物，`κ≈68 dB` 是上界不是可达值。门控让"模型版（上界）"与"真实版（δ 标定）"
各报一次，而不是偷偷换掉发布基线。

门控**只改记账、不改算法**：两种口径下 `residual` / `h_hat` / `c_h_diag` /
`eta_survive(_q)` / `i_res_structural` / `i_res_estimate` 逐位相同（由
`tests/test_residual_accounting_mode.py` 钉住）。解析侧 `i_res_pred` 同步同口径，
否则 `calibration_error_db` 会跨口径相比。

**门控跑出来的端到端**（`studies/direction1/scripts/run_direction1_gated.py`，n=20 配对，
`paper-canonical`；**不等于**生产链路的 0.4300，可迁移的是配对 ΔP_D）：

| 臂 | κ [dB] | P_D | ΔP_D |
|---|---:|---:|---:|
| `measured`（默认） | 37.17 | 0.5167 | — |
| **δ=3e-3**（不是"真实版"，见下） | 46.96 | 0.7167 | **+0.200 ± 0.061** |
| 模型版 `structural + δ=0`（**上界**） | 67.44 | 0.8667 | +0.350 ± 0.051 |

⚠️ **MC=20 分辨不了上表的 P_D 差**（2026-09-21 审计实测）：同一 δ=3e-3 配置、
只换 rng tag，两次 n=20 给出 **0.7167 vs 0.8000**（差 0.0833），而 κ 稳定到
`46.96 / 46.97`。⇒ κ 是低噪声量、P_D 的 SE≈0.05。
**用 κ 判机制，用 P_D 判收益但必须给 MC**；MC=20 只能分辨 ≥0.10 的 P_D 差异。
两次实现的均值：δ=0 → 0.875、δ=3e-3 → 0.758，**真实 gap ≈ 0.117 而非 0.150**。

⛔ **"真实版"这个标签作废**（改名"δ 敏感性"）：δ=3e-3 不是标定值，它来自
扫描档位 `{0, 1e-3, 3e-3, 1e-2, 3e-2}` 的中间一档，而那份扫描自己已算出达标
边界是 δ≈1.6e-3 ⇒ **3e-3 是已知超标 1.86 倍的档位**。报数必须写成
"若 δ=X 则 P_D=Y"。δ 的物理锚点是**时钟同步精度**（直连两端都是协同节点、
位置共享 ⇒ 几何上是算出来的，δ 只描述时钟/晶振误差）：

| 同步精度 | δ [格] | κ [dB] | 达标? |
|---|---:|---:|:---:|
| 0.5 ns | 9.6e-4 | 56.85 | ✅ |
| **0.84 ns（门槛）** | 1.615e-3 | 52.25 | — |
| 1 ns（IEEE 1588） | 1.92e-3 | 50.96 | ❌ |
| 1.56 ns（本轮取值） | 3e-3 | 47.11 | ❌ |
| 10 ns（GPS） | 1.92e-2 | **31.00** | ❌ |

⇒ **达标需亚纳秒同步（≈0.86 ns）**；GPS 级（10 ns）下 κ 仅 31 dB，
方向 1 收益几乎归零。本轮取值相对 GPS 级**仍偏乐观**。

**差距归因已闭合**（`studies/direction1/scripts/diag_delta_gap_attribution.py`，三判据全 PASS）：
H1 差距 **100% 由 κ 解释**（投到独立天花板曲线，残差 0.0743 ≤ 2·SE=0.1068）⇒
**无隐藏 bug**；H2 κ(δ) = `-10log10(2.2139·δ² + 1.803e-7)`，四档拟合误差
**≤0.12 dB**；H3 δ>0 三档分子固定/随臂的 P_D 差 **全为 0.0000** ⇒ δ 对分子的影响**可忽略**
（⚠️ 臂级实测 η_survive 相对变化 **4e-5**，不是逐位 0 —— 字典变了保护子空间跟着变。
所以"δ 只动分母"在逐位意义上**不成立**，正确说法是"分子变化 ≤1e-4，不改变判决"）。
⇒ **方向 1 结论已收口为断言**：`tests/test_direction1_convergence.py`（14 tests）。
一次性探针与停止条件见 `studies/direction1/README.md` 与 `studies/direction1/docs/DIRECTION1_CONVERGENCE.md`。

⚠️ **一致性缺陷**：`waveform_impairments.sync_delay_bins`（罚分子）
与 `cancellation.direct_estimation_sigma_delay_bins`（抬分母）描述**同一个同步误差**
却是两个键。当前默认都是 0.0 一致，**一旦标定必须设成同一值**。
且本轮只注入了**延迟维**，多普勒维未扫 ⇒ κ(δ) 是单维结果（两维联合待跑）。

⛔ **别手工算 `(i_res − i_est)`**：`estimation` 占 `i_res` 的 **99.8%**，这个差是
**灾难性抵消** —— 实测相对误差可达 `1e294` 量级（低位全丢，差值可归零或取到
舍入残留）。要结构残差就读 `CancellationResult.i_res_structural`，或用门控。
上一轮的处方脚本正是手工算的，**其数字只能作量级参考**。

### 5.7 方向 2（协作融合）定论与判据（2026-09-21，MC=120）

**判据**：`J6` 配对 |z| ≥ 1.96 且 ΔP_FA ≤ 0.01；`J7` `pd_per_mbit` 不降（禁止用比特换 P_D）；
`J10` Δweak_detected ≥ −0.05 —— **新增**，因为总 P_D 会掩盖弱目标塌陷（mc12 里 `robust`
总 P_D 持平而弱目标 **0.25 → 0.08**）。

**主表（默认 measured 口径，MC=120，7041 s，600 m / RCS 0.1）**

| 臂 | ΔP_D | z | 代价 | 判决 |
|---|---:|---:|---|---|
| baseline | **P_D = 0.3942** | — | — | 与发布 0.4300 同量级（略低因实测 κ≈37 < 原常数 40） |
| `robust` | +0.0150 (se .0115) | **1.30 不可分辨** | kbit 19.02→**11.24(−41%)**，pd/Mbit 20.7→**36.4(+76%)** | J6 挂 ⇒ **默认口径下它是效率选项，不是检出杠杆** |
| `links12` | +0.0150 | 2.33 | 效率 −20% | J7 挂 |
| **`robust_links12`** | **+0.0283** | 2.37 | Δweak = **−0.050**（压线） | **唯一 J6+J7+J10 全过** |

★ **两机制可加**（0.0283 ≈ 0.0150+0.0150）⇒ n=4 读到的"robust 抵消 links12"是噪声。

**三项已判死（不得再当优化臂）**：融合阈值（精确标定 ≡ Cornish–Fisher **逐位等价**）｜
节点功率（`sensing_power_scale_by_uav` 零处传非 None）｜融合权重（`fusion_weight_mode_for_method`
按方法名硬编码）。详见 `studies/direction2/docs/AUDIT_DIRECTION2_ATTRIBUTION.md`。

**κ 依赖（三口径，n=12 生产路径，同 seed 前缀）**

| 口径 | κ [dB] | baseline P_D | `robust` Δ (z) | `links12` Δ (z) |
|---|---:|---:|---:|---:|
| `measured` | ≈37 | 0.3333 | +0.0167 (0.35) | **+0.0500 (2.56)** |
| `structural δ=0`（上界） | **59.6** | 0.7417 | **+0.1250 (4.48)** | +0.0083 |
| `structural δ=3e-3` | **49.3** | 0.6667 | **+0.1583 (3.98)** | **+0.0000** |

★ **律：融合类干预的有效区间由 κ 决定，且方向相反** —— 低 κ（触顶）⇒ `links12` 有效；
高 κ（不触顶）⇒ `robust` 有效 ⇒ **报方向 2 收益必须先报 κ 区间**。
CI 不重叠（`robust` 95%CI：measured `[−.008,+.038]` n=120 vs structural δ=0 `[+.070,+.180]` n=12）。

⚠️ **推翻两条早期结论**：① `robust` 不是"效率杠杆"而是"用上报换检出"（structural 下
kbit ×2.5、`pd_per_mbit` 腰斩 ⇒ J7 挂；它在 measured 下"省比特"只是低 κ 触顶的产物，
**效率结论不可跨口径迁移**）；② `links12` 在 structural 下是**空操作**（高 κ 时 baseline
每目标只 2.6 条，离上限 6 很远）。
⚠️ 诊断探针口径（baseline 只有 14.7 条链路）读到的 +0.3333/+0.2167 **不可与生产口径并列**。

🔢 **量级**：方向 2 最佳 **+0.028** vs 方向 1 记账修正 **+0.27~+0.35** ⇒ **小一个数量级**
⇒ 先定方向 1 的默认口径更值。🟡 structural 两栏仅 n=12（各补 MC=120 需 ≈2 h），
仅当方向 1 采纳 structural 默认时才需补。
🟡 **待裁决**：`robust`（`prior.robust_geometry_for_scheduler`）是否设为默认 —— 属 B 类，会动发布数字。

---

## 6. 假设登记册（新增假设必须登记）

任何进入数字的、非实测的量都要登记，并在论文里与硬件假设同级声明。

| 假设 | 当前值 | 状态 |
|---|---|---|
| `G_hw` 雷达净增益 | 0 dB | 已定死 |
| κ 直连对消 | **已物理删除**（原 40 dB 占位）。实测随口径：measured ≈**37** dB、`structural δ=0` **59.6** dB（上界）、`δ=3e-3` **49.3** dB；需求 52.25 dB | ⛔ **不得再作参照臂 / 基线 / 讨论对象**；报数必须写明口径（§5.3、§5.7） |
| **双世界 belief 口径** | `belief_mode=True` ⇒ belief 表给调度、truth 表给检测 + `truth_captured_links` 过滤 | **必须声明**：只建一套表 = 完美先验假世界，融合类干预机制上必然 0.0000（§1.1 Q3） |
| DD-only 接收机口径 | 主线 | **必须声明**：角度维探针显示 DD 重叠主要是该抽象的产物（ρ≈0），小阵列可救回 43%/67%/82%（8/18/38 cm） |
| belief 位置误差 σ_pos | 150 m（≈1.9 距离单元） | 无跟踪器的单次快照；**当前最大真实杠杆** |
| belief 速度误差 | 15 m/s | 同上 |
| 相关性 ρ（四项） | 未标定 | 当前值是假设 |
| `robust_position_confidence` | 0.50 | **带峰值**：0.5 ⇒ P_D 0.820；0.9 ⇒ 0.665（低于基线） |
| `prior.robust_geometry_for_scheduler` | **False**（门控默认关 ⇒ 逐位不变） | 只改**调度视野**（误差球下界），检测侧照旧 truth。MC=120 定论见 §5.7；设为默认属 **B 类**（动发布数字），待裁决 |
| **直连路径参数估计精度 δ** | 默认 **0**（完美估计）；达标门槛 ≈ 1.7e-3 格 ≈ **0.27 m（0.9 ns）** | **已可切换**：`cancellation.direct_estimation_sigma_{delay,doppler}_bins`（默认 0.0，逐位不变）。2026-09-21 用**分数延迟偏移注入**重测（修正上一轮正交注入口径）：字典外能量比恒 = `1.55·δ²`（五档离散 <1%，与 `(π²/3)δ²` 同量级），κ_struct 门槛 1.7e-3 格。⚠️ **生产口径 κ_total 看不见 δ**（δ: 0→3e-3 只动 0.17 dB），因为噪声增强项主导 —— 报"模型版/真实版"两个数字时须说明这一点。🔴 **2026-09-21 再审：δ 从未标定，且物理锚点是时钟同步精度** —— 直连两端都是协同节点、位置共享 ⇒ 几何上算得出来，δ 只描述时钟/晶振误差。达标（52.25 dB）需 **≤1.615e-3 格 = 0.84 ns**；GPS 级（10 ns）下 κ 仅 **31 dB**。**本轮"真实版"取的 3e-3（1.56 ns）是扫描档位、非标定值、已知超标 1.86 倍 ⇒ "真实版"标签作废，改称"δ 敏感性"**。κ(δ) = `-10log10(2.2139·δ² + 1.803e-7)`，四档拟合 ≤0.12 dB |
| **`cancellation.n_cpi`（参考预算）** | 1 | 🔴 **2026-09-21 实测：未接到对消臂**。κ_total 在 n_cpi=1/4/16/64 下**逐位相同**（36.50）；它只进 `cancellation_glrt` 的 `C_res` 与解析桥 `predict`。原注释"深度按 10log10(n_cpi) 增长"**在实测路径上不存在**，已就地修正。**别拿它当已接线的深度杠杆** —— 它是"路径 2"：接上后按 `estimation/n_cpi` 解析预测 κ = 37.95/43.96/**49.96**/**55.86**/61.46 dB（n_cpi=1/4/16/64/256），**代价是登记新假设"存在独立参考观测"** |

---

## 7. 测试规范（细节口径见 `docs/TEST_STANDARD.md`）

**三条定位**：P1 测试是结论的载体（一条结论算收口 ⇔ 推翻它就有测试 FAIL）｜
P2 钉契约不钉数字｜P3 新增即门禁（先想"怎么变成会 FAIL 的断言"，再想文档）。

| 编号 | 规则 |
|---|---|
| **T1** | 提案/规格类需求落成**一致性测试**，缺口用 `xfail(strict=True)` 登记（实现当天会 XPASS 报错，逼你转常态断言） |
| **T2** | 新能力必须覆盖"开关关闭"与"开关打开"两侧，且关闭侧逐位不变 |
| **T3** | 把承诺写成**断言**，不写成注释（教训：`alpha_true` 注释写"切向真值恒零"而代码赋满） |
| **T4** | 测试文件 ≤350 行（S5，`tests/` 全目录纯红线），一条链路一个文件，共用代码进 `tests/_<topic>_common.py` |
| **T5** | **门控类测试必写四项**：默认关 / 不在冻结清单 / 恒等条件（退化输入下 on≡off）/ 真的接线（非退化下确实改动） |
| **T6** | **断言档位从高到低**：结构·签名探针 → 调用次数 → 恒等 → 单调·下界 → 比值 → 绝对阈值（绝对阈值只在有 oracle/上界对照时用） |
| **T7** | **禁重实验**：测试里不跑 MC≥20、不开 `production_wire`；单文件 ≤5 s |
| **T8** | **拆文件先立防丢基线**（`_test_inventory.py`）；有意删测试须同 commit 改基线 |

当前门禁落点：`tests/test_optimization_model_{contract,objective,resources,alternation}.py`
（提案骨架，37 passed / 7 xfailed）｜`test_direction1_convergence.py`（14 断言，方向 1 收口）｜
`test_direction2_fusion_audit.py` + `test_direction2_robust_gate.py`（方向 2 判死 + 门控）｜
`test_module_size.py`（150/350 行）｜`test_test_inventory.py`（防丢）｜
`test_study_docs_hygiene.py`（归档卫生）｜`test_test_file_hygiene.py`（测试结构卫生：W1/W2/W6/F4/F5）。
⚠️ **一次性猜想不要新写 `scripts/diag_*.py` 就收工** —— 能变成断言的进 `tests/`，
只是探索才进 `studies/directionN/scripts/` 且必须被该方向 README 索引。

---

## 8. 已知坑清单（踩过第二次的）

| 坑 | 现象 / 正确做法 |
|---|---|
| **死字段** | `cancellation.search_half_width` 全仓只有声明、**零处读取** ⇒ 不存在"局部 DD 搜索"。**任何"调这个旋钮"的建议，先 grep 它有没有被读**。同类（声明了但没接到被测路径）：`cancellation.n_cpi` —— 只进 GLRT 的 `C_res`，对消臂实测逐位不敏感 |
| **Jensen 反向** | `target_gain ∝ 1/(d_i²d_j²)` 是位置的**凸**函数 ⇒ 不确定集上取**期望**是**更乐观**（E[g] ≥ g(中心)），不是鲁棒。**鲁棒化必须取下界**（min / 置信球闭式） |
| **N/L 不是分辨率旋钮** | 延迟量化挂 `L`（`round(τ·L·Δf)`）、噪声带宽挂 `N`（`B=N·Δf`）；默认 `N=L=64` 让两者读数相等掩盖不一致。**碰撞统计跟随取整格，不跟随分辨率** |
| **解析桥不能当结果** | `cancellation.mode="predict"` median rinr 8.38 → **2513**（+24.8 dB）⇒ 只能规划 |
| **`eta_survive` 是整场口径** | 随 stage-2 支撑集大小漂移（0.705 ↔ 0.877）；问逐目标问题一律用 `eta_survive_q` |
| **`max_protected_targets=0`** | = 保护**全部**（不是"都不保护"），是消融变体，别按字面改 |
| **标称 P_FA ≠ 实现值** | 标称 0.05，实现 **0.0499952**（`threshold_from_pfa` 四位常数表 1.6449） |
| **V1/V1.1 归档数字不可复现** | 生成器与支撑口径都变了；只还原支撑维度用 `--candidate-policy statistic` |
| **残余 vs 噪声的口径** | 观测总噪声 `‖n‖² = n_bins · sigma2`，而链路表 `n0` 是**单 bin** 口径。拿 `‖n‖²` 当参考会低估残余 4 个数量级（我据此得出过"残余可忽略"的错误结论）。**比较前先确认分母口径** |
| **恒等式成立 ≠ 可以删** | `‖f(n)‖²` 不在残差里是真的（分解误差 1e-29）；但**裸删** ⇒ 深度由"字典完备"撑着 ⇒ 虚假增益。**归类错误不等于可以删除**。正确做法是配合 δ 门控：`i_res = ‖x−f(x)‖² + E[字典外能量(δ)]`。⚠️ **裸删依旧禁止**，但 δ 建模完成后该处方已可实施（δ=3e-3 档实测 +0.200）——见 §5.5 |
| 🔴 **默认路径没有对消 ⇒ 端到端扫描全躺地板** | `production_wire` 默认 `False` ⇒ `kappa_dc=1.0`、D_fuse≈1e-6、P_D=0.0000。**Δ 恒 0 会被误读成"饱和/无效"**（方向 2 第一版全 0）。端到端扫描必须先开（§4.4） |
| 🔴 **只建一套链路表 = 假世界** | `belief_mode=True` 下必须 belief 表给调度 + truth 表给检测。只建一套，"加链路"机制上必然 0.0000（§1.1 Q3） |
| 🔴 **preset 已设的开关再"打开"是空操作** | `paper-canonical` 已开 5 个开关 ⇒ 方向 2 一次砍掉 4 个臂。**开优化方向前先查 preset**（§1.1 Q1） |
| 🔴 **融合类干预的有效区间由 κ 决定且方向相反** | 低 κ（触顶）⇒ `links12` 有效；高 κ ⇒ `robust` 有效。**报收益必须先报 κ 区间**，不可跨口径迁移（§5.7） |
| 🔴 **总 P_D 会掩盖弱目标塌陷** | mc12 里 `robust` 总 P_D 持平而弱目标 0.25→0.08 ⇒ 判据必须含 Δweak（J10） |
| 🔴 **盘点行数用 Python `splitlines()`** | `Get-Content \| Measure-Object -Line` 少算末尾无换行的文件；本轮因此低估 60–120 行并**漏报 1 个超限文件** |
| 🔴 **中文 md 别用 `Get-Content \| Out-File` 搬** | 按 GB18030 解码后重存 ⇒ 方向 1 五份文档**永久损坏、不可逆**（数字没丢只因结论已落成断言）。用 Write/Edit 或 Python 显式 `encoding="utf-8"` |
| **拆测试文件前先立防丢门禁** | 失败模式是**静默丢 `def test_*`**（pytest 照跑、总数正常）。基线 key=原文件名，拆完不更新；删测试须同 commit 改基线（§7 T8）。**拆前先确认不是退役候选** |
| **结论先落断言、文档只做索引** | 文档会烂、探针不进 CI；只有断言会 FAIL |
| ★ **优化比值型指标前先做逐项占比分解** | 2026-09-21 根因：`i_res = structural + estimation` 中 `estimation` 占 **99.8%**（实测比值 **506**），而整个方向 1 都在动那 0.2% 的 `structural` ⇒ δ 门控、切向阶数全部打空。⛔ **占比 <5% 的项，改它必然看不到效果，不管改得多正确** |
| ★ **"归属错"与"瓶颈项"是两个问题** | 前者判"该不该记进分母"，后者问"谁主导该量"。`‖f(n)‖²` 归属确实错（它只占噪声 **0.12%**，是**被减掉**的那份，残差保留了 **99.93%** 的噪声），但**它同时是瓶颈项** ⇒ 就算归属错，压它仍是抬 κ 的唯一路径。把前者的裁决误当成"方向没有收益"，导致上一轮写出"深度收益 ≈ 0"的错误结论 |
| ★ **门控要在它生效的口径下评估** | δ 门控只在"分母没有 estimation"时可见：生产口径 δ: 0→3e-3 只动 **0.17 dB**（看起来无用），修正口径 κ_struct 差 **15 dB**（63.53 → 48.44）。串联的改动不能分别单独评估 |
| **注入失配要改两处** | `i_res_structural` 是臂内部拿 `obs.x_direct` **单独特算**的（它知道真值）。只改 `obs.y` 不会让它动 ⇒ κ 对注入完全不响应，那是**注入失效不是结论** |
| **PowerShell 120 s 超时** | 会**静默杀掉**长任务（无 traceback、exit 1）⇒ 重变体必须后台跑或拆开跑 |
| **PowerShell 重定向** | 产出 **UTF-16**，读的时候要 `decode('utf-16')`；脚本自己用 UTF-8 写日志更稳 |
| **bash coreutils 缺失** | 该环境 `ls/tail/head/dirname` 时有时无 ⇒ 用 python / PowerShell 代替 |

---

## 9. 迭代 checklist（一步一卡，不得跳步）

```
[ ] 1. 四问：preset 已设? / 死参数? / 单世界双世界? / 分辨力够吗?（§1.1）—— 答不上先补诊断
[ ] 2. 分诊：A / B / C / D（§1）—— 分不清按更高类别
[ ] 3. 判据：写进脚本 docstring，常数一致性自查（E1/E2）；定死 MC 与分辨力（E8）
[ ] 4. 读代码：确认要改的字段**真的被读取**（死字段坑，Q2）
[ ] 5. 实现：B 类必须"未登记新键 + 默认关闭"
[ ] 6. 单测：门控四项（默认关/不在冻结清单/恒等条件/真的接线）+ 关闭侧逐位不变（§7 T2/T5）
[ ] 7. 门禁：pytest（含防丢 + 归档卫生 + 测试结构卫生）+ release_identity + contract_refs（+ parity，D 类）
[ ] 8. 实验：诚实口径、同 run 配对、oracle/upper-bound 对照（E3/E4/E6）；
        端到端先确认 production_wire 与口径（§4.4、E9）
[ ] 9. 准入：G1/G2/G3 三条全过；不在虚假增益清单里（§5）；判据含 J10 弱目标（融合类）
[ ] 10. 归档：数字入 results/ 或 studies/directionN/data/；**结论入 tests/ 断言**，
        再写 .workbuddy/memory/YYYY-MM-DD.md（追加，不覆盖）；md 只做索引
[ ] 11. 提交：门禁全绿后 commit；推送用 ls-remote 实查（§2.3）
```

---

## 10. 当前版本状态与下一步排序

### 10.1 现状（600 m / RCS 0.1）

| 口径 | P_D | 说明 |
|---|---:|---|
| **全闭环 TP-UIC（`production_wire=true`，MC=20）** | **0.4300** | **唯一可报的发布级数字** |
| 默认路径（`production_wire` 关） | ≈0.0000 | κ 删除后的开放环路声明，不是回归（§4.4） |
| 半闭环 0.5150 / κ=40 时代的 0.7000、0.8200 | ⛔ **作废** | 建立在占位 κ 或错误证书作用域上（§5.3） |

### 10.2 待裁决（B 类，会动发布数字，需用户拍板）

| # | 议题 | 影响 | 建议 |
|---|---|---|---|
| **D-1** | 方向 1：`residual_accounting` 默认是否改 `structural`？ | 量级 **+0.27~+0.35**，比方向 2 全部收益大一个数量级 | 先裁这个；⚠️ 前提是 δ 有标定值（否则只是上界，不能当发布口径） |
| **D-2** | 方向 2：`robust` 是否设默认？ | measured 下 +0.0150（z1.30 不可分辨，纯效率项）；structural 下 +0.125~0.158 | D-1 裁完再定；`robust_links12` 是唯一三判据全过的组合臂 |
| **D-3** | δ 的标定值（现无；扫描档 3e-3 已知超标 1.86×） | 决定方向 1 能报多少；物理锚点是时钟同步（达标需 ≤0.84 ns） | 需要先有硬件/同步假设，否则只能报敏感性 |

### 10.3 改进项排序（决策价值 ÷ 成本）

| 优先级 | 动作 | 依据 | 类型 |
|---|---|---|---|
| **P0** | 降低 belief 位置误差（150 m → ≤78 m）：多帧跟踪 / 滤波 | **+0.259 且比特 −56%**（唯一"又准又省"的路线） | C（需要新能力：跟踪器） |
| **P0′** | 头条改用下界增益做边际判据（官方 `..._pd_robust` 已实现） | +0.120，代价 bits ×3.07；**conf 带峰值，0.5 最佳** | B/C（改默认 = C 类） |
| P1 | 把 `production_wire=true` 变成发布默认（先解决 κ≈37 < 需求 52.25 dB） | 当前默认路径没有对消 ⇒ 任何端到端数字都缺实现支撑 | C |
| P2 | 相关性 ρ 四项标定 | 有效链路数 6.4 → 5，第二梯队 | C |
| P3 | 融合侧（`robust_links12`，+0.0283） | 三判据全过但量级小一个数量级，且依赖 κ 区间 | B |
| ⛔ | κ 优化 / G_hw 增益 | 虚假增益，不讨论、不给改进项 | — |

⚠️ **换目标函数形状前先问"资源稀缺吗"**：默认预算下两臂都选满，A/B 差异常被"两边都选满"
掩盖。实测 softmin→max-min 的配对 Δworst P_D 只有 +0.0003，而结构约束 `max_tx_nodes`
（照射机 4.8→3.0）与价格项 λ_c（远程上报 5.6→6.8）的效应**大一个数量级**
⇒ **别指望靠换目标函数形状换性能**。

---

## 12. 数学骨架（提案 §1–§20）闭合状态

> 编号沿用历史引用（本节物理位置排在 §11 之前，引用不变）。

门禁：`pytest tests/test_optimization_model_*` ⇒ **37 passed / 7 xfailed**（含行数门禁）。
通过的＝骨架已落地；xfail 的＝缺口清单（实现当天会 XPASS 报错）。

### 12.1 已闭合

| 提案条款 | 状态 | 证据（测试项） |
|---|---|---|
| §1 变量只有 `x`(二值成员集) / `z`(二值目标集)，第一版不加 power/beam/CPI | ✅ | `test_selection_variable_is_a_binary_membership_set`、`test_protection_variable_is_a_binary_target_set`、`test_first_version_optimises_no_power_beam_or_cpi` |
| §2 接口恰好两个标量、η 乘分子、I_res 在分母、certificate 是 belief-side | ✅ | `test_certificate_is_exactly_two_scalars`、`test_survival_scales_the_numerator`、`test_residual_sits_in_the_denominator`、`test_certificate_is_belief_side_not_a_truth_measurement` |
| §3 式(2) 信息可加 `D_q=Σx·J(γ)`，J 由检测器导出而非手调 | ✅ | `test_information_is_additive_over_independent_observations`、`test_per_link_information_is_derived_not_hand_tuned` |
| §4 式(3) `P_D=G(D,P_FA*)`，门限只依赖设计 P_FA，P_D 对 D 单调 | ✅ | `test_pd_is_a_function_of_information_at_the_fixed_pfa`、`test_the_threshold_depends_on_the_design_pfa_and_nothing_else`、`test_pd_is_monotone_in_information` |
| §7–§14 C1/C2/C3/C4/C5/C7/C8/C9 全部作为**硬约束**存在 | ✅ | `test_c1_per_target_observation_budget`、`test_c2_total_observation_budget`、`test_c3_illuminator_budget`、`test_c4_receiver_and_fusion_processing_budgets`、`test_c5_protection_budget`、`test_c7_c8_reporting_feasibility_and_budget`、`test_c9_every_target_is_scored_at_one_pfa` |
| §15 C10 belief-only 信息集 | ✅ | `test_selector_consumes_no_truth_geometry`、`test_scheduler_certificates_are_declared_belief_only`、`test_belief_is_stated_not_guessed` |
| §18 交替 + §19 ε 单调接受（**能力**具备） | 🟡 | `test_acceptance_rule_is_monotone_with_an_epsilon_gate`、`test_the_accepted_quantity_can_be_the_worst_target_probability` |
| §5 max-min 势（leximin，Schur-凹，边际恒正） | 🟡 已落地但门控关闭 | `test_the_worst_case_quantity_is_the_exact_minimum`、`test_the_potential_is_schur_concave_not_just_minimising`、`test_maxmin_is_gated_off_by_default` |

### 12.2 未闭合（7 条缺口）

| # | 条款 | 缺口 | 为什么难 |
|---|---|---|---|
| 1 | **§2** | **证书没有生产方**：`CertificateView` 只能由调用方注入，而 `experiments/` **零处**注入（`cancellation.mode='off'`）⇒ 式(1) 里的 (I_res, η) 是**常数 κ 占位**。接口形态达标、内容未接线 | 最根本。先有生产方才有耦合；且 TP-UIC 实测 37 dB < 常数 40 dB，接上先变差 |
| 2 | §12/(C6′) | selection 与 protection **未耦合**：发布选择器签名 `(cfg,base,tables,plan,mask)`，无 z、无 η_min | 依赖 #1 |
| 3 | §19 | 接受量不是 Φ=min_q P_D,q，而是 `task_objective()=selection_utility − λ_c·cost`（`audits/theory.py`） | 切换前需跑主场景对照 |
| 4 | §5 | 头条 `target-local-v1` **未启用** max-min（默认仍是 softmin τ=0.10 + 二次 deficit 罚） | 换目标函数形状 = C 类新假设 |
| 5 | §6 | 头条仍带 `selector.lambda_c=0.005` 的时延价格 ⇒ 通信开销被线性折进检测效用 | 正是提案要避免的审稿问题 |
| 6 | §18 后半 | 按目标 deficit 更新保护集 z（`δ_q=[P_D^req−P_D,q]_+`，取 top-K）未实现；交替只发生在 x 侧 | 依赖 #2 |
| 7 | §7 | 只有 `K_q^max`，没有 `K_q^min` | 提案自己说只在鲁棒实验用，优先级最低 |

### 12.3 提案本身的两处自查问题（写论文前必须改）

- **§17 原则二的算例不成立**：`(0.95,0.94,0.25)` 的和是 **2.14**，`(0.75,0.75,0.75)` 的和是
  **2.25** ⇒ **sum 也选第二组**，这个例子**没有**证明"sum 会容忍掉队"（原文 "2.14>2.25" 是笔误）。
  能成立的例子要更极端：`(0.99,0.99,0.20)` 和 = 2.18 vs `(0.72,0.72,0.72)` 和 = 2.16
  ⇒ sum 选前者（worst **0.20**），max-min 选后者（worst **0.72**）。
  ⇒ **max-min 的正当性别靠算例，靠任务定义立论**（"不允许任何目标掉队"是任务要求，不是数值巧合）。
- **性能预期要降下来**：实测 softmin→max-min 的配对 Δworst P_D 只有 **+0.0003**
  （稀缺预算下 +0.0000）。固定骨架的价值在**可解释性、消融、审稿**，**不在涨点**。

### 12.4 闭合顺序（决策价值 ÷ 成本）

1. **#1 §2 生产方**（TP-UIC 证书接线）—— 没有它，后面 5 条都是空转；先解决 37 dB < 需求 52.25 dB
2. **#2 C6′ 耦合** → **#6 z 的 deficit 更新**（同一条链，一起做）
3. **#3 接受量改 Φ**（可独立，需主场景对照）
4. **#5 去 λ_c** → **#4 启用 max-min**（论文门面，性能收益最小）
5. **#7 K_q^min**（只给鲁棒性实验）

---

## 11. 文档索引

| 文档 | 内容 |
|---|---|
| **`docs/TEST_STANDARD.md`** | **测试书写规范的唯一细节口径**（文件组织 / 骨架模板 / 断言档位 / 门控四项 / 拆分与防丢） |
| `.workbuddy/PERFORMANCE_ROOT_CAUSE_AUDIT.md` | 本轮性能根因审计（全量数字与脚本） |
| **`studies/direction1/`** | **方向 1（TP-UIC 本体，已归档）全部探究产物**：`scripts/` 探针 + `data/` 实测 + `docs/` 审计 + `README.md` 汇总索引 |
| **`studies/direction2/`** | **方向 2（协作融合）**：`scripts/`（`diag_fusion2_attribution.py` / `diag_fusion2_belief.py` / `run_robust_verdict.py`）+ `data/`（含 `robust_verdict_mc120`）+ `docs/AUDIT_DIRECTION2_ATTRIBUTION.md` |
| `studies/TEST_SPLIT_2026-09-21.md` | 6 个历史大测试文件的拆分记录（20 新文件 + 4 脚手架） |
| `studies/AUDIT_TESTS_AND_ARTIFACTS_2026-09-21.md` | 测试与产物的全量盘点（口径/结论） |
| ⚠️ `studies/direction1/docs/TPUIC_RESIDUAL_ACCOUNTING_AUDIT.md` | 残余干扰记账判决 + off-grid 反向检验（§5.5 的依据）—— **正文编码已损坏不可读**，结论见 `tests/test_direction1_convergence.py` |
| ⚠️ `studies/direction1/docs/AUDIT_DIRECTION1_DELTA.md` | **方向 1**：δ 门控实现 + δ/切向阶/步长/n_cpi 实测扫描与判决（§6 的依据）—— **正文编码已损坏不可读** |
| `.workbuddy/memory/MEMORY.md` | 长期约定（铁律、口径、坑） |
| `.workbuddy/memory/YYYY-MM-DD.md` | 每日工作日志（追加） |
| `release/V1_STABLE_MANIFEST.json` | 发布身份（**95** 冻结键） |
| `docs/SYSTEM_MODEL.md` | 系统模型 |

---

## 13. 方向隔离与归档卫生（2026-09-21 立）

| 编号 | 规则 | 门禁 |
|---|---|---|
| **A-1** | 每个优化方向独占 `studies/directionN/`：`README.md` 是**唯一入口**（必须汇总该方向全部数字）、`scripts/` 放一次性探针、**不进 CI**、`data/` 放实测产物、`docs/` 放审计 | `test_study_docs_hygiene.py::test_every_study_script_is_indexed_by_its_readme` |
| **A-2** | 文档里写出的 `data/<dir>` 必须真实存在（防改名/迁移后留死链） | `test_study_docs_hygiene.py::test_readme_data_references_exist` |
| **A-3** | 含中文的 md **禁止用 PowerShell 管道搬运**（`Get-Content \| Out-File` 按 GB18030 解码 ⇒ 双重编码，无法映射的字节变 `?`，信息永久丢失）。用 Write/Edit 工具或 Python 显式 `encoding="utf-8"` | `test_study_docs_hygiene.py::test_no_new_mojibake_documents` |
| **A-4** | **`tools/` 下的一次性探针多为 untracked** ⇒ 迁移必须先 Copy + MD5 校验再删源 | 人工 |
| **A-5** | **结论先落断言、再落文档**：文档会烂（已毁 5 份）、探针不进 CI；只有断言会 FAIL | §7 T3 |

⚠️ 已知损坏清单（不可逆，正文已加横幅，数字未丢）：`studies/direction1/docs/` 下
`AUDIT_DIRECTION1_DELTA.md`、`AUDIT_DIRECTION1_FAILURE_ROOT_CAUSE.md`、
`AUDIT_REAL_VS_EXPECTED_GAP.md`、`AUDIT_TPUIC_RESIDUAL_ACCOUNTING.md`、
`TPUIC_RESIDUAL_ACCOUNTING_AUDIT.md` —— **不要再尝试还原**，也**不要凭记忆重写正文**。
⚠️ `.workbuddy/` 不清理：它是项目数据目录 + CI 基线所在。
