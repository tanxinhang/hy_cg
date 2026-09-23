# 系统迭代手册（ITERATION_PLAYBOOK）

> 适用：`D:\Desktop\conference`（ICC 2027 投稿线）。
> 生效：2026-09-21。回答四个问题：**系统怎么迭代 / 在原有基础上怎么改 / 怎么新建测试文件 / 怎么管文件**。
> 与 `docs/DEV_STANDARD.md`（口径与准入）、`docs/TEST_STANDARD.md`（测试书写细节）配套；
> 本文件是**操作流程**，那两份是**判定标准**。三者冲突时：判定标准优先。

---

## 0. 三句话总纲

1. **先判后做**：任何改动先分诊（A/B/C/D）→ 过四问 → 预注册判据 → **然后才动手**。
2. **新能力 = 未登记新键 + 默认关闭**：发布路径逐位不变，是唯一能让你放心改动的护栏。
3. **结论落断言、文档做索引**：文档会烂（已毁 5 份），测试会 FAIL，所以"收口"的标志是
   **推翻它就有测试失败**，不是"写了篇文档"。

---

## 1. 仓库地图：往哪放、谁能依赖谁

### 1.1 目录职责

| 目录 | 职责 | 规模门禁 | 依赖方向 |
|---|---|---|---|
| `isac_sim/core/config/` | 配置源：`physical/radio/comm/interference/cancellation/aperture/detection/fusion/prior/selector/coordination/run` + 子包 `presets/`（base + successors）与 `validate/` | **≤150 行/文件** | 只依赖 `isac_sim` 内部 |
| `isac_sim/sensing/` | 物理与信号：`model/`（几何/双基/碰撞/链路表）、`dd/`、`waveform/`、`fbl/`、`soft_channel/` | ≤150 | 同上 |
| `isac_sim/receiver/` | 接收端对消：`cancellation/`（TP-UIC，41 文件）、`cancellation_glrt/`（18 文件） | ≤150 | 同上 |
| `isac_sim/detection/` | 检测与融合：`llr/`、`fusion/`、`corr/`、`oracle/` | ≤150 | 同上 |
| `isac_sim/scenario/`、`cooperation/`、`types/` | belief/truth 世界、协同原语与上报、数据结构与证书 | ≤150 | 同上 |
| `experiments/flow/` | **生产链路**：`pipeline.py`（唯一 production path）、`simulate.py`、`sweeps.py` | 冻结现状（只许减少） | → `isac_sim` |
| `experiments/app/` | CLI（`cli.py`）、绘图、报告、显示命名 | 冻结现状 | → `flow`、`isac_sim` |
| `experiments/methods/` | `roster.py`（方法花名册）、`policy.py`（策略分派） | 冻结现状 | → `flow` |
| `tests/` | 契约与门禁 | **≤350 行/文件** | 可 import 全部 |
| `studies/directionN/` | 单个优化方向的探究（README 唯一入口 + `scripts/` + `data/` + `docs/`） | 不进 CI | → 全部 |
| `tools/` | 一次性审计/诊断脚本（107 个 `.py`） | 冻结现状 | → 全部 |
| `results/` | 可引用的正式结果（26 个 `results_*` 目录） | — | 数据 |
| `release/` | `V1_STABLE_MANIFEST.json`（发布身份） | — | 数据 |
| `docs/` | 规范（本文件 + DEV_STANDARD + TEST_STANDARD + SYSTEM_MODEL） | — | 文档 |
| `.workbuddy/` | 记忆、CI 基线、预注册判据 —— **`.gitignore` 排除，不进版本控制** | — | 本地 |

### 1.2 三条硬边界

| 编号 | 规则 | 门禁 |
|---|---|---|
| **B-1** | **依赖单向**：`experiments → isac_sim`，反向禁止（实测 `isac_sim/` 内零处 `import experiments`） | `tests/test_layer_boundaries.py` |
| **B-2** | **禁止兼容层**：不加 `sys.modules` 别名、不加 `try: import ... except` 兜底、不留"旧名字转发" | 同 B-1 |
| **B-3** | **变体不加 CLI 标志**：CLI 表面只有 `--mode/--mc/--seed/--out/--set/--config/--values/--axis`。新增实验/变体走 `--set k=v` 或写进 `sweeps.py` 的覆盖字典（cli.py docstring 原话：*"Adding an experiment never adds a flag again."*） | 人工 |

---

## 2. 迭代生命周期（11 步，每步都有产出物与退出条件）

```
0  提议      ──▶ 一句话：改什么、期望看到什么数字变化
1  分诊      ──▶ A 可标定 / B 门控新功能 / C 假设变更 / D 架构重构   （DEV_STANDARD §1）
2  四问      ──▶ preset 已设? 死参数? 单/双世界? 分辨力够吗?        （DEV_STANDARD §1.1）
3  预注册    ──▶ 判据写进脚本 docstring + 常数一致性自查            （E1/E2）
4  定位      ──▶ 查 §3 落点速查表，确定要动哪些文件
5  实现      ──▶ B 类：未登记新键 + 默认关闭
6  测试      ──▶ 门控四项 + 关闭侧逐位不变                          （TEST_STANDARD §4）
7  门禁      ──▶ pytest（含 ⑤⑥⑦ 自研门禁）+ release_identity + contract_refs
8  实验      ──▶ 诚实口径、同 run 配对、oracle/上界对照；端到端先确认 production_wire
9  准入      ──▶ G1/G2/G3（+ 融合类加 J10）                         （DEV_STANDARD §5）
10 归档      ──▶ 数字进 results/ 或 studies/directionN/data/；结论进 tests/；索引进 README
11 提交      ──▶ git status 全量核对 → add → commit → push 后 ls-remote 实查
```

**退出条件**（一步没过不许进下一步）：3 没写判据不许跑 8；6 没写"关闭侧逐位不变"不许进 7；
7 有红不许进 8；9 没过不许写进论文数字。

### 2.1 迭代的三种粒度

| 粒度 | 什么时候用 | 走哪些步 |
|---|---|---|
| **微调**（改个默认值做对比） | 探索期 | 0 → 2 → 8 → 10（**不进生产默认**） |
| **正式改动**（要进论文/进默认） | 决定采纳 | 全 11 步 |
| **重构**（不动数值） | 结构烂了 | 0 → 1(D) → 4 → 5 → 7（必跑 `parity_check`） |

---

## 3. 在原有基础上修改：五类改法的落点模板

### 3.1 B 类：加一个门控新能力（最常见，务必照抄）

```
1) 加键     isac_sim/core/config/<段>.py  →  新字段 + 默认值(False/0.0/None)
2) 校验     需要时进 config/validate/<段>.py
3) 接线     生产链路 experiments/flow/simulate.py 的对应分支
            （门控必须接在 run_one_trial 层：proposed_c2f_adaptive_pd* 命中缓存时
              run_method_on_trial 内的改动对它无效）
4) 测试     tests/test_<主题>_<gate>.py：默认关 / 不在冻结清单 / 恒等条件 / 真的接线
5) 门禁     pytest + check_release_identity（新键未登记 ⇒ 必须 CLEAN）
```

⚠️ **config 文件的 150 行门禁余量极小**（实测 `cancellation.py` **148**、`selector.py` 129、
`__init__.py` 114）⇒ **加键常常要同时压注释**，否则撞线。压注释时最容易误删代码行
（踩过两次）⇒ 改完必须读回文件确认。

**接线位置的三条铁律**
- 门控要**广播**就学 `production_wire`：用 contextvar 让该 trial 内**所有**调用点拿到同一证书，
  否则会出现"一处实测、一处冻结"的**两个世界**（链路照跑、数字照出，但物理上自相矛盾）。
- 只改调度视野就接 `base_belief`，绝不动 `base_truth`（`robust` 门控的做法）。
- 新随机量必须走**独立随机流**（`[seed, 10**6+index]` / `rng.spawn`），否则关闭侧不再逐位不变。

### 3.2 加一个方法变体（不加 CLI 标志）

```
experiments/methods/roster.py     登记方法名（花名册）
experiments/methods/policy.py     分派策略
experiments/app/naming.py         进 METHOD_ORDER（表格/图里的显示顺序与可读名）
跑法：python run_isac_sim.py --mode <mode> --set <键>=<值> ...
```

⚠️ 融合权重是按**方法名硬编码**的（`policy.py`），加变体时先确认它该继承哪一套；
别指望"加个名字就自动合理"。

### 3.3 改流程接线（`experiments/flow/`）

| 文件 | 行数 | 规则 |
|---|---|---|
| `simulate.py` | 1577 | **只许减少**：新逻辑优先新文件/新函数，不往里堆 |
| `sweeps.py` | 1361 | 实验变体写在这里的覆盖字典里 |
| `selection.py` | 1346 | 同上 |
| `pipeline.py` | 143 | **唯一 production path**，改它 = 改论文系统口径（C 类） |

⚠️ `simulate.py` 里 belief 与 truth 是**两套表**；任何只建一套的改动 = 静默跑在完美先验世界。

### 3.4 D 类：重构（不动数值）

```
1) 改前快照：基线 CSV / 关键数字（P_D、P_FA、bits、κ）逐位记下来
2) 结构：x.py → x/ 同名包（__init__.py + 子模块），调用点不动
3) 逐位：tools/parity_check.py 必跑
4) 门禁：150 行 + 分层边界（不能出现 sys.modules 别名 / try-except 兜底）
5) 注释中文化（S3）
```
⚠️ 按顶层 class 自动切分会丢顶层 `def` —— 切前先把函数搬进库或脚手架。

### 3.5 C 类：改假设 / 改物理量 / 改默认值

必须同时做三件事：① 在 `DEV_STANDARD` §6 假设册登记；② 重跑**全部**受影响结果；
③ 更新 manifest 并打 tag。改默认门控值也算 C 类（会动发布数字）。

---

## 4. 新建测试文件：决策树与模板

### 4.1 先问：它配不配做测试？

| 判据 | 结论 |
|---|---|
| 推翻它有后果吗？（有东西会错/会悄悄坏） | ✅ 写成测试 |
| 只是一次性探索、结论尚不稳 | ❌ 放 `studies/directionN/scripts/`，且必须被 README 索引 |
| 只是"跑出来一个数" | ❌ 数字进 `data/`，索引进 README（钉数字不是契约） |

### 4.2 落点决策树

```
已有 test_<同主题>_* 文件吗？
├─ 有 ──▶ 加进去（文件 <350 行时）──▶ 行数超了？按主题再拆一个切面文件
└── 没有 ─▶ 新建 tests/test_<主题>_<切面>.py
             主题 = 被测对象（cancellation_tp_uic / direction2_robust）
             切面 = 契约类别（arms / protection / oracle / gate / model …）
```

**现有命名规律（照抄）**：`test_<主题>_<切面>.py`，一个主题一组文件
（如 `test_cancellation_tp_uic_{arms,protection,oracle}.py`），
共用工厂下沉 `tests/_<topic>_common.py`（`_tpuic` / `_glrt` / `_receiverctx` / `_prodwire` / `_optmodel`）。

### 4.3 选哪种测试形态

| 形态 | 用在哪 | 写法 |
|---|---|---|
| **契约测试** | 钉"接线关系/恒等/单调/下界" | `unittest.TestCase` class 分组 + `if __name__ == "__main__"` |
| **门禁测试** | 钉"仓库结构/行数/卫生" | 裸 `assert` 函数（如 `test_module_size.py`、`test_test_file_hygiene.py`） |
| **缺口登记** | 提案要求但代码没做到 | `@pytest.mark.xfail(strict=True)` + 写明"为什么难" |

### 4.4 最小模板 + 必跑

```python
"""<钉什么契约>。

背景：<为什么值得钉 / 上次踩的坑>。
门控语义：<默认关 / 只改哪一侧 / 已发布数字逐位不变>。
不覆盖什么：<端到端增益需 MC≥N，放 studies/…>。
"""
from __future__ import annotations
import unittest

GATE = "prior.<key>"          # 常量提顶


def _cfg(**extra):            # 小几何 + 固定 seed + 不跑 MC
    ...


class GateDefault(unittest.TestCase):
    def test_gate_exists_and_defaults_to_off(self) -> None:
        ...


if __name__ == "__main__":
    unittest.main()
```

新建后必跑：`pytest tests/test_新文件.py`（应 ≤5 s）→ 全量 → `check_release_identity` →
`check_contract_refs`。拆/改名时同 commit 维护 `tests/_test_inventory.py`。
细节口径见 `docs/TEST_STANDARD.md`。

---

## 5. 文件管理

### 5.1 新文件往哪放（决策表）

| 你要放的东西 | 放哪 | 命名 | 入 git? |
|---|---|---|---|
| 积木/算法代码 | `isac_sim/<域>/…` | `snake_case.py`，≤150 行 | ✅ |
| 流程/实验编排 | `experiments/flow|app|methods/` | 同上 | ✅ |
| 契约测试 | `tests/` | `test_<主题>_<切面>.py`，≤350 行 | ✅ |
| 测试脚手架 | `tests/` | `_<topic>_common.py` | ✅ |
| 方向探究脚本 | `studies/directionN/scripts/` | `<动作>_<对象>.py`，**必须被 README 索引** | ✅ |
| 方向实测数据 | `studies/directionN/data/<脚本名>_<变体>/` | 脚本名前缀 + 变体后缀 | ✅（小） |
| 正式结果（可引用） | `results/results_<实验>_<变体>/` | 同上；进 README/论文 | ✅ |
| 冒烟/预览（MC≤2） | `results_*_smoke/` | `.gitignore` 已排除 | ❌ |
| 一次性审计/诊断 | `tools/` | `<动作>_<对象>.py` | ✅ |
| 规范文档 | `docs/` | `UPPER_SNAKE.md` | ✅ |
| 记忆 / CI 基线 / 预注册判据 | `.workbuddy/` | — | ❌（已被忽略） |
| 临时产物 | `/tmp/`（已忽略） | — | ❌ |

### 5.2 版本控制规则

| 编号 | 规则 |
|---|---|
| **V-1** | **源码、规范、测试、方向产物必须入库**：`isac_sim/`、`experiments/`、`tests/`、`docs/`、`studies/`、`tools/`、`release/` 下不得有"从没 add 过"的文件 |
| **V-2** | **删除必须登记**：重构把旧模块换成包之后，旧路径的删除要一起 `git add`，否则 clone 出来是重构前的旧结构（porcelain 里 ` D` 违规、`D ` 合规） |

**这两条已做成门禁**：`tests/test_repo_tracking.py`（2 条，1.3 s；非 git 环境自动跳过）。
2026-09-21 首次运行即为红 —— 实测发现 `docs/` 与 `studies/` 两个目录**整体从未入库**，
重构删掉的 43 个 `isac_sim` 旧模块 + 19 个旧测试未登记删除，41 个新测试文件未入库；
已按用户裁决执行 `git add -A docs studies tests isac_sim experiments release tools`
（**只入索引，未 commit**），门禁转绿。
⚠️ 仍**未处理**的历史遗留（在 V-1/V-2 范围之外，需另行决定）：
`_archive/` 543 条删除、`archive/` 134 条、`results*/` 数百条、以及根目录一批旧 md
（`V1_STABLE_RELEASE.md`、`TP_UIC_V12.md`、`COORDINATION_WIRING.md` …）显示为已删除。
| **V-3** | `.workbuddy/` **永远不入库**（记忆 + CI 基线）。⇒ 逐位门禁在 CI 里跑不了，CI 只有三道环境无关门禁，这是**已知且接受**的 |
| **V-4** | 大产物/中间物走 `.gitignore`（`*_smoke/`、`/tmp/`、`*.bak`、`ppt/**/*.pptx`），不靠"记得不 add" |
| **V-5** | 提交前 `git status --porcelain` **全量核对**（本仓曾长期 1500+ 条未提交脏状态）；推送后用 `git ls-remote origin refs/heads/main` 实查 |

### 5.3 删除规则

1. 先确认它是不是**受版本控制**（`git ls-files`）—— `tools/` 下的一次性探针历史上有 13/13 **全部未跟踪**，
   直接删就是永久丢失。
2. 迁移类删除：`Copy` + **MD5 逐文件校验 0 失配** → 再删源。
3. `Remove-Item` 有 safe-delete 钩子，**常报错但文件实际已删** ⇒ 必须 `Test-Path` 复查，别重复操作。
4. 删测试 = 删契约：**必须同 commit 改 `tests/_test_inventory.py` 基线**，让删除可审查。

### 5.4 文档与归档卫生

- 每个 `studies/directionN/` 的 `README.md` 是**唯一入口**，必须汇总该方向全部数字并索引每个脚本。
- md 里写出的 `data/<dir>` 必须真实存在（防死链）。
- **含中文的 md 禁止用 PowerShell `Get-Content | Out-File` 搬运**（按 GB18030 解码 ⇒ 双重编码，
  已毁 5 份且不可逆）。用 Write/Edit 或 Python 显式 `encoding="utf-8"`。
- 盘点行数用 Python `splitlines()`，别用 `Measure-Object -Line`（少算末尾无换行的文件）。

---

## 6. 落点速查表（改 X 要动哪些文件 + 跑哪些门禁）

| 要改的东西 | 主落点 | 还要动 | 必跑 |
|---|---|---|---|
| 配置项 / 预设 | `isac_sim/core/config/<段>.py`、`presets/{base,successors}.py` | `validate/<段>.py`、`release/` manifest（若属冻结键） | pytest + `check_release_identity` |
| 接收机对消 | `isac_sim/receiver/cancellation/*` | 证书/链路表入口、`result.py` 字段 | pytest + `test_cancellation_tp_uic_*` |
| 检测/融合 | `isac_sim/detection/{llr,fusion,corr}/*` | `experiments/methods/policy.py` | pytest + 端到端 MC |
| 生产链路 | `experiments/flow/pipeline.py`（唯一 production path） | `simulate.py` | pytest + `parity_check` |
| 实验变体 | `experiments/flow/sweeps.py` 覆盖字典 | —（**不加 CLI 标志**） | 该实验本身 |
| 方法名/显示 | `experiments/methods/roster.py` + `app/naming.py` | — | `test_method_roster.py` |
| 论文数字 | `results/results_*/` | 任何改默认值的动作都是 C 类 | 全量 + 重跑受影响结果 |
| 发布身份 | `release/V1_STABLE_MANIFEST.json` | 打 tag | `check_release_identity --check` |
| 契约区间 | `tools/check_contract_refs.py` 的区间**必须手改**（符号出现在别处不算契约） | — | `check_contract_refs` |

---

## 7. 一页 checklist（打印版）

```
[ ] 提议：一句话说清"改什么 + 期望看到什么变化"
[ ] 分诊 A/B/C/D（不清按更高）  +  四问（preset/死参数/双世界/分辨力）
[ ] 判据预注册（写进脚本 docstring）+ 常数一致性自查
[ ] 查 §6 落点速查表，确认要动的文件
[ ] 实现：新键默认关闭；新随机量走独立流；门控接在正确的世界（belief vs truth）
[ ] 测试：门控四项 + 关闭侧逐位不变；≤350 行；共用品进 _common
[ ] 门禁：pytest + release_identity + contract_refs（+ parity，D 类）
[ ] 实验：诚实口径 + 同 run 配对 + oracle/上界；端到端先开 production_wire
[ ] 准入：G1/G2/G3（融合类再加 J10）
[ ] 归档：数字进 results/ 或 studies/directionN/data/；结论进 tests/；索引进 README
[ ] 文件：新文件入库、删除登记、状态干净（git status --porcelain）
[ ] 提交：commit → push → git ls-remote 实查
```
