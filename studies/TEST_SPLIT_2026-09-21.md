# 测试文件拆分记录（2026-09-21）

> 上一份审计（`studies/AUDIT_TESTS_AND_ARTIFACTS_2026-09-21.md`）的第 2.1 节判定
> 「5 个历史大文件占 48%，暂不在 350 行门禁内」。本轮把它们拆完并取消豁免。
>
> **先立防丢门禁，再动手拆** —— 顺序不能反。

---

## 1. 为什么先立门禁

拆测试文件的失败模式是**静默丢测试**：pytest 照常跑、总数看着差不多，
少了几条 `def test_*` 没人会发现，一条契约就此停止被检查。

所以第一件事不是拆，是冻结"拆之前每个文件拥有哪些测试名"：

| 文件 | 内容 |
|---|---|
| `tests/_test_inventory.py` | 自动生成。6 个超限文件各自拥有的测试名，共 **155 条**。key 是**原文件名**，拆完也不需要更新 —— 它只是出处记录 |
| `tests/test_test_inventory.py` | 3 条断言：①每个基线名字仍在 `tests/` 某处定义；②测试定义总数不低于 155；③基线自洽 |

用 AST 数 `def test_*` 而不跑 `--collect-only`：口径自洽、无 pytest 嵌套、
0.17 s。参数化展开不影响计数（基线与实际同口径）。

⚠️ 基线**故意不随拆分更新**。删测试仍允许，但必须在同一个 commit 里改基线，
这样"删掉一条契约"是个可审查的动作，而不是搬运事故。

---

## 2. 拆之前先修正盘点

上一份审计的行数是 `Get-Content | Measure-Object -Line` 数出来的，**低估了
60~120 行**，还漏了一个文件。用 Python `splitlines()` 重数：

| 文件 | 上轮记录 | 实测 | 差 |
|---|---:|---:|---:|
| `test_target_local_v1.py` | 736 | **825** | +89 |
| `test_cancellation_tp_uic.py` | 655 | **773** | +118 |
| `test_canonical_consistency.py` | 689 | **752** | +63 |
| `test_cancellation_glrt.py` | 437 | **512** | +75 |
| `test_receiver_context.py` | 404 | **473** | +69 |
| `test_cancellation_production_wiring.py` | *漏报* | **390** | — |

⇒ 实际是 **6 个文件 3335 行**，不是 5 个 2921 行。教训：盘点工具本身也要校对，
`Get-Content` 对末尾无换行的文件会少算。

---

## 3. 拆分结果

原则：**方法体逐字搬运，只改文件边界**；能保持 class 名的就保持
（`ObjectiveAndMomentTests` 未改名 ⇒ node id 稳定）；工厂函数下沉到
`tests/_<topic>_common.py`（沿用 `_optmodel_common.py` 范式）。

| 原文件 | 行数 | 拆成 | 各组行数 |
|---|---:|---|---:|
| `test_canonical_consistency.py` | 752 | `test_canonical_presets` / `_selection` / `_moments` | 217 / 294 / 299 |
| `test_target_local_v1.py` | 825 | `test_target_local_v1_preset` / `_allocation` / `_bundle` / `_end_to_end` | 8 / 7 / 8 / 9 条 |
| `test_cancellation_tp_uic.py` | 773 | `test_cancellation_tp_uic_arms` / `_protection` / `_oracle` | 16 / 9 / 9 条 |
| `test_cancellation_glrt.py` | 512 | `test_cancellation_glrt_model` / `_identifiability` | 13 / 10 条 |
| `test_receiver_context.py` | 473 | `test_receiver_context_belief` / `_bridge` | 6 / 8 条 |
| `test_cancellation_production_wiring.py` | 390 | `test_cancellation_production_denominator` / `_numerator` | 9 / 7 条 |

新增脚手架：`_tpuic_common.py`、`_glrt_common.py`、`_receiverctx_common.py`、
`_prodwire_common.py`（都以 `_` 开头，pytest 不收集）。

**行数开销**：拆完总计比原来多约 100 行（每个新文件重复的 import 与模块
docstring）。这是为可读性付的、可接受的税。

---

## 4. 分组依据（不是按行数均分）

按**主题**切，不按行数切。以 `test_target_local_v1.py` 为例：

- **preset** —— preset 差异、波形隔离、几何视图、运行计时
- **allocation** —— 融合节点分配、容量上限、本地锚点、弱目标固定顺序
- **bundle** —— 联合 oracle、restricted master、列生成、RCS 稳健
- **end-to-end** —— polish 单调性、worker 数确定性、分布式投标、校准回放

这样切的好处是新加一条契约时**落点唯一**；按行数均分则每次都要重新判断。

---

## 5. 门禁扩围

`tests/test_module_size.py` 的 `GATED_TEST_PREFIXES` 被**删除**，
`test_new_test_modules_are_at_most_350_lines` 改为 `test_test_modules_are_at_most_350_lines`，
对 `tests/` 全目录生效。

**当前：0 个文件超过 350 行。**

---

## 6. 验证

| 检查 | 结果 |
|---|---|
| 防丢门禁（155 条基线） | ✅ 每次拆分后单独跑过 |
| 各新文件单独跑 | ✅ 39 / 35 / 37 / 26 / 17 / 19 passed |
| 全量回归 | 见下 |
| `check_release_identity --check` | CLEAN |
| `check_contract_refs` | 32/32 CLEAN |

**未改任何产品代码** —— 全部改动都在 `tests/` 内。

---

## 7. 方法论

> **拆之前先冻结"有什么"，拆之后才谈"在哪里"。**
> 拆分的失败模式不是报错，是静默少几条。防丢基线让这个失败模式变成红灯。
> 同理适用于任何"搬运型重构"：搬运的正确性不能靠"我数过了"。

排在三问之后的关键一问：**这些文件会不会退役？** 上一轮我写"若确定不退役
再做，否则拆了白拆"——本轮先确认这 6 个文件全是**活契约**（发布口径一致性、
TP-UIC 不变量、生产接线等价性），不存在退役选项，才动的手。
