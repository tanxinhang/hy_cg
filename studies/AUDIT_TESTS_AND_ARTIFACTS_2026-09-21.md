# 测试代码与结果文件审计（2026-09-21）

> 范围：`tests/`（测试代码）、`studies/direction{1,2}/`（结果文件与归档文档）。
> 方法：先盘点，再按"**有没有人引用 / 会不会腐烂 / 能不能被断言钉住**"三问筛，
> 最后把**腐烂变成会 FAIL 的断言**。

---

## 1. 总判

| 项 | 结论 |
|---|---|
| 测试套件 | **708 passed / 7 xfailed / 6 subtests**，165 s。**健康**：无 skip、无过期 xfail、无读外部数据文件（全部自洽，可离线重跑） |
| 测试**结构** | ⚠️ 29 文件 6027 行，**5 个文件占了 48%**（2921 行），其中最大 736 行 —— 集中度过高 |
| 结果文件 | 🔴 **5 份方向 1 审计文档编码损坏且不可恢复**（有字节丢失） |
| 引用完整性 | ⚠️ 发现 **1 处数据死链 + 3 处测试文档失效引用**（已修） |
| 数据资产 | ✅ direction1 12 组 / direction2 14 组，**全部有脚本归属**，仅 1 组是被取代的 n=2 冒烟（已标注） |

**一句话**：测试**跑得动且可信**，问题在**归档侧腐烂没人管**；本轮把腐烂的三类（编码、
死链、无主脚本）全部变成门禁断言。

---

## 2. 测试代码审计

### 2.1 规模分布

| 分组 | 文件数 | 行数 | 说明 |
|---|---:|---:|---|
| 历史大文件（>350 行） | 5 | **2921** | `test_target_local_v1` 736、`test_canonical_consistency` 689、`test_cancellation_tp_uic` 655、`test_cancellation_glrt` 437、`test_receiver_context` 404 |
| 优化模型门禁组 | 4+1 | 682 | `test_optimization_model_*` + `_optmodel_common.py`，**已在 350 行门禁内** |
| 方向 1/2 探究组 | 5 | 734 | 全部 <350 行 |
| 其余 | 15 | 1690 | 门禁/小契约 |

⚠️ 5 个历史大文件**暂不在 350 行门禁内**（`test_module_size.py` 只卡
`test_optimization_model_*`）。它们是"后续可能整批退役"的历史债。

### 2.2 耗时

前 5 慢：`test_aperture_manifold::...escape_fraction` **35.5 s**、
`test_cancellation_production_wiring::...receiver_shape` 15.5 s、
`test_perlink_rho`（2 项）10.2 / 9.8 s、`test_cancellation_glrt` 6.3 s。
合计 165 s —— **可接受**，暂无拆分/加标记的紧迫性。

### 2.3 三个正面性质（值得保住）

1. **无 skip、无过期 xfail**：7 个 xfailed 全部是 `strict=True` 的缺口登记（实现当天会 XPASS 报错）。
2. **不读外部数据**：全仓 `tests/` 里 `studies/`、`results/` 引用**只出现在 docstring**，
   没有一条断言依赖归档 CSV ⇒ 归档损坏**不影响**测试可信度。这是这次事故没更大的原因。
3. **公共随机数配对**：方向 1/2 的定论扫描都用同 seed 前缀，ΔP_D 可逐 trial 配对。

---

## 3. 结果文件审计

### 3.1 🔴 编码损坏（最严重）

`studies/direction1/docs/` 下 **5/6 份**文档为双重编码乱码：

```
AUDIT_DIRECTION1_DELTA.md              特征字符 150 处
AUDIT_DIRECTION1_FAILURE_ROOT_CAUSE.md 特征字符 316 处
AUDIT_REAL_VS_EXPECTED_GAP.md          特征字符 150 处
AUDIT_TPUIC_RESIDUAL_ACCOUNTING.md     特征字符 166 处
TPUIC_RESIDUAL_ACCOUNTING_AUDIT.md     特征字符 161 处
（DIRECTION1_CONVERGENCE.md —— 完好）
```

**成因**：UTF-8 字节被按 **GB18030** 解码后重新存盘（已用 `只记录` 的字节序列反推验证：
`b'\xe5\x8f\xaa\xe8\xae\xb0'` → GB18030+replace 恰好产出文件里观察到的
`U+9359 / U+E047 / U+E187 / U+8930`）。**无法映射的字节被替换成 `?` ⇒ 信息已丢**。

**恢复尝试（均失败，记录在 `.workbuddy/tmp_enc_probe*.py`）**：
`gb18030` 逆向 round-trip 只能拿回约 96%，残留 314 处替换字符，且**对齐仍错位一层**；
两轮迭代反而更差。⇒ **判定不可无损还原**。

**数字没丢**：全部结论在 `studies/direction1/README.md` §3（完好）与
`tests/test_direction1_convergence.py`（14 条断言）。
备份在 `.workbuddy/enc_backup/`；5 份文件已加损坏横幅。

### 3.2 引用完整性

| 问题 | 位置 | 处理 |
|---|---|---|
| `data/diag_attribution_gain/` 目录不存在（且 `--profile gain` 这个 profile 也不存在） | `direction2/README.md:79` | ✅ 改为 `data/diag_attribution/` + 默认 profile |
| 3 处 docstring 指向 `.workbuddy/AUDIT_*.md`（早已迁到 `studies/direction1/docs/`，且其中 2 处目标已损坏） | `test_direction1_convergence.py`、`test_direct_estimation_error.py`、`test_residual_accounting_mode.py` | ✅ 改指向 `studies/direction1/README.md` §3 |
| `results/sweep_direct_estimation_delta/delta_sweep.json`（`results/` 顶层已空） | 损坏文档内 | 由横幅覆盖；数字在 README §3.4 |

### 3.3 数据资产盘点

| 方向 | 数据组 | 脚本 | 归属 |
|---|---:|---:|---|
| direction1 | 12 组（含 2 组已作废仅留痕） | 11 | ✅ 全部被 README 索引 |
| direction2 | 14 组 | 3 | ✅ 全部被 README 索引 |

孤儿 / 被取代：`data/robust_verdict/`（879 B，n=2 冒烟，已被 `robust_verdict_mc12` 取代）
→ 已在 README 标注"仅留痕"，**不删**（留痕数据有诊断价值）。

---

## 4. 已执行的收敛动作

| # | 动作 | 位置 |
|---|---|---|
| 1 | **新增归档卫生门禁**（3 条断言：不得新增乱码文档 / 引用的 `data/<dir>` 必须存在 / `scripts/` 每个脚本必须被 README 索引） | `tests/test_study_docs_hygiene.py` |
| 2 | **350 行门禁扩围**：`tests/test_optimization_model_*` → 增加 `tests/test_direction*` | `tests/test_module_size.py` |
| 3 | 修 3 处测试失效文档引用 | 3 个测试文件 |
| 4 | 修 1 处 README 数据死链 + 补 2 行（孤儿留痕、structural 冒烟） | `studies/direction2/README.md` |
| 5 | 5 份损坏文档加横幅 + 备份到 `.workbuddy/enc_backup/` | `studies/direction1/docs/` |
| 6 | 方向 1 README §8 改为"状态表"，标明 5 份损坏 / 1 份完好 + 数字保全口径 | `studies/direction1/README.md` |

**门禁结果**：全量 **708 passed / 7 xfailed**（+3 新断言）；
`check_release_identity --check` **CLEAN**（95 冻结键 0 违例）；`check_contract_refs` **32/32 CLEAN**。
**未改任何产品代码。**

---

## 5. 未做（需用户点头）

| 项 | 代价 | 风险 / 我的建议 |
|---|---|---|
| **重写 5 份损坏文档** | 高 | ⛔ **不建议**：只能凭 README + 测试重述，等于"凭记忆补正文"，有编造风险。宁可留"损坏"标记 |
| **拆 5 个 >350 行历史测试**（2921 行，48%） | 高 | ⚠️ 若这些文件确定不退役再做；否则拆了白拆 |
| **清理 `.workbuddy/` 200+ 临时文件** | 低 | ⛔ 不动：系统规定 `.workbuddy/` 是项目数据目录，且 CI 基线在其中 |
| **structural 两版补 MC=120**（各 ≈2 h） | 中 | 仅当方向 1 采纳 `structural` 为默认口径才需要 |

---

## 6. 一条方法论（下次直接套）

> **"文档里的数字"和"文档里的正文"要分开保全。**
> 这次正文全毁但结论无损，唯一原因是方向 1 收口时把结论**搬进了断言**
> （`test_direction1_convergence.py` 14 条）和 README 表格。
> 反过来，只写在审计文档正文里的东西（如 `results/` 路径）就跟着一起烂了。
> ⇒ **新结论先落断言，再落文档；文档只做索引。**
