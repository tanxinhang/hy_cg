# 测试书写规范（TEST_STANDARD）

> 适用：`D:\Desktop\conference` 的 `tests/` 全目录。
> 生效：2026-09-21（6 个历史大文件按主题拆完后立）。与 `docs/DEV_STANDARD.md` §7 配套，
> 本文件是**唯一细节口径**；两者冲突时以本文件为准。
> 现状基线（2026-09-21）：48 个文件、**718 passed / 7 xfailed**、全量 **≈167 s**、0 个文件超 350 行。

---

## 0. 三条定位原则

| 编号 | 原则 | 为什么 |
|---|---|---|
| **P1** | **测试是结论的载体，不是代码的附属品**。一条结论算收口 ⇔ **推翻它就有测试 FAIL**。 | 文档不会失败、探针不进 CI。方向 1 五份审计文档被编码损坏而结论无损，唯一原因就是结论已落成断言 |
| **P2** | **钉契约，不钉数字**。钉"接线关系/恒等条件/单调性/下界性"，不钉"某次跑出来的 P_D=0.43"。 | 数字随口径（κ、记账、MC）漂移；契约不漂移 |
| **P3** | **新增即门禁**。新能力、新判据、新坑，先想"它怎么变成会 FAIL 的断言"，再想文档。 | 只写进 md 的东西会烂（已发生一次） |

---

## 1. 文件组织与命名

| 编号 | 规则 | 检查方式 |
|---|---|---|
| **F1** | 文件名 `test_<主题>_<切面>.py`，主题取自被测对象，切面取自契约（如 `test_direction2_robust_gate.py`） | 人工 |
| **F2** | **一条链路一个文件**；同一主题的不同切面拆成多个文件，不塞进一个 | 350 行门禁 |
| **F3** | 每个文件 **≤350 行**（`tests/` 全目录纯红线，无豁免，不含脚手架） | `tests/test_module_size.py::test_test_modules_are_at_most_350_lines` |
| **F4** | 共用工厂/常量下沉 `tests/_<topic>_common.py`（下划线开头 ⇒ 不参与收集） | 同 F3 |
| **F5** | **禁止从 `test_*.py` 模块 import**（会重复收集、制造隐式耦合）⇒ 只 import `_common` | 人工 + 收集数不翻倍 |
| **F6** | 拆分/删改测试名，必须同 commit 更新 `tests/_test_inventory.py` 基线 | `tests/test_test_inventory.py` |
| **F7** | 新结论优先落**比值型 / 结构性断言**，不要复制一份探针到 `scripts/` | §3 |

**可静态检查的部分已做成门禁**：`tests/test_test_file_hygiene.py` 用 AST 钉住
W1（必须有模块 docstring）/ W2（测试名与 class 名 ASCII）/ W6（unittest 风格文件必须有
`__main__` 守卫）/ F4（脚手架 `_` 开头）/ F5（禁止从 `test_*` import）。
判定不了的（断言档位、性能预算、docstring 内容质量）留在 review。

---

## 2. 文件骨架（照抄这个结构）

```python
"""<一句话：钉什么契约>。

背景（为什么值得钉）：<触发它的事件 / 上次踩的坑>。
门控语义：<默认关 / 只改哪一侧 / 已发布数字逐位不变>。

这里钉的是**接线契约**（调用次数、恒等条件、检测侧不受污染），不是端到端增益 ——
后者需要 MC≥120，放在 ``studies/directionN/`` 的数据里。
"""
from __future__ import annotations

import copy
import unittest
from pathlib import Path

import numpy as np

from isac_sim.core.config import Config, apply_overrides, apply_preset

REPO = Path(__file__).resolve().parents[1]
GATE = "prior.robust_geometry_for_scheduler"     # 常量提顶，不散落在断言里


def _cfg(**extra) -> Config:                      # 工厂：小几何、固定 seed
    cfg = apply_preset(Config(), "paper-canonical")
    cfg.scale.M, cfg.scale.Q = 4, 2
    cfg.run.seed = 4242
    return apply_overrides(cfg, extra) if extra else cfg


class GateDefault(unittest.TestCase):             # 一个 class = 一个契约切面
    def test_gate_exists_and_defaults_to_off(self) -> None:
        self.assertFalse(Config().prior.robust_geometry_for_scheduler)


if __name__ == "__main__":                        # 允许单文件直跑
    unittest.main()
```

| 编号 | 规则 |
|---|---|
| **W1** | **模块 docstring 必写三件事**：钉什么契约 / 为什么（背景、踩过的坑）/ **不覆盖什么**（把端到端增益、MC 定论明确排除，避免后来者误以为它被钉住了） |
| **W2** | **测试函数名一律英文、描述性长名**（`test_gate_is_the_identity_when_the_belief_is_exact`），读名即知断言；class 名英文 |
| **W3** | docstring 正文**中文**（写"为什么"），拆出来的历史文件里的英文 docstring 不回改 |
| **W4** | 一个文件内**只用一种风格**：要么 `unittest.TestCase` class 分组（契约类，推荐），要么裸 `assert` 函数（门禁类，如 `test_module_size.py`）。不要混 |
| **W5** | 常量（门控键、方法名、路径）提到模块顶，断言里不写字面量 |
| **W6** | 结尾 `if __name__ == "__main__": unittest.main()` |

---

## 3. 断言写法：契约优先

优先级从高到低，**能用上一档就别用下一档**：

| 档 | 断言类型 | 例子 |
|---|---|---|
| **1** | **结构/签名探针** —— 把"耦合不存在"变成会翻面的断言 | `inspect.signature(fn).parameters & {"z","eta_min"}` 为空 ⇒ 记 xfail |
| **2** | **调用次数 / 分支触发**（spy 包一层计数） | 门控关 ⇒ `n_calls == 0`；开 ⇒ `== 1` |
| **3** | **恒等条件**（退化输入下必须逐位相等） | σ=0 / belief_mode=False / δ=0 ⇒ on 与 off 结果相同 |
| **4** | **单调性 / 下界性 / 可加性** | 势函数 Schur-凹；`i_res_structural ≤ i_res` |
| **5** | **比值型断言**（占 99.8% 的那一项在哪） | `estimation / structural > 100` |
| **6** | **绝对阈值**（只在有 oracle / upper-bound 对照时才写） | oracle 闭合 P_FA ≤ 0.06 |

| 编号 | 禁区 |
|---|---|
| **A1** | ⛔ **不钉"某次跑出来的端到端数字"**当作契约（P_D=0.43 这种）。要钉就钉它与对照臂的**关系**，或明确写成 xfail 缺口 |
| **A2** | ⛔ **不做无上界的 `assertAlmostEqual`**（places 放太大等于没测）⇒ 逐位比较就说逐位，并注明**环境绑定**（py3.11.0 + numpy 2.2.6） |
| **A3** | ⛔ **不用 `skip` 掩盖失败**。`skipTest` 只允许用于"资产不存在/环境不具备"（如 release manifest 缺失），并写明原因 |
| **A4** | ⛔ **不在测试里 `print` 或写文件到仓库**；临时产物用 `tmp_path` / `TemporaryDirectory` |
| **A5** | ⛔ **不在测试里跑重实验**：禁 MC≥20、禁 `production_wire`（≈26 s/trial）、禁 MC 扫描 —— 那是 `studies/` 的事 |
| **A6** | ✅ 缺口用 **`@pytest.mark.xfail(strict=True)`** 登记（实现当天 XPASS 报错，逼你转常态断言），并在 docstring 写"为什么难" |

---

## 4. 门控（feature gate）类测试的必写四项

任何"未登记新键 + 默认关闭"的新能力，测试至少覆盖这四条（参考 `tests/test_direction2_robust_gate.py`）：

| 编号 | 断言 | 缺了会怎样 |
|---|---|---|
| **G-1** | **默认关闭**：`Config()` 与头条 preset 下均为 False | 悄悄改了发布数字 |
| **G-2** | **未在冻结清单里**：manifest JSON 中不含该键名 | 一不留神把 free 改动升级成 C 类 |
| **G-3** | **恒等条件**：退化输入下 on ≡ off（逐位） | 门控打在错误世界 ⇒ 污染检测侧（双世界陷阱） |
| **G-4** | **真的接线了**：非退化输入下，若干 trial 内至少改到一次调度 | 死接线（preset 已设 / 死参数坑） |

补充：**确定性**（同 seed 两次跑结果相同）、**只改一侧**（如 belief 侧动、truth 侧不动）也要钉。

---

## 5. 体积、拆分与防丢

### 5.1 拆分前置（顺序不能反）

```
1) 先证明它不是退役候选（全都是活契约才值得拆）
2) 立防丢基线：tests/_test_inventory.py 冻结原文件的全部 def test_* 名（key = 原文件名）
3) 按主题分组（不按行数均分）—— 新契约必须有唯一落点
4) 方法体逐字搬运，不改断言语义
5) 工厂/常量下沉 tests/_<topic>_common.py
6) 每拆完一个单跑 tests/test_test_inventory.py
7) 最后跑全量 + check_release_identity + check_contract_refs
```

### 5.2 为什么必须有防丢门禁

拆大文件的失败模式是**静默的**：pytest 照跑、总数看着正常，只是几个 `def test_*` 没搬过来，
症状是某个契约悄悄没人检查了。防丢门禁按 **AST** 扫描 `tests/*.py` 的全部 `def test*`
（排除 `_` 开头），逐名比对基线 ⇒ 丢一个就 FAIL。

⚠️ **按顶层 class 自动切分会丢顶层 `def`**（切丢过 `_state`）⇒ 切前先把顶层函数搬进脚手架。
⚠️ **基线是"出处"不是"配置"**：拆完**不更新**基线；有意删测试才改基线，且必须同 commit。

---

## 6. 运行与性能预算

| 项 | 口径 |
|---|---|
| 命令 | `E:/anaconda/3_11_python/python.exe -m pytest -q`（**逐位门禁只在这个环境下有结论**：3.11.0 + numpy 2.2.6） |
| 全量预算 | ≈164 s；明显变慢要查是不是有人偷偷跑了重实验 |
| 单文件预算 | 新文件 `pytest tests/test_xxx.py` 应 **≤5 s** |
| CI | 跑不了逐位门禁（基线在被 `.gitignore` 排除的 `.workbuddy/`）⇒ CI 是三道环境无关门禁 + 显式声明未覆盖项 |

---

## 7. 反模式清单（会被 review 打回）

| 反模式 | 正确做法 |
|---|---|
| 把结论写进 `studies/*/docs/*.md` 就收工 | 结论落成断言；md 只做索引（P1） |
| 新写一个 `scripts/diag_*.py` 探针验证一次性猜想 | 能变成断言的进 `tests/`；只是探索才进 `studies/directionN/scripts/` 且**必须被 README 索引** |
| 从 `test_*.py` import 共用代码 | `_<topic>_common.py`（F5） |
| 一个文件塞 5 个主题共 800 行 | 按主题拆（F2/F3） |
| 用绝对阈值钉住一次跑分 | 钉关系 / 用 xfail 登记（A1） |
| `skip` 掉一个红了但不想修的测试 | 修它，或 xfail(strict=True) 登记缺口并写原因（A3） |
| 测试里跑 MC / 开生产测量 | 交给 `studies/`（A5） |
| 用 PowerShell `Get-Content` 统计/搬运含中文的测试与文档 | Python `splitlines()` 统计；Write/Edit 工具搬运（GB18030 双重编码已毁 5 份文档） |

---

## 8. 新增/修改测试的 checklist

```
[ ] 1. 归主题：属于哪个已有 test_<主题>_* ？都不属于才新建文件（F1/F2）
[ ] 2. 写 docstring：钉什么 / 为什么 / 不覆盖什么（W1）
[ ] 3. 选断言档位：能用结构/签名/次数/恒等就别用绝对阈值（§3）
[ ] 4. 门控类：默认关 + 不在冻结清单 + 恒等条件 + 真的接线（§4）
[ ] 5. 固定 seed，禁止重实验（A5），单文件 ≤5 s（§6）
[ ] 6. 行数 ≤350；共用代码进 _common（F3/F4/F5）
[ ] 7. 重命名/删除 ⇒ 同 commit 改 tests/_test_inventory.py（F6）
[ ] 8. 全量 pytest + check_release_identity + check_contract_refs
```
