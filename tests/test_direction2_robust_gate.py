"""方向 2 的 ``prior.robust_geometry_for_scheduler`` 门控接线契约。

门控语义：**只改调度器的视野**（链路选择 / 融合节点分配 / 预算看到的双基增益
换成 belief 位置误差球上的下界），检测侧照旧用 ``base_truth``。默认关闭 ⇒
已发布数字逐位不变。

这里钉的是**接线契约**（调用次数、恒等条件、检测侧不受污染），不是端到端
增益 —— 后者需要 MC≥120，放在 ``studies/direction2/`` 的数据里。
"""

from __future__ import annotations

import copy
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

from isac_sim.core.config import Config, apply_overrides, apply_preset
from isac_sim.scenario import belief as belief_pkg
from experiments.flow.simulate import run_one_trial

REPO = Path(__file__).resolve().parents[1]
GATE = "prior.robust_geometry_for_scheduler"
METHOD = "proposed_c2f"


def _cfg(**extra) -> Config:
    cfg = apply_preset(Config(), "paper-canonical")
    cfg.scale.M, cfg.scale.Q = 4, 2
    cfg.detect.num_false_per_target = 1
    cfg.run.seed = 4242
    return apply_overrides(cfg, extra) if extra else cfg


class GateDefault(unittest.TestCase):
    def test_gate_exists_and_defaults_to_off(self) -> None:
        self.assertFalse(Config().prior.robust_geometry_for_scheduler)
        self.assertFalse(apply_preset(Config(), "paper-canonical")
                         .prior.robust_geometry_for_scheduler)

    def test_gate_is_not_in_the_frozen_release_manifest(self) -> None:
        """A new, unregistered key is a *free* change; this keeps it that way."""
        import json

        path = REPO / "release" / "V1_STABLE_MANIFEST.json"
        if not path.exists():
            self.skipTest("no release manifest yet")
        blob = json.loads(path.read_text(encoding="utf-8"))
        text = json.dumps(blob)
        self.assertNotIn("robust_geometry_for_scheduler", text)


class GateWiring(unittest.TestCase):
    def setUp(self) -> None:
        self.calls: list[int] = []
        self.real = belief_pkg.geometry_robust_base

        def spy(cfg, base):
            self.calls.append(1)
            return self.real(cfg, base)

        self._patch = unittest.mock.patch.object(
            belief_pkg, "geometry_robust_base", spy
        )
        self._patch.start()

    def tearDown(self) -> None:
        self._patch.stop()

    def _run(self, cfg: Config, trial: int = 0):
        self.calls.clear()
        res = run_one_trial(cfg, trial, methods=[METHOD])[METHOD]
        return res, len(self.calls)

    def test_off_never_calls_the_robust_view(self) -> None:
        _, n = self._run(_cfg(**{GATE: False}))
        self.assertEqual(n, 0)

    def test_on_calls_the_robust_view_once(self) -> None:
        _, n = self._run(_cfg(**{GATE: True}))
        self.assertEqual(n, 1)

    def test_on_is_the_identity_when_the_belief_is_exact(self) -> None:
        """sigma = 0 => no ball => nothing to be robust about."""
        base_cfg = _cfg(**{"prior.belief_sigma_pos_m": 0.0})
        off, _ = self._run(base_cfg)
        cfg_on = apply_overrides(copy.deepcopy(base_cfg), {GATE: True})
        on, _ = self._run(cfg_on)
        self.assertEqual(on.selected_links, off.selected_links)
        self.assertEqual(on.overhead_bits, off.overhead_bits)
        self.assertEqual(int(on.detected), int(off.detected))

    def test_on_is_the_identity_outside_belief_mode(self) -> None:
        base_cfg = _cfg(**{"prior.belief_mode": False})
        off, _ = self._run(base_cfg)
        cfg_on = apply_overrides(copy.deepcopy(base_cfg), {GATE: True})
        on, _ = self._run(cfg_on)
        self.assertEqual(on.selected_links, off.selected_links)
        self.assertEqual(int(on.detected), int(off.detected))


class GateEffect(unittest.TestCase):
    """The gate has to actually move the schedule, and only the schedule."""

    def _pair(self, cfg: Config, trial: int):
        off = run_one_trial(cfg, trial, methods=[METHOD])[METHOD]
        cfg_on = apply_overrides(copy.deepcopy(cfg), {GATE: True})
        on = run_one_trial(cfg_on, trial, methods=[METHOD])[METHOD]
        return off, on

    def test_gate_changes_the_schedule_when_the_prior_is_wrong(self) -> None:
        cfg = _cfg()
        moved = False
        for trial in range(6):
            off, on = self._pair(cfg, trial)
            if off.selected_links != on.selected_links:
                moved = True
                break
        self.assertTrue(
            moved, "the gate never changed a schedule in 6 trials -- dead wiring"
        )

    def test_gate_leaves_the_detection_world_untouched(self) -> None:
        """False-alarm bookkeeping is built from the truth world, not belief."""
        cfg = _cfg()
        for trial in range(3):
            off, on = self._pair(cfg, trial)
            self.assertEqual(int(on.total_false), int(off.total_false))
            self.assertEqual(int(on.total_targets), int(off.total_targets))
            self.assertEqual(int(on.active_targets), int(off.active_targets))

    def test_gate_is_deterministic(self) -> None:
        cfg = _cfg(**{GATE: True})
        a = run_one_trial(cfg, 1, methods=[METHOD])[METHOD]
        b = run_one_trial(cfg, 1, methods=[METHOD])[METHOD]
        self.assertEqual(a.selected_links, b.selected_links)
        self.assertEqual(a.overhead_bits, b.overhead_bits)


if __name__ == "__main__":
    unittest.main()
