"""TP-UIC 生产接线的契约测试。

提案 §2 要求式(1) 的 ``(I_res, eta)`` 是**接收机产出的 belief-side 证书**，
而此前生产链路里它由冻结常数 ``interference.direct_cancellation_db`` 占位。
本文件测的是**接线本身**，不是数值：

* 门控关闭时上下文恒为空 ⇒ 逐位不变（发布路径不受影响）；
* 上下文一旦给出，**所有**链路表调用点消费同一个证书 —— 这正是
  ``tools/run_receiver_closed_loop.py`` 要消除的"两个世界"（一处实测、
  一处冻结，数字合理但接收机模型自相矛盾）；
* 显式传参优先于上下文（调用方的显式意图不被广播覆盖）。
"""

from __future__ import annotations

import numpy as np
import pytest

from isac_sim.core.config import Config, apply_overrides, apply_preset
from isac_sim.sensing.model.link_tables import compute_link_tables
from isac_sim.sensing.model.link_tables.trial_certificate import (
    TrialCertificate,
    current_trial_certificate,
    trial_certificate,
)

BASE = {
    "geometry.area_xy": 600.0,
    "geometry.h_uav_min": 200.0,
    "geometry.h_uav_max": 500.0,
    "geometry.h_target_min": 200.0,
    "geometry.h_target_max": 500.0,
    "geometry.comm_range": 1000.0,
    "detect.target_rcs": 0.1,
    "run.seed": 2026,
}


def _cfg(**extra):
    over = dict(BASE)
    over.update(extra)
    return apply_overrides(apply_preset(Config(), "target-local-v1"), over)


def _base(cfg, seed=7):
    from isac_sim.sensing.model import build_base_gains, generate_geometry

    rng = np.random.default_rng([cfg.run.seed, seed])
    return build_base_gains(cfg, generate_geometry(cfg, rng), rng)


class TestCertificateContext:
    """上下文的生命周期。"""

    def test_default_is_empty(self):
        """门控关闭（默认）时没有证书 —— 这是逐位不变的前提。"""
        assert current_trial_certificate() is None

    def test_scope_restores_on_exit(self):
        cfg = _cfg()
        m, q = cfg.scale.M, cfg.scale.Q
        cert = TrialCertificate(np.full(m, 0.5), np.ones((m, q)))
        with trial_certificate(cert):
            assert current_trial_certificate() is cert
        assert current_trial_certificate() is None

    def test_nested_scope_restores_outer(self):
        """方法内部再开一次上下文也不能把外层的证书弄丢。"""
        cfg = _cfg()
        m, q = cfg.scale.M, cfg.scale.Q
        outer = TrialCertificate(np.full(m, 0.5), np.ones((m, q)))
        inner = TrialCertificate(np.full(m, 0.1), np.ones((m, q)))
        with trial_certificate(outer):
            with trial_certificate(inner):
                assert current_trial_certificate() is inner
            assert current_trial_certificate() is outer


class TestBroadcastToEveryCallSite:
    """一个 trial 只测一次接收机，所有调用点必须消费同一个证书。"""

    def test_context_changes_the_receiver_model(self):
        """上下文给出证书后，链路表的干扰项必须与默认路径不同。"""
        cfg = _cfg()
        base = _base(cfg)
        m, q = cfg.scale.M, cfg.scale.Q

        plain = compute_link_tables(cfg, base)
        cert = TrialCertificate(np.full(m, 0.5), np.ones((m, q)))
        with trial_certificate(cert):
            wired = compute_link_tables(cfg, base)

        # 删除 direct_cancellation_db 后默认路径是**没有对消**（fraction=1），
        # 所以接线必然把残余压下去；方向反过来就说明证书没被消费。
        assert float(np.median(wired.rinr)) < float(np.median(plain.rinr))

    def test_coarse_and_fine_tables_share_one_certificate(self):
        """粗表与细表（dd_gain 不同）必须拿到同一个接收机模型。"""
        cfg = _cfg()
        base = _base(cfg)
        m, q = cfg.scale.M, cfg.scale.Q
        cert = TrialCertificate(np.full(m, 0.5), np.ones((m, q)))

        with trial_certificate(cert):
            coarse = compute_link_tables(cfg, base)
            fine = compute_link_tables(cfg, base, dd_gain=base.eta_fine)

        plain_coarse = compute_link_tables(cfg, base)
        plain_fine = compute_link_tables(cfg, base, dd_gain=base.eta_fine)

        # 两处都得偏离各自的默认路径；只偏一处就是"两个世界"。
        assert float(np.median(coarse.rinr)) < float(np.median(plain_coarse.rinr))
        assert float(np.median(fine.rinr)) < float(np.median(plain_fine.rinr))

    def test_retention_reaches_the_numerator(self):
        """回波存活率必须走分子：只给分母等于把对消器夺走的能量仍记给目标。"""
        cfg = _cfg()
        base = _base(cfg)
        m, q = cfg.scale.M, cfg.scale.Q

        full = TrialCertificate(np.full(m, 1e-4), np.ones((m, q)))
        half = TrialCertificate(np.full(m, 1e-4), np.full((m, q), 0.5))
        with trial_certificate(full):
            t_full = compute_link_tables(cfg, base)
        with trial_certificate(half):
            t_half = compute_link_tables(cfg, base)

        # 分母相同、分子减半 ⇒ 感知 SINR 必然下降。
        assert float(np.median(t_half.gamma_sense)) < float(
            np.median(t_full.gamma_sense)
        )


class TestExplicitArgumentsWin:
    """广播不能覆盖调用方的显式意图。"""

    def test_explicit_fraction_beats_context(self):
        cfg = _cfg()
        base = _base(cfg)
        m, q = cfg.scale.M, cfg.scale.Q
        cert = TrialCertificate(np.full(m, 0.5), np.ones((m, q)))

        with trial_certificate(cert):
            explicit = compute_link_tables(
                cfg, base, residual_fraction_by_receiver=np.full(m, 1e-3)
            )
        # 显式给出的是 1e-3，而不是上下文的 0.5 ⇒ 残余必须比上下文路线小。
        with trial_certificate(cert):
            from_context = compute_link_tables(cfg, base)
        assert float(np.median(explicit.rinr)) < float(
            np.median(from_context.rinr)
        )


class TestGateDefaultsOff:
    """新增配置键必须默认关闭，且不进入冻结键集合。"""

    def test_defaults(self):
        cfg = _cfg()
        assert cfg.cancellation.production_wire is False
        assert cfg.cancellation.production_wire_arm == "tp_uic_full"
        assert cfg.cancellation.production_wire_retention == "q"

    def test_release_identity_unaffected(self):
        """新键是未登记的，不得出现在 96 个冻结键里。"""
        import json
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        man = json.loads(
            (root / "release" / "V1_STABLE_MANIFEST.json").read_text(encoding="utf-8")
        )
        frozen = man.get("frozen", [])
        flat = json.dumps(frozen, ensure_ascii=False)
        assert "production_wire" not in flat


def _measure(cfg, receivers=(0, 1)):
    """在少数接收机上跑一次测量（全部接收机会让测试慢 15 倍）。"""
    from isac_sim.receiver import cancellation as cx
    from isac_sim.sensing.model import build_base_gains, generate_geometry

    rng = np.random.default_rng([cfg.run.seed, 7])
    geom = generate_geometry(cfg, rng)
    base = build_base_gains(cfg, geom, rng)
    ctx = cx.ReceiverContext.from_trial(cfg, geom, geom, base, arm="tp_uic_full")
    return cx.measure_receiver_context(
        ctx, rng=np.random.default_rng([cfg.run.seed, 10 ** 6]),
        receivers=list(receivers),
    )


class TestArmPruningIsBitIdentical:
    """装配剪枝是**工程提速**：它只许跳过没被读的臂，不许改被测臂的数值。

    生产只读 ``tp_uic_full``，而 ``cancellation_arms`` 原本装配八个臂。剪枝
    带来的收益是 1.31×（35.3 → 26.9 s），代价是"少装配"这个行为本身可能
    意外改变数值 —— 例如某个臂在装配时往 ``ctx`` 上挂了后续会读到的中间
    结果。所以等价性必须是**逐位**的，不是 approx，而且要有测试钉住：
    注释承诺过的东西会过期，断言不会。
    """

    FIELDS = ("fraction", "kappa_db", "eta_survive", "eta_survive_q",
              "eta_protect", "i_res", "i_in", "i_res_pred", "i_in_pred")

    def test_pruned_matches_full_assembly(self):
        full = _measure(_cfg(**{"cancellation.measure_prune_arms": False}))
        pruned = _measure(_cfg(**{"cancellation.measure_prune_arms": True}))
        for name in self.FIELDS:
            a, b = getattr(full, name), getattr(pruned, name)
            assert np.array_equal(np.asarray(a), np.asarray(b)), (
                "剪枝改变了 %s —— 装配剪枝必须是逐位等价的提速，"
                "否则它是算法改动，得另立判据" % name
            )

    def test_the_needed_arm_is_still_built(self):
        """剪枝后被测臂必须还在（否则上面那条会因 KeyError 而红，但信息更少）。"""
        from isac_sim.receiver import cancellation as cx
        from isac_sim.sensing.model import build_base_gains, generate_geometry

        cfg = _cfg(**{"cancellation.measure_prune_arms": True})
        rng = np.random.default_rng([cfg.run.seed, 7])
        geom = generate_geometry(cfg, rng)
        ctx = cx.ReceiverContext.from_trial(
            cfg, geom, geom, build_base_gains(cfg, geom, rng), arm="tp_uic_full"
        )
        from isac_sim.receiver.cancellation.build import build_observation
        obs = build_observation(
            cfg, ctx.geom_true, ctx.geom_belief, ctx.base, 0,
            rng=np.random.default_rng(11), sense_power=ctx.sense_power,
            radiated_power=ctx.radiated_power,
            processing_gain=ctx.processing_gain, hw_gain=ctx.hw_gain,
            active_mask=ctx.active_mask, base_belief=ctx.base_belief,
            include_echo=True, weak_index=0,
        )
        arms = cx.cancellation_arms(cfg, obs, only="tp_uic_full")
        assert "tp_uic_full" in arms
        # 默认 protected-only 支撑集由保护声明直接决定，生产路径
        # 对 Stage 1 没有数值依赖。
        assert "tp_uic_stage1" not in arms
        # 被剪掉的臂确实没装配
        for gone in ("plain_ls", "ridge_ls", "soft_tpuic", "targeted_tpuic_full"):
            assert gone not in arms
