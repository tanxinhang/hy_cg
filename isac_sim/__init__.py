"""Simplified Lagrangian-guided DOTFS-ISAC cooperative-sensing simulator.

The package replaces the former single-file prototype
(``lagrangian_dotfs_isac_simplified_v10_paper_plots.py``) with a small set of
single-responsibility modules:

    config       configuration groups and dotted-path overrides
    model        geometry, channel gains and per-link quantity tables
    dd           DD-domain leakage kernels and the C2F (coarse-to-fine)
                 eta^c / eta^loc / eta^f gain model
    waveform     physical OTFS PSF kernel for end-to-end validation
    prior        target-state prior-uncertainty utilities
    fusion       soft-information fusion, deflection and target priority
    selection    the proposed Lagrangian rule, C2F and the baselines
    simulate     Monte-Carlo execution and metric aggregation
    experiments  sweep / ablation definitions (data, not flags)
    report       console summary, CSV and LaTeX tables
    plotting     figures
    cli          the unified command-line interface
    naming       display order and human-readable labels

Typical use::

    from isac_sim import Config, run_simulation
    summary = run_simulation(Config())

or from the shell::

    python -m isac_sim --mode ablation --mc 100 --out runs/
    python -m isac_sim --mode c2f --mc 200
    python -m isac_sim --mode prior-sweep --mc 50
    python -m isac_sim --mode waveform-check
"""

from __future__ import annotations

from .config import Config, apply_overrides, default_config, iter_leaf_paths
from .experiments import ABLATION_VARIANTS, DD_VARIANTS, EXPERIMENTS
from .model import BaseGains, LinkTables, build_base_gains, compute_link_tables, generate_geometry
from .oracle import greedy_objective, oracle_exhaustive
from .prior import perturbed_geometry, predicted_geometry
from .selection import C2F_METHODS, METHODS
from .simulate import run_one_trial, run_simulation, summarize
from .waveform import psf_local_capture, psf_main_bin, sweep_compare_analytic_vs_psf

__all__ = [
    "Config",
    "default_config",
    "apply_overrides",
    "iter_leaf_paths",
    "run_simulation",
    "run_one_trial",
    "summarize",
    "generate_geometry",
    "build_base_gains",
    "compute_link_tables",
    "BaseGains",
    "LinkTables",
    "METHODS",
    "C2F_METHODS",
    "EXPERIMENTS",
    "ABLATION_VARIANTS",
    "DD_VARIANTS",
    "oracle_exhaustive",
    "greedy_objective",
    "perturbed_geometry",
    "predicted_geometry",
    "psf_main_bin",
    "psf_local_capture",
    "sweep_compare_analytic_vs_psf",
]

__version__ = "1.2.0"
