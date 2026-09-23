#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thin entry point for the refactored simulator.

All logic lives in the :mod:`isac_sim` (building blocks) and
    :mod:`experiments` (flow) packages.  Run ``--help`` for the full
(and deliberately small) option set::

    python run_isac_sim.py --help
    python run_isac_sim.py --mc 200 --out runs/paper
    python run_isac_sim.py --mode ablation --mc 100

The legacy ``lagrangian_dotfs_isac_simplified_v10_paper_plots.py`` remains in
the repository only as the frozen reference for the numbers already reported;
new work should use this entry point.
"""

from experiments.app.cli import main

if __name__ == "__main__":
    main()
