#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility entrypoint for the refactored OTFS DD-collision experiments.

The implementation now lives in the ``gate_otfs_collision`` package. Existing
commands that execute this file directly still work.
"""

from gate_otfs_collision.cli import main


if __name__ == "__main__":
    main()
