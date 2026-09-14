# Figure 2 release QA

- Core conclusion: target-balanced C2F selection preserves most of the
  all-neighbor detection rate with far lower reporting cost, while global
  ranking starves targets.
- Archetype: quantitative grid with panel (a) as the hero evidence.
- Backend: Python/matplotlib only.
- Final size: 7.2 in double-column width; PDF is the manuscript asset.
- Panel map: (a) detection and 95% CI; (b) serial reporting delay on a log
  scale; (c) fraction of targets receiving at least one selected report.
- Statistics: 1000 paired Monte-Carlo geometries; Wilson intervals are shown
  for detection probability. Paired method differences are in `main.csv`.
- Source data: `main/main/main.csv`; resolved settings are in the adjacent
  `config.json`.
- Exports: editable-text PDF and SVG plus a 600-dpi PNG preview.
- Reviewer risk: all-neighbor is an upper-resource reference, not a detection
  upper bound; cost-aware and exact-marginal ranking coincide under fixed
  blocklength for the same candidate utilities.
