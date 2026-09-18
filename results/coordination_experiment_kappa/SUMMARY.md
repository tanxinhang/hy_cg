# coordination x direct-path cancellation depth (kappa_dc)

`tools/probe_coordination_kappa.py --mc 200 --seeds 4 --penalty 0.1 --looks 16 --rounds 12`

working point: 500 m / RCS 0.2 m^2 / L=16 | 800 trials per arm per cell

| kappa_dc (dB) | arm | worst-P_D MIN | mean-of-worst | std | mean P_D | #TX | #links | P_FA |
|---|---|---|---|---|---|---|---|---|
| 40 | uncoordinated | 0.7800 | 0.7988 | 0.0114 | 0.8395 | 15.0 | 23.8 | 0.0481-0.0490 |
| 40 | mask_only | 0.8150 | 0.8425 | 0.0175 | 0.8816 | 9.9 | 23.8 | 0.0481-0.0491 |
| 40 | sparse | 0.8950 | 0.9075 | 0.0083 | 0.9326 | 3.4 | 17.0 | 0.0482-0.0504 |
| 50 | uncoordinated | 0.9400 | 0.9437 | 0.0041 | 0.9614 | 15.0 | 13.6 | 0.0484-0.0493 |
| 50 | mask_only | 0.9550 | 0.9662 | 0.0089 | 0.9790 | 5.6 | 13.6 | 0.0485-0.0493 |
| 50 | sparse | 0.9500 | 0.9525 | 0.0025 | 0.9709 | 2.4 | 12.5 | 0.0480-0.0496 |
| 60 | uncoordinated | 0.9500 | 0.9600 | 0.0061 | 0.9770 | 15.0 | 11.6 | 0.0479-0.0493 |
| 60 | mask_only | 0.9550 | 0.9637 | 0.0054 | 0.9794 | 4.3 | 11.6 | 0.0479-0.0493 |
| 60 | sparse | 0.9500 | 0.9575 | 0.0056 | 0.9728 | 2.1 | 11.9 | 0.0481-0.0491 |
| 80 | uncoordinated | 0.9500 | 0.9538 | 0.0041 | 0.9754 | 15.0 | 11.3 | 0.0477-0.0492 |
| 80 | mask_only | 0.9500 | 0.9538 | 0.0041 | 0.9754 | 4.1 | 11.3 | 0.0477-0.0492 |
| 80 | sparse | 0.9500 | 0.9575 | 0.0083 | 0.9735 | 2.1 | 11.8 | 0.0473-0.0485 |

## median sensing rinr, uncoordinated arm (dB; >0 = interference limited)

| kappa_dc | uncoordinated | mask_only | sparse |
|---|---|---|---|
| 40 | +10.87 | +8.67 | +2.40 |
| 50 | +1.64 | -2.46 | -4.89 |
| 60 | -4.19 | -5.46 | -5.73 |
| 80 | -5.82 | -5.83 | -5.84 |

## coordination gain vs kappa (sparse - uncoordinated)

Both calibres are listed because the runner prints them in different places (see ROUTE 13.6 / ROOT_CAUSE 3.6); the verdict uses the MIN calibre only.

| kappa_dc | gain (worst-P_D **MIN**) | gain (mean-of-worst) | uncoordinated pass |
|---|---|---|---|
| 40 | **+0.1150** | +0.1087 | 0/4 |
| 50 | **+0.0100** | +0.0087 | 1/4 |
| 60 | **+0.0000** | -0.0025 | 4/4 |
| 80 | **+0.0000** | +0.0037 | 4/4 |

## pre-registered verdict

- coordination gain (sparse - uncoordinated) at kappa=40: **+0.1150**; at kappa=80: **+0.0000**
- kappa 40->80 buys: uncoordinated +0.1700, sparse +0.0550
- interaction (sub-additivity) = **+0.1150**
- uncoordinated noise-limited from kappa: [60.0, 80.0]
- **VERDICT: SUBSTITUTES**
- convergence: kappa=40:800/800, kappa=50:800/800, kappa=60:800/800, kappa=80:800/800
