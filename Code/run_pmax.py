import subprocess
import sys

p_list = ["0.20", "0.35", "0.50"]

for p in p_list:
    out_dir = f"sweep_pmax_{p}"
    cmd = [
        sys.executable,
        "gate_otfs_collision_experiments_v18.py",
        "--gate", "tracking",
        "--tracking-mc", "100",
        "--num-frames", "30",
        "--inner-mc", "50",
        "--tracking-M", "15",
        "--tracking-Q", "10",
        "--eval-kernel-mode", "dirichlet",
        "--target-pd-fusion", "or",
        "--tracking-reinit-lost",
        "--tracker-mode", "dd_jpda_shared",
        "--meas-pollution", "biased_peak",
        "--pollution-pmax", p,
        "--assignment-hysteresis",
        "--hys-abs-margin", "0.05",
        "--kf-innovation-gate",
        "--kf-gate-action", "skip",
        "--kf-gate-chi2", "13.3",
        "--kf-abs-pos-gate-m", "500",
        "--kf-gate-pos-std-cap-m", "500",
        "--kf-reject-pos-std-m", "1200",
        "--include-global-assignment-baselines",
        "--save-diagnostics",
        "--num-workers", "10",
        "--fresh-out-dir",
        "--out-dir", out_dir,
    ]

    print("\nRunning:", " ".join(cmd))
    subprocess.run(cmd, check=True)
