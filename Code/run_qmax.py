import subprocess
import sys
from pathlib import Path


def run_one(q: int) -> None:
    script_dir = Path(__file__).resolve().parent
    exp_script = script_dir / "gate_otfs_collision_experiments_v18.py"

    cmd = [
        sys.executable,
        str(exp_script),

        "--gate", "tracking",
        "--tracking-mc", "100",
        "--num-frames", "30",
        "--inner-mc", "50",
        "--tracking-M", "15",
        "--tracking-Q", str(q),

        "--eval-kernel-mode", "dirichlet",
        "--target-pd-fusion", "or",
        "--tracking-reinit-lost",
        "--tracker-mode", "dd_jpda_shared",
        "--meas-pollution", "biased_peak",
        "--pollution-pmax", "0.35",

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
        "--out-dir", f"sweep_Q_{q}",
    ]

    print("=" * 80)
    print(f"Running Q={q}")
    print(" ".join(cmd))
    print("=" * 80)

    subprocess.run(cmd, cwd=script_dir, check=True)


def main() -> None:
    q_list = [6, 8, 10, 12]

    for q in q_list:
        run_one(q)

    print("\nAll Q sweeps finished successfully.")


if __name__ == "__main__":
    main()
