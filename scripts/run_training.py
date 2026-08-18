import argparse
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=False)
    parser.add_argument(
        "--list", action="store_true", help="List available experiments and exit"
    )
    args = parser.parse_args()

    if args.list:
        experiments_dir = REPO_ROOT / "experiments"
        expt_names = sorted(
            p.name
            for p in experiments_dir.iterdir()
            if p.is_dir() and not p.name.startswith("_")
        )
        print("Available experiments:")
        for name in expt_names:
            print(f"  {name}")
        return

    if not args.tag:
        parser.error("--tag is required unless --list is passed")

    script = REPO_ROOT / "experiments" / args.tag / "run.py"
    expt_config = REPO_ROOT / "experiments" / args.tag / "config.yaml"

    if not script.exists():
        raise FileNotFoundError(f"Experiment script not found: {script}")

    print(f"Running experiment: {args.tag}")
    print(f"Script: {script}")
    print(f"Config: {expt_config}")

    subprocess.run(
        [sys.executable, str(script)],
        cwd=script.parent,
        check=True,
    )


if __name__ == "__main__":
    main()
