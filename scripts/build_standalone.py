"""Build and smoke-test the standalone command-line executable."""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIST_DIR = PROJECT_ROOT / "dist"
DEFAULT_WORK_DIR = PROJECT_ROOT / "build" / "pyinstaller"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a single-file coding-agent executable with PyInstaller."
    )
    parser.add_argument(
        "--dist-dir",
        type=Path,
        default=DEFAULT_DIST_DIR,
        help="directory for the final executable",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
        help="directory for temporary PyInstaller build files",
    )
    parser.add_argument(
        "--skip-smoke-test",
        action="store_true",
        help="build without running the generated executable's --help command",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if importlib.util.find_spec("PyInstaller") is None:
        print(
            "PyInstaller is not installed. Install the standalone build dependencies "
            'with: python -m pip install -e ".[standalone]"',
            file=sys.stderr,
        )
        return 2

    dist_dir = args.dist_dir.expanduser().resolve()
    work_dir = args.work_dir.expanduser().resolve()
    spec_file = PROJECT_ROOT / "agent.spec"
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(work_dir),
        str(spec_file),
    ]
    build_environment = os.environ.copy()
    build_environment["PYINSTALLER_CONFIG_DIR"] = str(work_dir / "config")
    try:
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
            env=build_environment,
        )
    except subprocess.CalledProcessError as exc:
        return exc.returncode or 1

    executable = dist_dir / ("agent.exe" if sys.platform == "win32" else "agent")
    if not executable.is_file():
        print(f"Build did not create the expected executable: {executable}", file=sys.stderr)
        return 1

    if not args.skip_smoke_test:
        completed = subprocess.run(
            [str(executable), "--help"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0 or "usage: agent" not in completed.stdout:
            print("Standalone executable smoke test failed.", file=sys.stderr)
            if completed.stdout:
                print(completed.stdout, file=sys.stderr)
            if completed.stderr:
                print(completed.stderr, file=sys.stderr)
            return completed.returncode or 1

    size_mib = executable.stat().st_size / (1024 * 1024)
    print(f"Standalone executable ready: {executable} ({size_mib:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
