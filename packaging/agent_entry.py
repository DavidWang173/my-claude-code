"""PyInstaller entrypoint for the standalone command-line executable."""

from src.main import main


if __name__ == "__main__":
    raise SystemExit(main())
