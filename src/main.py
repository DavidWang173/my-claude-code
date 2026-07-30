"""Executable module and installed console-script entrypoint."""

from __future__ import annotations

from .cli import build_parser, main

__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
