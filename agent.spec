# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller specification for the single-file console executable."""

from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules


PROJECT_ROOT = Path(SPECPATH)
BUNDLED_SOURCES = [
    (
        str(source_file),
        str(source_file.parent.relative_to(PROJECT_ROOT)),
    )
    for source_file in sorted((PROJECT_ROOT / "src").rglob("*.py"))
]

analysis = Analysis(
    [str(PROJECT_ROOT / "packaging" / "agent_entry.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=BUNDLED_SOURCES,
    hiddenimports=collect_submodules("src"),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

python_archive = PYZ(analysis.pure)

executable = EXE(
    python_archive,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    [],
    name="agent",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
