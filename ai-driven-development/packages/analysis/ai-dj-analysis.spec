# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for AI DJ Analysis sidecar binary.

Produces a single-file executable that Tauri bundles as an external binary.
Build: pyinstaller ai-dj-analysis.spec --clean
"""

import sys
from pathlib import Path

block_cipher = None

# Resolve paths relative to spec file location
spec_dir = Path(SPECPATH)
src_dir = spec_dir / "src"

# Hidden imports that PyInstaller cannot detect via static analysis
hidden_imports = [
    # Core analysis
    "numpy",
    "scipy",
    "scipy.signal",
    "scipy.fft",
    # Audio analysis (optional deps)
    "essentia",
    "essentia.standard",
    "librosa",
    "librosa.core",
    "librosa.feature",
    "pyacoustid",
    # ML inference (optional deps)
    "onnxruntime",
    # JSON-RPC server
    "json",
    "uuid",
    "dataclasses",
    "typing",
    # Application modules
    "src.interface.server",
    "src.interface.handlers",
    "src.application.use_cases.analyze_track",
    "src.application.use_cases.batch_analyze",
    "src.application.use_cases.import_library",
    "src.application.use_cases.search_streaming",
    "src.application.use_cases.analyze_streaming_track",
    "src.infrastructure.analyzers.essentia_analyzer",
    "src.infrastructure.fingerprinters.chromaprint_fingerprinter",
    "src.infrastructure.persistence.sqlite_track_repository",
    "src.infrastructure.parsers.rekordbox_parser",
    "src.infrastructure.parsers.traktor_parser",
    "src.infrastructure.parsers.serato_parser",
    "src.infrastructure.parsers.mixcloud_parser",
    "src.infrastructure.parsers.soundcloud_parser",
    "src.infrastructure.parsers.csv_parser",
]

# Filter out imports that may not be installed
available_imports = []
for imp in hidden_imports:
    try:
        __import__(imp.split(".")[0])
        available_imports.append(imp)
    except ImportError:
        pass

a = Analysis(
    [str(src_dir / "interface" / "server.py")],
    pathex=[str(spec_dir)],
    binaries=[],
    datas=[],
    hiddenimports=available_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch",
        "onnx",
        "onnxscript",
        "matplotlib",
        "tkinter",
        "PIL",
        "cv2",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="ai-dj-analysis",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Sidecar communicates via stdio, needs console
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
