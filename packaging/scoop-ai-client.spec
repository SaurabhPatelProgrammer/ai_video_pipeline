from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

project = Path(SPECPATH).resolve().parent
datas = collect_data_files("scoop_ai")
binaries = []
hiddenimports = [
    "keyring.backends.Windows",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtNetwork",
]
datas += [
    (
        str(project / "models" / "ice-cream-item-rfdetr-nano-v2" / "model-manifest.json"),
        "models/ice-cream-item-rfdetr-nano-v2",
    ),
    (
        str(project / "models" / "ice-cream-item-rfdetr-nano-v2" / "checkpoint_best_total.pth"),
        "models/ice-cream-item-rfdetr-nano-v2",
    ),
]

a = Analysis(
    [str(project / "src" / "scoop_ai" / "client.py")],
    pathex=[str(project / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "pytest",
        "ruff",
        # PySide6 now ships: the client's main window is the desktop shell. Only
        # the Qt modules it never touches are pruned, to keep the bundle small.
        "PySide6.Qt3DAnimation",
        "PySide6.Qt3DCore",
        "PySide6.Qt3DExtras",
        "PySide6.Qt3DInput",
        "PySide6.Qt3DLogic",
        "PySide6.Qt3DRender",
        "PySide6.QtCharts",
        "PySide6.QtDataVisualization",
        "PySide6.QtMultimedia",
        "PySide6.QtQuick3D",
        "PySide6.QtSql",
        "PySide6.QtTest",
        "pytorch_lightning",
        "lightning",
        "wandb",
        "tensorboard",
        "mlflow",
        "clearml",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True, name="ScoopAIClient",
    debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
    console=False, disable_windowed_traceback=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="ScoopAIClient")
