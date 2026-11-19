"""Utilitas bersama: device, checkpoint (untuk resume Colab), simpan JSON, timer."""
from __future__ import annotations

import json
import time
from pathlib import Path


def get_device(pref: str = "auto"):
    import torch
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def save_checkpoint(path: Path, **state) -> None:
    """Simpan atomik (tulis ke .tmp lalu rename) supaya file tidak korup
    jika sesi Colab putus di tengah penyimpanan."""
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, map_location="cpu"):
    import torch
    path = Path(path)
    if not path.exists():
        return None
    return torch.load(path, map_location=map_location, weights_only=False)


def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def env_info() -> dict:
    """Versi library yang dipakai, dicatat di output agar laporan bisa diverifikasi."""
    import platform
    info = {"python": platform.python_version()}
    for mod in ("numpy", "pandas", "sklearn", "scipy", "torch", "torchvision", "umap", "matplotlib"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except ImportError:
            info[mod] = None
    try:
        import torch
        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    return info


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.time() - self.t0
