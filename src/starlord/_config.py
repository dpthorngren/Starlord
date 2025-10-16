import os
import platform
from pathlib import Path
from types import SimpleNamespace

__version__ = "0.1.2"

config = SimpleNamespace(
    system=platform.system(),
    base_dir=Path.home() / ".starlord",
    cache_dir=Path.home() / ".starlord" / "cycache",
    grid_dir=Path.home() / ".starlord" / "grids",
)


def _load_config():
    config.system = platform.system()
    if "STARLORD_DATA_DIR" in os.environ.keys():
        config.base_dir = Path(os.environ['STARLORD_DATA_DIR'])
        assert config.base_dir.parent.exists()
        config.base_dir.mkdir(exist_ok=True)
    elif config.system == "Linux":
        assert Path.home().exists()
        config.base_dir = Path.home() / ".config" / "starlord"
        config.base_dir.mkdir(parents=True, exist_ok=True)
    elif config.system == "Darwin":
        assert Path.home().exists()
        config.base_dir = Path.home() / "Library" / "Application Support" / "starlord"
        config.base_dir.mkdir(parents=True, exist_ok=True)
    else:
        assert Path.home().exists()
        config.base_dir = Path.home() / ".starlord"
        config.base_dir.mkdir(parents=True, exist_ok=True)
    config.cache_dir = config.base_dir / "cycache"
    config.cache_dir.mkdir(exist_ok=True)
    config.grid_dir = config.base_dir / "grids"
    config.grid_dir.mkdir(exist_ok=True)


_load_config()
