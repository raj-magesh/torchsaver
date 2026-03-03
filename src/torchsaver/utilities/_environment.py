import os
from pathlib import Path

from xdg_base_dirs import xdg_cache_home

TORCHSAVER_HOME = Path(
    os.getenv(
        "TORCHSAVER_HOME",
        str(xdg_cache_home() / "torchsaver"),
    ),
)
