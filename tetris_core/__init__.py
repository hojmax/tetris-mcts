from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)  # merge source tree + site-packages  # type: ignore[name-defined]
from .tetris_core import *  # noqa: F401, F403
