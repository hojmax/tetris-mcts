from __future__ import annotations

from importlib import machinery, util
from pathlib import Path
from pkgutil import extend_path
from types import ModuleType
import sys

__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]


def _load_native_module() -> ModuleType:
    try:
        from . import tetris_core as native
    except ModuleNotFoundError as exc:
        missing_name = exc.name or ""
        if missing_name not in {f"{__name__}.tetris_core", "tetris_core.tetris_core"}:
            raise
    else:
        return native

    # Some environments install the extension as a top-level shared object.
    for raw_path in sys.path:
        search_root = Path(raw_path or ".").resolve()
        for suffix in machinery.EXTENSION_SUFFIXES:
            candidate = search_root / f"tetris_core{suffix}"
            if not candidate.is_file():
                continue
            spec = util.spec_from_file_location(f"{__name__}.tetris_core", candidate)
            if spec is None or spec.loader is None:
                continue
            module = util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module

    raise ModuleNotFoundError(
        "Could not locate native tetris_core extension. "
        "Rebuild it with `make build-dev` or `make build`."
    )


_native = _load_native_module()
_exported = getattr(_native, "__all__", None)
if _exported is None:
    _exported = [name for name in dir(_native) if not name.startswith("_")]

globals().update({name: getattr(_native, name) for name in _exported})
__all__ = list(_exported)
__doc__ = _native.__doc__
