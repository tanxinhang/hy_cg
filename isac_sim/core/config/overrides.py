"""点分路径覆盖机制（实验变体全部走这里，不进配置文件）。"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Dict, Iterator, Tuple
from isac_sim.core.config.run import Config


# --------------------------------------------------------------------------
# 点分路径覆盖机制
# --------------------------------------------------------------------------
def _resolve(cfg: Any, path: str) -> Tuple[Any, str]:
    parts = path.split(".")
    obj: Any = cfg
    for part in parts[:-1]:
        if not is_dataclass(obj) or not hasattr(obj, part):
            raise KeyError(f"no configuration group {part!r} in path {path!r}")
        obj = getattr(obj, part)
    return obj, parts[-1]

def _coerce(current: Any, value: Any) -> Any:
    """Cast ``value`` to the type of the existing field."""
    if isinstance(current, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if current is None:  # e.g. sensing_processing_gain
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    return value

def apply_overrides(cfg: Config, overrides: Dict[str, Any]) -> Config:
    """返回 ``cfg`` 的深拷贝，并施加点分路径的 ``overrides``。

    例
    --
    >>> apply_overrides(cfg, {"selector.lambda_c": 0.02, "dd.use_otfs_bin_validity": False})
    """
    import copy as _copy

    out = _copy.deepcopy(cfg)
    for path, value in overrides.items():
        owner, name = _resolve(out, path)
        if not hasattr(owner, name):
            raise KeyError(f"unknown configuration field {path!r}")
        setattr(owner, name, _coerce(getattr(owner, name), value))
    return out

def iter_leaf_paths(cfg: Config) -> Iterator[Tuple[str, Any]]:
    """Yield ``(dotted_path, value)`` for every scalar leaf, for logging/diffing."""

    def walk(obj: Any, prefix: str) -> Iterator[Tuple[str, Any]]:
        for f in fields(obj):
            value = getattr(obj, f.name)
            path = f"{prefix}{f.name}"
            if is_dataclass(value):
                yield from walk(value, f"{path}.")
            else:
                yield path, value

    yield from walk(cfg, "")
