import copy
import re
from pathlib import Path

import yaml


class ConfigError(ValueError):
    """Raised when a config is invalid."""


class DotDict(dict):
    def __getattr__(self, item):
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        if isinstance(value, dict) and not isinstance(value, DotDict):
            value = DotDict(value)
            self[item] = value
        return value

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def _merge_dicts(base, update):
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dicts(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ConfigError(f"config at {path} must deserialize to a mapping.")

    if "inherits" in cfg:
        parent_path = (path.parent / cfg["inherits"]).resolve()
        parent = load_config(parent_path)
        cfg = _merge_dicts(
            dict(parent), {k: v for k, v in cfg.items() if k != "inherits"}
        )

    return DotDict(cfg)


def _override_parts(path):
    """Parse dotted paths and list indices such as policies[1].checkpoint_dir."""
    normalized = re.sub(r"\[(\d+)\]", r".\1", str(path).strip())
    parts = normalized.split(".")
    if not normalized or any(not part for part in parts):
        raise ConfigError(f"invalid override path: {path!r}")
    return parts


def _list_index(part, path):
    if not part.isdigit():
        raise ConfigError(
            f"override path {path!r} must use an integer list index, got {part!r}"
        )
    return int(part)


def apply_overrides(cfg, overrides):
    updated = copy.deepcopy(dict(cfg))
    for key, value in overrides.items():
        parts = _override_parts(key)
        cursor = updated
        for position, part in enumerate(parts[:-1]):
            next_part = parts[position + 1]
            if isinstance(cursor, list):
                index = _list_index(part, key)
                if index >= len(cursor):
                    raise ConfigError(
                        f"override path {key!r} selects list index {index}, "
                        f"but the list has length {len(cursor)}"
                    )
                child = cursor[index]
                if not isinstance(child, (dict, list)):
                    raise ConfigError(
                        f"override path {key!r} cannot descend through "
                        f"{type(child).__name__} at index {index}"
                    )
                cursor = child
                continue

            if not isinstance(cursor, dict):
                raise ConfigError(
                    f"override path {key!r} cannot descend through {type(cursor).__name__}"
                )
            if part not in cursor:
                cursor[part] = [] if next_part.isdigit() else {}
            child = cursor[part]
            if not isinstance(child, (dict, list)):
                raise ConfigError(
                    f"override path {key!r} cannot descend through "
                    f"{type(child).__name__} at {part!r}"
                )
            cursor = child

        final = parts[-1]
        if isinstance(cursor, list):
            index = _list_index(final, key)
            if index >= len(cursor):
                raise ConfigError(
                    f"override path {key!r} selects list index {index}, "
                    f"but the list has length {len(cursor)}"
                )
            cursor[index] = value
        elif isinstance(cursor, dict):
            cursor[final] = value
        else:
            raise ConfigError(
                f"override path {key!r} cannot assign through {type(cursor).__name__}"
            )
    return DotDict(updated)


def _to_builtin(value):
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_builtin(v) for v in value)
    return value


def save_config(cfg, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(_to_builtin(cfg), f, sort_keys=False)
