"""Create any BRT environment by name: ``biorobot.make("brittle_star/light_escape", backend="mjx")``."""

from __future__ import annotations

import importlib
from typing import Callable, Dict, List

from biorobot.core.env import BRTEnv

_REGISTRY: Dict[str, Callable[..., BRTEnv]] = {}
_ANIMAL_PACKAGES = ("brittle_star", "jumping_spider")


def register(name: str) -> Callable[[Callable[..., BRTEnv]], Callable[..., BRTEnv]]:
    """Decorator registering ``factory(backend="mjc", **kwargs) -> BRTEnv`` under ``"<animal>/<task>"``."""

    def decorator(factory: Callable[..., BRTEnv]) -> Callable[..., BRTEnv]:
        _REGISTRY[name] = factory
        return factory

    return decorator


def _load_all() -> None:
    for package in _ANIMAL_PACKAGES:
        importlib.import_module(f"biorobot.{package}")


def available() -> List[str]:
    """Names of all registered environments."""
    _load_all()
    return sorted(_REGISTRY)


def make(name: str, backend: str = "mjc", **kwargs) -> BRTEnv:
    """Build a registered environment with its default morphology and arena.

    Keyword arguments are forwarded to the environment's factory; every factory accepts at least ``config``
    (an ``EnvConfig`` subclass instance) and the fields of its config class as overrides (e.g. ``render_mode``).
    """
    _load_all()
    if name not in _REGISTRY:
        raise KeyError(f"Unknown environment '{name}'. Available: {available()}")
    return _REGISTRY[name](backend=backend, **kwargs)
