"""Tool registry with a middleware chain (M5).

A tool is environment-side code the agent can call by name with JSON arguments. It may read the
photo's sanitized pixels and the train index, never the photo's labels or metadata (hard rule 5).
Every call goes through the middleware chain (outermost first), e.g. trace -> fault -> cache ->
placebo -> tool, so tracing, fault injection, caching and placebo mode work for every tool alike.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class Photo:
    id: str
    image_path: str          # sanitized JPEG; the only view of the photo the agent ever gets
    group_id: str = ""       # env-private: lets similar() skip the photo's own capture sequence


@dataclass
class Obs:
    text: str
    image: bytes | None = None   # JPEG bytes (view)


@dataclass
class Ctx:
    photo: Photo
    rng: np.random.Generator
    trace: list[dict] = field(default_factory=list)


@dataclass
class Tool:
    name: str
    description: str                        # shown to the model
    args: dict[str, dict]                   # JSON-schema properties; all required
    fn: Callable[..., Obs]                  # fn(ctx, **args)
    placebo: Callable[..., Obs] | None = None   # same signature; plausible but uninformative
    per_photo: bool = True                  # result depends on the photo (cache key includes it)


class Middleware(Protocol):
    def __call__(self, ctx: Ctx, tool: Tool, args: dict, nxt: Callable[[Ctx, Tool, dict], Obs]) -> Obs: ...


class Registry:
    def __init__(self, tools: list[Tool], middleware: list[Middleware] | None = None):
        self.tools = {t.name: t for t in tools}
        self.middleware = middleware or []

    def call(self, ctx: Ctx, name: str, args: dict[str, Any]) -> Obs:
        tool = self.tools[name]

        def base(c: Ctx, t: Tool, a: dict) -> Obs:
            return t.fn(c, **a)

        call = base
        for mw in reversed(self.middleware):
            call = (lambda m, n: (lambda c, t, a: m(c, t, a, n)))(mw, call)
        return call(ctx, tool, args)
