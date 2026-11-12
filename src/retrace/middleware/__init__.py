"""Tool-call middleware (M5): tracing, fault injection, caching, placebo mode.

Order used by the harness (outermost first): Trace, Fault, Cache, Placebo.
"""
from __future__ import annotations

import json
import time

from retrace.tools.registry import Ctx, Obs, Tool


class Trace:
    """Records every tool call (args, output text, latency, flags) into ctx.trace."""

    def __call__(self, ctx: Ctx, tool: Tool, args: dict, nxt) -> Obs:
        t = time.time()
        rec = {"tool": tool.name, "args": args}
        try:
            obs = nxt(ctx, tool, args)
        except Exception as e:  # noqa: BLE001 - a crashing tool becomes an error observation
            obs = Obs(f"error: {tool.name} failed ({type(e).__name__})")
            rec["exception"] = f"{type(e).__name__}: {str(e)[:200]}"
        rec.update(text=obs.text, image=obs.image is not None, latency_s=round(time.time() - t, 4))
        ctx.trace.append(rec)
        return obs


class Fault:
    """With probability `rate`, the tool 'fails'. For robustness tests; rate 0 = off."""

    def __init__(self, rate: float = 0.0):
        self.rate = rate

    def __call__(self, ctx: Ctx, tool: Tool, args: dict, nxt) -> Obs:
        if self.rate and ctx.rng.random() < self.rate:
            return Obs(f"error: {tool.name} is temporarily unavailable")
        return nxt(ctx, tool, args)


class Cache:
    """Memoizes identical calls (same tool, args and, for per-photo tools, photo)."""

    def __init__(self):
        self.store: dict[tuple, Obs] = {}
        self.hits = 0

    def __call__(self, ctx: Ctx, tool: Tool, args: dict, nxt) -> Obs:
        key = (tool.name, ctx.photo.id if tool.per_photo else "", json.dumps(args, sort_keys=True))
        if key in self.store:
            self.hits += 1
            return self.store[key]
        obs = self.store[key] = nxt(ctx, tool, args)
        return obs


class Placebo:
    """Replaces the named tools' outputs with plausible but uninformative ones (spec baseline #4):
    measures whether the agent uses what tools say or just benefits from extra turns."""

    def __init__(self, names: set[str]):
        self.names = names

    def __call__(self, ctx: Ctx, tool: Tool, args: dict, nxt) -> Obs:
        if tool.name in self.names and tool.placebo is not None:
            return tool.placebo(ctx, **args)
        return nxt(ctx, tool, args)
