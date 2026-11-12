"""M5 harness: env budgets/schema, middleware chain, agent loop with a scripted fake model."""
import json

import numpy as np
from PIL import Image

from retrace.env import Env
from retrace.harness.agent import run_episode
from retrace.middleware import Cache, Fault, Placebo, Trace
from retrace.tools.registry import Ctx, Obs, Photo, Registry, Tool


def echo_tool(calls):
    def fn(ctx, x):
        calls.append(x)
        return Obs(f"real {x}")
    return Tool("echo", "echo", {"x": {"type": "integer"}}, fn, lambda ctx, x: Obs(f"placebo {x}"))


def ctx():
    return Ctx(Photo("p1", "img.jpg"), np.random.default_rng(0))


def test_middleware_cache_placebo_trace():
    calls = []
    reg = Registry([echo_tool(calls)], [Trace(), Fault(0), Cache()])
    c = ctx()
    assert reg.call(c, "echo", {"x": 1}).text == "real 1"
    reg.call(c, "echo", {"x": 1})
    assert calls == [1] and len(c.trace) == 2
    preg = Registry([echo_tool(calls)], [Trace(), Placebo({"echo"})])
    assert preg.call(ctx(), "echo", {"x": 2}).text == "placebo 2"


def test_fault_and_crash_become_observations():
    reg = Registry([echo_tool([])], [Trace(), Fault(1.0)])
    assert "unavailable" in reg.call(ctx(), "echo", {"x": 1}).text
    bad = Tool("bad", "", {}, lambda ctx: 1 / 0)
    c = ctx()
    assert "failed" in Registry([bad], [Trace()]).call(c, "bad", {}).text and "exception" in c.trace[0]


class FakeModel:
    """Calls echo until told answer is the only choice."""
    def complete(self, messages, schema, max_tokens):
        acts = schema["properties"]["action"]
        only_answer = "anyOf" not in acts
        a = ({"tool": "answer", "args": {"country": "NL", "lat": 52.0, "lon": 5.0, "radius_km": 50,
                                         "utc_datetime": "2018-10-01T12:00:00Z", "time_uncertainty_hours": 24}}
             if only_answer else {"tool": "echo", "args": {"x": len(messages)}})
        return json.dumps({"thought": "t", "action": a}), {"prompt_tokens": 100, "completion_tokens": 10}, 0.01


def test_episode_respects_step_budget(tmp_path):
    Image.new("RGB", (64, 64)).save(tmp_path / "img.jpg")
    env = Env(Registry([echo_tool([])], [Trace()]), max_steps=4)
    orig = env.reset
    env.reset = lambda photo, seed=0: orig(photo, seed, data_root=tmp_path)
    out = run_episode(FakeModel(), env, Photo("p1", "img.jpg"))
    a, tr = out["answer"], out["trajectory"]
    assert a["n_steps"] == 4 and a["tools_used"] == ["echo"] * 3 and a["lat"] == 52.0
    assert tr["steps"][-1]["allowed"] == ["answer"] and tr["steps"][0]["obs"].startswith("real")
