"""Model adapter + agent loop (M5).

ChatModel talks to any OpenAI-compatible server (vLLM locally; an API model later). run_episode
drives one photo: model turn (schema-forced JSON action) -> env.step -> observation -> ...
"""
from __future__ import annotations

import json
import time

import requests

from retrace.env import Env
from retrace.tools.registry import Photo


class ChatModel:
    def __init__(self, url: str = "http://127.0.0.1:8000/v1", model: str | None = None,
                 temperature: float = 0.0, seed: int = 0):
        self.url, self.temperature, self.seed = url, temperature, seed
        self.model = model or requests.get(f"{url}/models", timeout=30).json()["data"][0]["id"]

    def complete(self, messages: list[dict], schema: dict, max_tokens: int) -> tuple[str, dict, float]:
        body = {"model": self.model, "messages": messages, "temperature": self.temperature,
                "seed": self.seed, "max_tokens": max_tokens,
                "response_format": {"type": "json_schema", "json_schema": {"name": "action", "schema": schema}}}
        t = time.time()
        r = requests.post(f"{self.url}/chat/completions", json=body, timeout=600)
        r.raise_for_status()
        j = r.json()
        return j["choices"][0]["message"]["content"], j["usage"], time.time() - t


def run_episode(model: ChatModel, env: Env, photo: Photo, seed: int = 0, max_tokens: int = 400) -> dict:
    """Returns {"answer": answer row for scoring, "trajectory": loggable record (no image bytes)}."""
    msgs = env.reset(photo, seed)
    steps, prompt_tokens, t0, err, text = [], 0, time.time(), None, ""
    tin = tout = 0
    while True:
        allowed = env.allowed(prompt_tokens)
        text = ""
        try:
            text, usage, lat = model.complete(msgs, env.schema(allowed), max_tokens)
            out = json.loads(text)
        except Exception as e:  # noqa: BLE001 - recorded; scored as no answer
            err = f"{type(e).__name__}: {str(e)[:200]}" + (f" | output tail: {text[-200:]!r}" if text else "")
            break
        prompt_tokens = usage["prompt_tokens"]
        tin, tout = tin + usage["prompt_tokens"], tout + usage["completion_tokens"]
        msgs.append({"role": "assistant", "content": text})
        n_trace = len(env.ctx.trace)
        obs, done = env.step(out["action"])
        rec = {"thought": out["thought"], "action": out["action"], "allowed": allowed,
               "prompt_tokens": usage["prompt_tokens"], "completion_tokens": usage["completion_tokens"],
               "model_latency_s": round(lat, 3)}
        if len(env.ctx.trace) > n_trace:
            tr = env.ctx.trace[-1]
            rec.update(obs=tr["text"], tool_latency_s=tr["latency_s"], **({"exception": tr["exception"]}
                                                                          if "exception" in tr else {}))
        steps.append(rec)
        if done:
            break
        msgs.append(obs)

    ans = {"id": photo.id, "tokens_in": tin, "tokens_out": tout, "latency_s": round(time.time() - t0, 3),
           "n_steps": len(steps), "tools_used": [s["action"]["tool"] for s in steps if s["action"]["tool"] != "answer"]}
    if env.answer is not None:
        a = env.answer
        ans.update(lat=a["lat"], lon=a["lon"], radius_km=a["radius_km"], utc_datetime=a["utc_datetime"],
                   time_uncertainty_hours=a["time_uncertainty_hours"], country_guess=a["country"])
    else:
        ans["error"] = err or "no answer"
    return {"answer": ans, "trajectory": {"id": photo.id, "steps": steps, **({"error": ans["error"]}
                                                                           if "error" in ans else {})}}
