"""Retrace environment (M5): reset(photo) -> first message; step(action) -> observation.

The env owns the photo, the tool registry and the budgets. It never holds or reveals labels:
scoring happens afterwards, outside the episode (hard rule 5).
Actions are schema-forced JSON {"thought": str, "action": {"tool": name, "args": {...}}};
the schema for each turn only lists the actions the budgets still allow.
"""
from __future__ import annotations

import base64

import numpy as np

from retrace.tools.registry import Ctx, Photo, Registry

ANSWER_ARGS = {
    "country": {"type": "string", "maxLength": 60},
    "lat": {"type": "number", "minimum": -90, "maximum": 90},
    "lon": {"type": "number", "minimum": -180, "maximum": 180},
    "radius_km": {"type": "number", "minimum": 0.1, "maximum": 20000},
    "utc_datetime": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?Z$"},
    "time_uncertainty_hours": {"type": "number", "minimum": 0, "maximum": 100000},
}
ANSWER_DESC = ("final answer, ends the episode. lat/lon: best single guess; radius_km: radius within which "
               "the true spot lies with about 40% probability; utc_datetime: capture time in UTC "
               "(e.g. 2019-07-14T09:30:00Z); time_uncertainty_hours: how far off that time might be.")

INSTRUCTION = """You are an expert at working out where and when a photo was taken, using only what is visible.
Look at road markings, signs and scripts, vehicles, vegetation, terrain, architecture, the sun and shadows, weather and season.
You can call tools before answering. Each turn, reply with JSON: "thought" (one or two short sentences of reasoning) and "action" ({{"tool": ..., "args": {{...}}}}).
Tools:
{tools}
You have at most {steps} turns in total; the last one must be answer.{skills}"""


def data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()


class Env:
    def __init__(self, registry: Registry, max_steps: int = 6, max_images: int = 4,
                 ctx_limit: int = 6144, step_reserve: int = 1000, skills: str = ""):
        self.skills = skills   # skill/playbook text appended to the instruction (D-046)
        self.reg, self.max_steps, self.max_images = registry, max_steps, max_images
        self.ctx_limit, self.step_reserve = ctx_limit, step_reserve   # reserve: next observation + reply

    def reset(self, photo: Photo, seed: int = 0, data_root=None) -> list[dict]:
        from retrace.tools.builtin import DATA
        self.ctx = Ctx(photo, np.random.default_rng(seed))
        self.steps, self.images, self.answer = 0, 1, None
        tools = "\n".join(f"- {t.name}({', '.join(t.args)}): {t.description}" for t in self.reg.tools.values())
        tools += f"\n- answer({', '.join(ANSWER_ARGS)}): {ANSWER_DESC}"
        jpeg = ((data_root or DATA) / photo.image_path).read_bytes()
        return [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url(jpeg)}},
            {"type": "text", "text": INSTRUCTION.format(tools=tools, steps=self.max_steps,
                                                               skills="\n\n" + self.skills if self.skills else "")}]}]

    def allowed(self, prompt_tokens: int) -> list[str]:
        """Actions the budgets allow this turn (answer is always allowed)."""
        if self.steps >= self.max_steps - 1 or prompt_tokens + self.step_reserve > self.ctx_limit:
            return ["answer"]
        names = [n for n in self.reg.tools if not (n == "view" and self.images >= self.max_images)]
        return names + ["answer"]

    def schema(self, names: list[str]) -> dict:
        def act(name: str, props: dict) -> dict:
            return {"type": "object", "additionalProperties": False, "required": ["tool", "args"],
                    "properties": {"tool": {"type": "string", "enum": [name]},
                                   "args": {"type": "object", "properties": props, "required": list(props),
                                            "additionalProperties": False}}}
        acts = [act(n, ANSWER_ARGS if n == "answer" else self.reg.tools[n].args) for n in names]
        return {"type": "object", "additionalProperties": False, "required": ["thought", "action"],
                "properties": {"thought": {"type": "string", "maxLength": 400},
                               "action": acts[0] if len(acts) == 1 else {"anyOf": acts}}}

    def step(self, action: dict) -> tuple[dict | None, bool]:
        """Run one action. Returns (observation message or None, done)."""
        self.steps += 1
        name, args = action["tool"], action["args"]
        if name == "answer":
            self.answer = args
            return None, True
        obs = self.reg.call(self.ctx, name, args)
        left = self.max_steps - self.steps
        content = [{"type": "text", "text": f"[{name}] {obs.text}\n(turns left: {left})"}]
        if obs.image is not None:
            self.images += 1
            content.insert(0, {"type": "image_url", "image_url": {"url": data_url(obs.image)}})
        return {"role": "user", "content": content}, False
