"""M5/M6: run the tool-using agent on a split (or a frozen dev subset) and write a scorable run.

  bash scripts/serve_vllm.sh &                          # vLLM on 127.0.0.1:8000 (MAXIMG=4)
  uv run python scripts/run_agent.py val --subset dev300 --tools view,similar,sun,gazetteer
  uv run python scripts/run_agent.py val --subset dev300 --tools view,similar,sun,gazetteer --placebo all
  uv run python scripts/score_run.py <run_id> --vs <other_run_id>

Writes $RETRACE_DATA/runs/<run_id>/{run.json, answers.jsonl, trajectories.jsonl}.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from retrace.env import Env
from retrace.harness.agent import ChatModel, run_episode
from retrace.middleware import Cache, Fault, Placebo, Trace
from retrace.runs import run_dir, write_run
from retrace.splits.dev import subset_ids
from retrace.tools.builtin import FACTORIES
from retrace.tools.registry import Photo, Registry

DATA = Path(os.environ["RETRACE_DATA"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("--dataset", default="osv5m")
    ap.add_argument("--subset", help="frozen subset, e.g. dev300")
    ap.add_argument("--limit", type=int, help="first N photos only (smoke tests; not a real run)")
    ap.add_argument("--tools", default="view,similar,sun,gazetteer", help="comma list, or 'none'")
    ap.add_argument("--placebo", default="", help="comma list of tools to make placebo, or 'all'")
    ap.add_argument("--fault-rate", type=float, default=0.0)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--max-images", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--skill", default="", help="skills/<name>.md appended to the instruction")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="extra suffix for the run id")
    args = ap.parse_args()

    names = [] if args.tools == "none" else args.tools.split(",")
    placebo = set(names) if args.placebo == "all" else {p for p in args.placebo.split(",") if p}
    if placebo - set(names):
        ap.error(f"--placebo names not in --tools: {sorted(placebo - set(names))}")
    t = time.time()
    tools = [FACTORIES[n]() for n in names]
    print(f"tools loaded in {time.time() - t:.1f}s: {names}", flush=True)
    # middleware matches Tool.name, which can differ from the factory key (similar_sum -> "similar")
    placebo_tools = {tool.name for n, tool in zip(names, tools) if n in placebo}
    reg = Registry(tools, [Trace(), Fault(args.fault_rate), Cache(), Placebo(placebo_tools)])
    model = ChatModel(temperature=args.temperature, seed=args.seed)

    m = pd.read_parquet(DATA / "manifests" / f"{args.dataset}_{args.split}_sanitized.parquet")
    m = m[~m.junk]
    if args.subset:
        m = m[m.id.isin(subset_ids(args.dataset, args.split, args.subset))]
    if args.limit:
        m = m.head(args.limit)
    skills = (Path(__file__).resolve().parents[1] / "skills" / f"{args.skill}.md").read_text().strip()         if args.skill else ""
    photos = [Photo(r.id, r.sanitized_path, r.group_id) for r in m.itertuples()]

    def one(p: Photo) -> dict:
        env = Env(reg, max_steps=args.max_steps, max_images=args.max_images, skills=skills)
        return run_episode(model, env, p, seed=args.seed)

    t = time.time()
    with ThreadPoolExecutor(args.concurrency) as ex:
        res = list(ex.map(one, photos))
    wall = time.time() - t

    short = re.sub(r"[^a-z0-9]+", "-", model.model.split("/")[-1].lower()).strip("-")
    tag = "+".join(names) or "notools"
    if args.skill:
        tag += f"_skill-{args.skill}"
    if args.temperature:
        tag += f"_t{args.temperature:g}-s{args.seed}"
    elif args.seed:   # greedy runs with a non-default seed must not overwrite the seed-0 run
        tag += f"_s{args.seed}"
    if placebo:
        tag += "_placebo-" + ("all" if placebo == set(names) else "+".join(sorted(placebo)))
    ds = "" if args.dataset == "osv5m" else f"{args.dataset}_"
    run_id = f"agent_{short}_{tag}_{ds}{args.split}" + (f"_{args.subset}" if args.subset else "") \
        + (f"_n{args.limit}" if args.limit else "") + (f"_{args.tag}" if args.tag else "")
    answers = [r["answer"] for r in res]
    n_err = sum("error" in a for a in answers)
    write_run(run_id, {
        "dataset": args.dataset, "split": args.split, "subset": args.subset, "model": model.model,
        "config": {"runtime": "vllm 0.33.0 fp16", "tools": names, "placebo": sorted(placebo),
                   "fault_rate": args.fault_rate, "skill": args.skill or None, "max_steps": args.max_steps, "max_images": args.max_images,
                   "temperature": args.temperature, "seed": args.seed, "max_tokens_per_turn": 400, "action_format": "schema-forced JSON",
                   "concurrency": args.concurrency},
        "notes": f"agent loop (M5). wall {wall:.0f}s for {len(answers)} photos, {n_err} errors",
    }, answers)
    with open(run_dir(run_id) / "trajectories.jsonl", "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r["trajectory"]) + "\n")

    calls = pd.Series([x for a in answers for x in a["tools_used"]]).value_counts().to_dict()
    print(f"{run_id}: {len(answers)} photos in {wall:.0f}s ({len(answers) / wall:.2f}/s), {n_err} errors; "
          f"mean steps {sum(a['n_steps'] for a in answers) / len(answers):.2f}; tool calls {calls}")


if __name__ == "__main__":
    main()
