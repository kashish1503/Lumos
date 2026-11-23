"""Run eval cases and log results."""

import json
import operator
from datetime import datetime
from pathlib import Path

from lumos.vlm import get_edits

OPS = {">": operator.gt, "<": operator.lt, "==": operator.eq}


def check(value, rule):
    for sym in ("==", ">", "<"):
        if rule.startswith(sym):
            return OPS[sym](float(value), float(rule[len(sym):]))
    raise ValueError(rule)

NEUTRAL = {"brightness": 1.0, "contrast": 1.0, "saturation": 1.0,
           "warmth": 0, "sharpness": 1.0}

def run_evals(model, processor, image, cases_path="evals/cases.json", out_dir="evals/runs"):
    cases = json.loads(Path(cases_path).read_text())
    results, passed = [], 0
    for case in cases:
        edits = get_edits(model, processor, image, case["command"])
        if not isinstance(edits, dict):
            ok, failures = False, ["no valid JSON"]
        else:
            failures = [f"{k} {r} (got {edits.get(k)})"
                        for k, r in case["expect"].items()
                        if k not in edits or not check(edits[k], r)]
	    failures += [f"{k} should stay {v} (got {edits[k]})"
                         for k, v in NEUTRAL.items()
                         if k not in case["expect"] and k in edits
                         and abs(float(edits[k]) - v) > 0.05 * max(1, abs(v)) + (2 if k == "warmth" else 0)]
            ok = not failures
        passed += ok
        results.append({"command": case["command"], "edits": edits,
                        "pass": ok, "failures": failures})
        print(f"{'PASS' if ok else 'FAIL'}  {case['command']:25} {failures or ''}")

    print(f"\n{passed}/{len(cases)} passed")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(out_dir) / f"run-{stamp}.json"
    out.write_text(json.dumps({"passed": passed, "total": len(cases), "results": results}, indent=2))
    print(f"Saved {out}")
    return results
