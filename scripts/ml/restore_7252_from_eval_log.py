#!/usr/bin/env python3
"""Restore 7252 human action_state labels from VLM eval log + partial JSON."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tenniscut.ml.labels import apply_annotation_prefill, get_action_state, is_annotation_complete
from tenniscut.ml.manifest_io import LabelStore, load_jsonl, write_jsonl

LOG_PATH = Path("/tmp/qwen3_7252_eval.log")
EVAL_JSON = ROOT / "datasets/eval/qwen3_vl_2b_7252_v2/qwen3_vl.json"
MANIFEST = ROOT / "datasets/player_actions/manifests/7252_unlabeled.jsonl"


def recover_from_log(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="ignore")
    pat = re.compile(r"\] (OK|MISS|PRED) (\S+) pose=([^/]+)/")
    out: dict[str, str] = {}
    for m in pat.finditer(text):
        _status, sample_id, true_pose = m.groups()
        if true_pose != "-":
            out[sample_id] = true_pose
    return out


def recover_from_eval(path: Path) -> dict[str, str]:
    if not path.exists():
        for legacy in (
            ROOT / "datasets/eval/qwen3_vl_2b_7252/qwen3_vl_crop.json",
            ROOT / "datasets/eval/qwen3_vl_2b_7252/qwen3_vl.json",
        ):
            if legacy.exists():
                path = legacy
                break
        else:
            return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for row in data.get("predictions", []):
        if row.get("has_human_label") and row.get("true_pose"):
            out[row["sample_id"]] = row["true_pose"]
    return out


def main() -> None:
    recovered = recover_from_log(LOG_PATH)
    recovered.update(recover_from_eval(EVAL_JSON))

    store = LabelStore(MANIFEST)
    restored = 0
    for sample in store.samples:
        sid = sample["sample_id"]
        row = apply_annotation_prefill(store._by_id[sid], relabel=False)
        if sid in recovered:
            row["action_state"] = recovered[sid]
            restored += 1
        store._by_id[sid] = row
    store.flush()

    rows = load_jsonl(store.labeled_path)
    complete = sum(1 for r in rows if is_annotation_complete(r))
    missing = [r["sample_id"] for r in rows if get_action_state(r) == "unsure"]
    print(
        json.dumps(
            {
                "restored_action_state": restored,
                "complete": complete,
                "total": len(rows),
                "missing_layer1": len(missing),
                "action_counts": dict(Counter(get_action_state(r) for r in rows)),
                "labeled_path": str(store.labeled_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    missing_path = ROOT / "datasets/player_actions/manifests/7252_missing_layer1.txt"
    missing_path.write_text("\n".join(missing) + "\n", encoding="utf-8")
    print(f"Wrote missing sample ids -> {missing_path}")


if __name__ == "__main__":
    main()
