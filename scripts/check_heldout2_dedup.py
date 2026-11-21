#!/usr/bin/env python3
"""B10: prove no heldout2 sentence has n-gram overlap > 0.8 with any train row.
Writes reports/heldout2_dedup.json. Gate: max_overlap <= 0.8."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def ngrams(text, n=5):
    words = text.lower().split()
    return {tuple(words[i:i+n]) for i in range(len(words)-n+1)} if len(words) >= n else {tuple(words)}

def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def main():
    train = [json.loads(l)["text"] for l in open(ROOT / "data/processed/train.jsonl", encoding="utf-8")]
    train_ng = [ngrams(t) for t in train]
    held = [json.loads(l)["text"] for l in open(ROOT / "data/processed/mustcatch_heldout2.jsonl", encoding="utf-8")]
    held += [json.loads(l)["text"] for l in open(ROOT / "data/processed/mustcatch_heldout3.jsonl", encoding="utf-8")]
    worst = []
    for t in held:
        ng = ngrams(t)
        best = max((jaccard(ng, tng) for tng in train_ng), default=0.0)
        worst.append((best, t))
    worst.sort(reverse=True)
    mx = worst[0][0]
    rep = {"n_heldout2": 91, "n_heldout3": len(held) - 91, "n_train": len(train), "max_5gram_overlap": round(mx, 4),
           "gate": 0.8, "pass": mx <= 0.8, "worst_items": [(round(w, 3), t[:80]) for w, t in worst[:10]]}
    out = ROOT / "reports/heldout2_dedup.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in rep.items() if k != "worst_items"}, indent=2))
    for w, t in worst[:5]:
        print(f"  {w:.3f} {t}")
    return 0 if rep["pass"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
