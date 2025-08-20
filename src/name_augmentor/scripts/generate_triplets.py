#!/usr/bin/env python3
import argparse
import csv
import json

# sys.path.insert(0, "../..")
from name_augmentor.triplets import TripletGenerator

# import sys


# =============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate triplets from a CSV.")
    ap.add_argument("--input", required=True, help="Input CSV with columns including a cluster id and a name")
    ap.add_argument("--cluster_col", default="canonical", help="Column to use as cluster id (e.g., 'id' or 'canonical')")
    ap.add_argument("--name_col", default="name", help="Column with full name")
    ap.add_argument("--lang_col", default="language", help="Column with language code (optional)")
    ap.add_argument("--output", required=True, help="Output CSV with triplets")
    ap.add_argument("--per_cluster", type=int, default=2)
    ap.add_argument("--noise_rate", type=float, default=0.5)
    ap.add_argument("--max_ops", type=int, default=2)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    tg = TripletGenerator(seed=args.seed)

    rows = []
    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            cid = r.get(args.cluster_col) or r.get("id") or r.get("canonical")
            name = r.get(args.name_col) or r.get("name")
            lang = r.get(args.lang_col) or r.get("language") or None
            if cid and name:
                rows.append((str(cid), name.strip(), (lang or "").strip() or None))

    trips = tg.generate(rows, per_cluster=args.per_cluster, noise_rate=args.noise_rate, max_ops=args.max_ops)

    with open(args.output, "w", newline="", encoding="utf-8") as g:
        fieldnames = ["cluster_id", "anchor", "anchor_lang", "pos", "pos_lang", "neg", "neg_lang", "neg_strategy", "ops_anchor", "ops_pos"]
        w = csv.DictWriter(g, fieldnames=fieldnames)
        w.writeheader()
        for t in trips:
            w.writerow({
                "cluster_id": t.cluster_id,
                "anchor": t.anchor,
                "anchor_lang": t.anchor_lang or "",
                "pos": t.pos,
                "pos_lang": t.pos_lang or "",
                "neg": t.neg,
                "neg_lang": t.neg_lang or "",
                "neg_strategy": t.neg_strategy,
                "ops_anchor": json.dumps(t.ops_anchor, ensure_ascii=False),
                "ops_pos": json.dumps(t.ops_pos, ensure_ascii=False),
            })
