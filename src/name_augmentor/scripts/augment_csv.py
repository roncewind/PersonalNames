#!/usr/bin/env python3
import argparse
import csv
import json

# sys.path.insert(0, "../..")
from name_augmentor import Augmentor, load_default_config

# import sys


# =============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Augment names from a CSV (id,name,language).")
    ap.add_argument("--input", required=True, help="Input CSV with columns: id,name,language(optional)")
    ap.add_argument("--output", required=True, help="Output CSV path")
    ap.add_argument("--max_ops", type=int, default=2)
    ap.add_argument("--noise_rate", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    aug = Augmentor(config=load_default_config(), seed=args.seed)

    with open(args.input, newline="", encoding="utf-8") as f, open(args.output, "w", newline="", encoding="utf-8") as g:
        reader = csv.DictReader(f)
        fieldnames = ["id", "name", "language", "augmented", "ops"]
        writer = csv.DictWriter(g, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            name = row.get("name", "").strip()
            lang = row.get("language", None)
            augmented, ops = aug.augment(name, lang=lang or None, max_ops=args.max_ops, noise_rate=args.noise_rate)
            out = {
                "id": row.get("id", ""),
                "name": name,
                "language": lang,
                "augmented": augmented,
                "ops": json.dumps(ops, ensure_ascii=False)
            }
            writer.writerow(out)
