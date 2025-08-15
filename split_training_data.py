# Split a CSV file up into triplets (+ group) sets for training, validation and test.
# Assumes two columns in the CSV a "group" and a "name" column

import argparse
import itertools
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset import TripletDataset


# -----------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)


# -----------------------------------------------------------------------------
def load_groups_from_csv(csv_path, data_col='name'):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[data_col])
    df[data_col] = df[data_col].astype(str)
    return df


# -----------------------------------------------------------------------------
def split_by_group(df: pd.DataFrame, train_min_size: int = 50, val_test_min: int = 10, val_ratio=0.5, random_state=42):
    # count each group of names
    group_counts = df['id'].value_counts()

    # select groups for training
    train_group_ids = group_counts[group_counts >= train_min_size].index.tolist()

    # remaining groups become valdation and test groups
    remaining_group_ids = group_counts[(group_counts < train_min_size) & (group_counts >= val_test_min)].index.tolist()

    # split the test and validation groups with no overlap
    val_group_ids, test_group_ids = train_test_split(remaining_group_ids, test_size=1 - val_ratio, random_state=random_state)

    # assign rows to splits based on group ids
    train_df = df[df['id'].isin(train_group_ids)].copy()
    val_df = df[df['id'].isin(val_group_ids)].copy()
    test_df = df[df['id'].isin(test_group_ids)].copy()

    return train_df, val_df, test_df


# -----------------------------------------------------------------------------
def stratified_split_by_group(
    df: pd.DataFrame,
    group_col: str = "id",
    min_size: int = 2,
    cutoff: Optional[int] = None,                      # keep groups >= cutoff (falls back to min_size if None)
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),  # (train, val, test) must sum to 1
    bins: Sequence[float] = (10, 20, 50, 100, 200, np.inf),
    random_state: int = 42,
    max_groups_per_bucket: Optional[Dict[str, int]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split by group size (no cross-split leakage).
    Returns (train_df, val_df, test_df), each containing whole groups only.
    """

    assert abs(sum(ratios) - 1.0) < 1e-9, "ratios must sum to 1.0"

    # 1) One size per group
    sizes = df.groupby(group_col).size()

    # 2) Apply lower bound
    keep_threshold = cutoff if cutoff is not None else min_size
    sizes = sizes[sizes >= keep_threshold]

    if sizes.empty:
        raise ValueError("No groups meet the size threshold.")

    # 3) Define bucket edges/labels
    edges = [keep_threshold - 1] + list(bins)
    labels = [
        f"{int(edges[i]) + 1}-{('∞' if np.isinf(edges[i+1]) else int(edges[i+1]))}"
        for i in range(len(edges) - 1)
    ]

    # 4) Assign bucket per group
    buckets = pd.cut(sizes, bins=edges, labels=labels, right=True, include_lowest=True)

    rng = np.random.RandomState(random_state)
    train_ids, val_ids, test_ids = [], [], []

    # For summary
    summary_data = []

    # 5) Iterate buckets
    for label in buckets.cat.categories:
        idx = sizes.index[buckets == label]
        n = len(idx)
        if n == 0:
            continue

        # Optional: cap bucket size
        if max_groups_per_bucket and label in max_groups_per_bucket:
            cap = max_groups_per_bucket[label]
            if n > cap:
                idx = rng.choice(idx, size=cap, replace=False)
                n = cap

        idx = np.array(idx)
        rng.shuffle(idx)

        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))

        train_ids.extend(idx[:n_train])
        val_ids.extend(idx[n_train:n_train + n_val])
        test_ids.extend(idx[n_train + n_val:])

        summary_data.append({
            "bucket": label,
            "total_groups": n,
            "train_groups": n_train,
            "val_groups": n_val,
            "test_groups": n - n_train - n_val
        })

    # 6) Sanity checks
    expected_groups = len(sizes)
    all_ids = set(train_ids) | set(val_ids) | set(test_ids)
    assert len(all_ids) == expected_groups, "Some groups are missing from splits!"
    assert len(all_ids) == (len(train_ids) + len(val_ids) + len(test_ids)), "Overlap detected between splits!"

    # 7) Summary printout
    summary_df = pd.DataFrame(summary_data)
    summary_df.loc["TOTAL"] = [
        "TOTAL",
        summary_df["total_groups"].sum(),
        summary_df["train_groups"].sum(),
        summary_df["val_groups"].sum(),
        summary_df["test_groups"].sum(),
    ]
    print("\n--- Split Summary by Bucket ---")
    print(summary_df.to_string(index=False))

    # 8) Build DataFrames
    train_df = df[df[group_col].isin(train_ids)].copy()
    val_df = df[df[group_col].isin(val_ids)].copy()
    test_df = df[df[group_col].isin(test_ids)].copy()

    return train_df, val_df, test_df


# -----------------------------------------------------------------------------
def build_triplet_list(groups: pd.DataFrame):

    group_to_names = defaultdict(list)
    for _, row in groups.iterrows():
        group_to_names[row['id']].append(row['name'])

    all_group_ids = list(group_to_names.keys())

    triplet_set = set()

    progress = tqdm(all_group_ids, initial=0, desc="Building", leave=False)
    for anchor_group in progress:
        positives = group_to_names[anchor_group]
        if len(positives) < 2:
            continue

        # all unique anchor-positive pairs
        ap_pairs = list(itertools.combinations(positives, 2))
        random.shuffle(ap_pairs)
        # TODO: should we restrict the max number of triplets in a group? (EG ap_pairs[:max_trips]:)
        for anchor, postive in ap_pairs:
            # pick a negative from a different group
            negative_group_choices = [g for g in all_group_ids if g != anchor_group and len(group_to_names[g]) > 0]
            if not negative_group_choices:
                continue
            neg_group = random.choice(negative_group_choices)
            negative = random.choice(group_to_names[neg_group])

            triplet = (anchor, postive, negative, anchor_group)

            triplet_set.add(triplet)

    return triplet_set


# -----------------------------------------------------------------------------
# sample up to K unordered pairs from a list of items without materializing all combinations
def sample_pairs(items: List[str], rng, k: Optional[int]) -> List[Tuple[str, str]]:
    """_summary_

    Args:
        items (List[str]): list of strings to create combinations of
        rng (Generator): Random number generator
        k (Optional[int]): number of unordered pairs to generate

    Returns:
        List[Tuple[str, str]]: pair of strings generated from the list
    """
    s = len(items)
    # total unordered pairs
    total = s * (s - 1) // 2
    if not k or k >= total:
        # small group: generate all combinations
        # (safe because total is small or uncapped)
        from itertools import combinations
        pairs = list(combinations(items, 2))
        rng.shuffle(pairs)
        return pairs
    # large group: sample k unique unordered pairs
    # sample indices until we have k unique (a<p)
    seen = set()
    pairs_idx = []
    # oversample factor to reduce resampling loops
    target = k
    while len(pairs_idx) < target:
        a = rng.integers(0, s, size=target - len(pairs_idx))
        p = rng.integers(0, s, size=target - len(pairs_idx))
        mask = a != p
        a = a[mask]
        p = p[mask]
        # enforce unordered (a<p)
        hi = np.maximum(a, p)
        lo = np.minimum(a, p)
        for i, j in zip(lo.tolist(), hi.tolist()):
            key = (i, j)
            if key not in seen:
                seen.add(key)
                pairs_idx.append(key)
                if len(pairs_idx) == target:
                    break
    return [(items[i], items[j]) for (i, j) in pairs_idx]


# -----------------------------------------------------------------------------
def build_triplet_list_fast(
    groups: pd.DataFrame,
    group_col: str = "id",
    name_col: str = "name",
    min_group_size: int = 2,
    max_pos_pairs_per_group: Optional[int] = 200,   # cap per group; None = no cap
    negatives_per_pair: int = 1,                    # >1 if you want multiple negatives per (a,p)
    seed: int = 42
):
    """
    Build (anchor, positive, negative, group_id) triplets.

    - Sample unordered anchor-positive pairs (combinations) per group,
      capped by `max_pos_pairs_per_group`.
    - Sample negatives by random other group (resampling if it matches anchor's group).
    - Skips groups with < min_group_size.
    """

    rng = np.random.default_rng(seed)

    # 1) group -> [names]
    g2names = groups.groupby(group_col, sort=False)[name_col].apply(list)

    # 2) keep only usable groups
    g2names = g2names[g2names.map(len) >= min_group_size]
    all_group_ids = g2names.index.to_numpy()
    n_groups = len(all_group_ids)
    if n_groups == 0:
        return []

    triplets: List[Tuple[str, str, str, object]] = []

    # 3) main loop (fast ops only; no per-iteration list rebuilds)
    for gid in tqdm(all_group_ids, desc="Building triplets", leave=True):
        names = g2names.loc[gid]
        # sample positive pairs
        pos_pairs = sample_pairs(names, rng, max_pos_pairs_per_group)

        # negatives: sample group ids uniformly until it differs from gid
        for (a, p) in pos_pairs:
            for _ in range(negatives_per_pair):
                # draw a negative group (avoid anchor's group)
                while True:
                    j = int(rng.integers(0, n_groups))
                    neg_gid = all_group_ids[j]
                    if neg_gid != gid:
                        break
                # pick a random name from negative group
                neg_name = rng.choice(g2names.loc[neg_gid])
                triplets.append((a, p, neg_name, gid))

    return triplets


# -----------------------------------------------------------------------------
def save_triplets_to_jsonl(examples, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for ex in tqdm(examples, desc="Saving", leave=True):
            assert len(ex.texts) == 4
            # json.dump({
            #     "anchor": ex.texts[0],
            #     "positive": ex.texts[1],
            #     "negative": ex.texts[2],
            #     "anchor_group": ex.texts[3],
            # }, f, ensure_ascii=False)
            # f.write('\n')
            trips = {
                "anchor": ex.texts[0],
                "positive": ex.texts[1],
                "negative": ex.texts[2],
                "anchor_group": ex.texts[3],
            }
            f.write(json.dumps(trips, ensure_ascii=False) + "\n")


# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="split_training_data", description="Splits a CSV file into training, validation, and testing triplet (+group) JSONL files."
    )
    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV file.')
    parser.add_argument('--out_path', type=str, required=True, help='Path to output files.')
    parser.add_argument('--group_col', type=str, required=False, default='id', help='Column header in CSV file that denotes the group. Default: \'id\'')
    parser.add_argument('--data_col', type=str, required=False, default='name', help='Column header in CSV file that denotes the data. Default: \'name\'')
    parser.add_argument('--min_large', type=int, required=False, default=20, help='Defines the minium size of a \'large\' group. Default: 20')
    parser.add_argument('--min_small', type=int, required=False, default=2, help='Defines the minium size of a \'small\' group. Default: 2')
    args = parser.parse_args()

    csv_path = args.csv_path
    os.makedirs(args.out_path, exist_ok=True)

    print(f"📌 output path = {args.out_path}")

    # initialize the random number generators to something repeatable
    set_seed()

    print(f"⏳ Loading and splitting groups from {args.csv_path}...")
    groups = load_groups_from_csv(csv_path, data_col=args.data_col)
    training_df, validation_df, test_df = stratified_split_by_group(groups, cutoff=args.min_small, random_state=42, bins=(30, 40, 50, 100, 200, np.inf))
    # training_df, validation_df, test_df = split_by_group(groups, train_min_size=args.min_large, val_test_min=args.min_small, val_ratio=0.5, random_state=42)

    # Split training data into a training, validation, and test data.
    print("🚧 Creating triplet training dataset")
    # training_set = build_triplet_list(training_df)
    training_set = build_triplet_list_fast(training_df, min_group_size=args.min_small)
    training_dataset = TripletDataset(training_set)
    triplet_data_file = os.path.join(args.out_path, "training_triplets.jsonl")
    print(f"📝 Saving training triplets to {triplet_data_file}...")
    save_triplets_to_jsonl(training_dataset, triplet_data_file)

    # print("🚧 Creating triplet validation dataset")
    # validation_set = build_triplet_list(validation_df)
    validation_set = build_triplet_list_fast(validation_df, min_group_size=args.min_small)
    validation_dataset = TripletDataset(validation_set)
    triplet_data_file = os.path.join(args.out_path, "validation_triplets.jsonl")
    print(f"📝 Saving validation triplets to {triplet_data_file}...")
    save_triplets_to_jsonl(validation_dataset, triplet_data_file)

    print("🚧 Creating triplet test dataset")
    # test_set = build_triplet_list(test_df)
    test_set = build_triplet_list_fast(test_df, min_group_size=args.min_small)
    test_dataset = TripletDataset(test_set)
    triplet_data_file = os.path.join(args.out_path, "test_triplets.jsonl")
    print(f"📝 Saving test triplets to {triplet_data_file}...")
    save_triplets_to_jsonl(test_dataset, triplet_data_file)

    exit(0)

# python split_training_data.py --csv_path data/20250526_biznames_wikidata.csv --out_path output/20250718-test --min_large 50 --min_small 10
