#!/usr/bin/env python3

# mine for hard negatives using faiss

# Inline comments like `# [Nc, D], CPU` are just quick **shape/device annotations** for readability:

# * `[...]` = **tensor shape** (rows, columns, …).

#   * Example: `[#rows, #cols]` → a 2-D tensor.
# * Letters are **dimension names**:

#   * `N` = total names
#   * `B` = batch size (queries in a batch)
#   * `D` = embedding dimension
#   * `K` = number of neighbors
#   * `Nc` = number of **candidates** after concatenation
#   * `Qa` = number of **queries/anchors**
#   * `Ng` = gallery size
# * The trailing `CPU` / `cuda:0` tells you which **device** the tensor lives on.

# Examples:

# * `Ncat = torch.cat(cand_embs, dim=0)  # [Nc, D], CPU` --> a 2-D tensor with `Nc` candidate vectors, each `D`-dim, stored on CPU.
# * `emb[i:i+1]  # [1, D]` --> a single row (kept 2-D) for matmul.
# * `a @ Ncat.T  # [Nc]` --> cosine scores for the anchor against all `Nc` candidates.

import argparse
import json
import math
import os
import random
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# quiet the HF tokenizers warning if you use num_workers elsewhere
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ------------------------- utils ---------------------------------------------
CANON_PAT = re.compile(r"[\W_]+", flags=re.UNICODE)


# -----------------------------------------------------------------------------
def canonicalize(s: str) -> str:
    s = s.lower().strip()
    s = CANON_PAT.sub("", s)
    return s


# -----------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# -----------------------------------------------------------------------------
def load_triplets_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


# ------------------------- pool build -------------------------


# -----------------------------------------------------------------------------
def build_name_pool_from_triplets(
    triplets: List[dict],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[int]]]:
    """
    From train triplets JSONL, collect unique names per group (anchors & positives).
    Returns:
      names:  np.ndarray[str] shape [N]
      groups: np.ndarray[str]  shape [N]
      grp2idx: dict[group_id] -> list of indices in names/groups
    """
    groups_to_names = defaultdict(set)
    for t in triplets:
        g = str(t["anchor_group"])
        groups_to_names[g].add(str(t["anchor"]))
        groups_to_names[g].add(str(t["positive"]))
        # negatives belong to other groups; don't add them as positives here

    # only keep groups with >=2 examples (need positives)
    names, groups = [], []
    for g, s in groups_to_names.items():
        if len(s) >= 2:
            for n in s:
                names.append(n)
                groups.append(g)

    names = np.array(names, dtype=object)
    groups = np.array(groups, dtype=object)

    grp2idx = defaultdict(list)
    for i, g in enumerate(groups):
        grp2idx[g].append(i)

    return names, groups, grp2idx


# ------------------------- embedding -----------------------------------------
# -----------------------------------------------------------------------------
def encode_all(
    model: SentenceTransformer, texts: List[str], batch_size=2048
) -> np.ndarray:
    """
    Encode and L2-normalize; returns float32 np.ndarray [N, D].
    """
    model.eval()
    with torch.no_grad():
        emb = model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,  # unit length -> cosine == inner product
            show_progress_bar=True,
        ).astype(np.float32, copy=False)
    model.train()
    return emb


# ------------------------- FAISS index ---------------------------------------
# -----------------------------------------------------------------------------
def build_faiss_index(
    emb: np.ndarray,
    index_type: str = "flat",
    nlist: Optional[int] = None,
    use_gpu: bool = False,
    gpu_device: int = 0,
    ivf_nprobe: int = 16,
):
    """
    index_type: "flat" (exact) or "ivf" (coarse quantizer; faster on big N).
    If "ivf", nlist defaults to ~4*sqrt(N).
    """
    import faiss

    d = emb.shape[1]
    if index_type == "flat":
        index = faiss.IndexFlatIP(d)  # inner product == cosine for normalized vectors
    elif index_type == "ivf":
        if nlist is None:
            n = emb.shape[0]
            nlist = max(1024, int(4 * math.sqrt(n)))
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        # train on a sample
        train_sz = min(500_000, emb.shape[0])
        subset = np.random.choice(emb.shape[0], size=train_sz, replace=False)
        index.train(emb[subset])
        index.nprobe = ivf_nprobe
    else:
        raise ValueError("index_type must be 'flat' or 'ivf'")

    if use_gpu:
        # move to a single GPU (or to all GPUs if you prefer)
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, gpu_device, index)

    index.add(emb)
    return index


# ------------------------- mining --------------------------------------------
# -----------------------------------------------------------------------------
# TODO: implement a strategy picker parameter:
#    strategy: "random" (fast) or "nearest" (requires per-group search; not implemented)
def pick_positive(i: int, grp2idx: Dict[str, List[int]], groups: np.ndarray) -> int:
    """
    Choose a positive index j != i from the same group.
    """
    g = groups[i]
    idxs = grp2idx[g]
    if len(idxs) == 2:
        return idxs[0] if idxs[1] == i else idxs[1]
    # random positive ≠ i
    while True:
        j = random.choice(idxs)
        if j != i:
            return j


# -----------------------------------------------------------------------------
def mine_hard_negatives_faiss(
    model: SentenceTransformer,
    train_triplets_path: str,
    out_jsonl: str,
    *,
    index_type: str = "flat",  # "flat" or "ivf"
    use_gpu: bool = False,
    gpu_device: int = 0,
    ivf_nprobe: int = 16,
    batch_query: int = 4096,
    k_neighbors: int = 200,  # neighbors to fetch per anchor (before filtering)
    n_neg_per_anchor: int = 4,  # write at most this many negatives per anchor
    semi_hard: bool = True,  # apply sim_pos - m <= sim_neg < sim_pos
    margin: float = 0.05,  # semi-hard margin in cosine space
    cap_per_group: Optional[int] = None,  # optional: total triplets per anchor_group
    seed: int = 42,
):
    set_seed(seed)

    # 1) Build pool
    print("⏳ Loading training triplets...")
    triplets = load_triplets_jsonl(train_triplets_path)
    print(f"   Triplets: {len(triplets):,}")

    print("⏳ Building name pool from triplets (anchors & positives only)...")
    names, groups, grp2idx = build_name_pool_from_triplets(triplets)
    N = names.shape[0]
    uniq_groups = list(grp2idx.keys())
    print(f"   Names: {N:,}  |  Groups (>=2): {len(uniq_groups):,}")

    if N == 0:
        raise RuntimeError(
            "No groups with >=2 examples found in the provided triplets."
        )

    # 2) Embed all names once
    print("🎛️  Encoding & normalizing embeddings...")
    emb = encode_all(
        model, names.tolist(), batch_size=2048
    )  # [N, D], float32, L2-normalized

    # 3) Build FAISS index
    print(f"🏗️  Building FAISS index ({index_type}, use_gpu={use_gpu})...")
    index = build_faiss_index(
        emb,
        index_type=index_type,
        use_gpu=use_gpu,
        gpu_device=gpu_device,
        ivf_nprobe=ivf_nprobe,
    )
    print("✅ Index ready.")

    # 4) Stream mining & writing
    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    out = open(out_jsonl, "w", encoding="utf-8")

    # precompute canonical strings to avoid trivial near-duplicates
    canon = np.array([canonicalize(s) for s in names], dtype=object)

    # optional per-group cap
    written_per_group = defaultdict(int)
    total_written = 0

    print("⛏️  Mining hard negatives (global NN search)...")
    for start in tqdm(range(0, N, batch_query)):
        stop = min(start + batch_query, N)
        Q = emb[start:stop]  # [B, D]
        # topK neighbors per anchor; get extra to survive filtering
        K = k_neighbors
        D, I = index.search(Q, K + 5)  # include self; we will skip it

        for row in range(stop - start):
            i = start + row
            g_i = groups[i]

            # skip if this group hit its cap
            if cap_per_group is not None and written_per_group[g_i] >= cap_per_group:
                continue

            # choose a positive j from same group
            j = pick_positive(i, grp2idx, groups)
            sim_pos = float(np.dot(emb[i], emb[j]))  # cosine (unit vectors)

            # scan neighbors; filter: not self, different group, not trivial canon match
            # then apply semi-hard (or hard), and keep top n_neg_per_anchor
            picked = 0
            for idx, n_idx in enumerate(I[row]):
                if n_idx < 0:
                    continue
                if n_idx == i:
                    continue
                if groups[n_idx] == g_i:
                    continue
                if canon[n_idx] == canon[i]:
                    continue

                sim_neg = float(D[row, idx])  # already cosine
                if semi_hard:
                    if not (sim_neg >= sim_pos - margin and sim_neg < sim_pos):
                        continue
                else:
                    # "hard": allow sim_neg >= sim_pos - margin (may include >= sim_pos)
                    if sim_neg < sim_pos - margin:
                        continue

                # write triplet
                trip = {
                    "anchor": str(names[i]),
                    "positive": str(names[j]),
                    "negative": str(names[n_idx]),
                    "anchor_group": str(g_i),
                }
                out.write(json.dumps(trip, ensure_ascii=False) + "\n")
                total_written += 1
                written_per_group[g_i] += 1
                picked += 1
                if picked >= n_neg_per_anchor:
                    break

    out.close()
    print(f"✅ Done. Wrote {total_written:,} hard triplets to {out_jsonl}")
    if cap_per_group is not None:
        nonzero = [c for c in written_per_group.values() if c > 0]
        if nonzero:
            print(
                f"   Per-group written stats — mean: {np.mean(nonzero):.1f}, p95: {np.percentile(nonzero,95):.0f}, max: {np.max(nonzero)}"
            )


# ------------------------- CLI -------------------------


# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("Mine global hard negatives with FAISS")
    parser.add_argument(
        "--train_triplets",
        type=str,
        required=True,
        help="Path to training triplets JSONL (no leakage).",
    )
    parser.add_argument(
        "--out_jsonl",
        type=str,
        required=True,
        help="Where to write mined hard-negative triplets.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="SentenceTransformer model (fine-tuned).",
    )
    parser.add_argument(
        "--index_type",
        type=str,
        default="flat",
        choices=["flat", "ivf"],
        help="FAISS index type.",
    )
    parser.add_argument("--use_gpu", action="store_true", help="Use FAISS GPU index.")
    parser.add_argument("--gpu_device", type=int, default=0)
    parser.add_argument("--ivf_nprobe", type=int, default=16)
    parser.add_argument("--batch_query", type=int, default=4096)
    parser.add_argument("--k_neighbors", type=int, default=200)
    parser.add_argument("--n_neg_per_anchor", type=int, default=4)
    parser.add_argument(
        "--semi_hard",
        action="store_true",
        help="Enable semi-hard filter (sim_pos - m <= sim_neg < sim_pos).",
    )
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--cap_per_group", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"📌 Using model: {args.model_path}")
    print(
        f"📌 Index: {args.index_type}  GPU: {args.use_gpu}  k={args.k_neighbors}  n_neg/anchor={args.n_neg_per_anchor}"
    )
    print(f"📌 Semi-hard={args.semi_hard}  margin={args.margin}")
    print(f"📌 batch_query={args.batch_query}  ivf_nprobe={args.ivf_nprobe}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📌 Torch device: {device}")

    set_seed(args.seed)
    model = SentenceTransformer(args.model_path)
    model.to(device)

    mine_hard_negatives_faiss(
        model,
        train_triplets_path=args.train_triplets,
        out_jsonl=args.out_jsonl,
        index_type=args.index_type,
        use_gpu=args.use_gpu,
        gpu_device=args.gpu_device,
        ivf_nprobe=args.ivf_nprobe,
        batch_query=args.batch_query,
        k_neighbors=args.k_neighbors,
        n_neg_per_anchor=args.n_neg_per_anchor,
        semi_hard=args.semi_hard,
        margin=args.margin,
        cap_per_group=args.cap_per_group,
        seed=args.seed,
    )


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()

# python mine_hard_negatives_faiss.py \
#   --train_triplets /path/to/train_triplets.jsonl \
#   --out_jsonl /path/to/hard_negatives.jsonl \
#   --model_path /path/to/FINAL-fine_tuned_model \
#   --index_type flat \
#   --k_neighbors 200 \
#   --n_neg_per_anchor 4 \
#   --semi_hard \
#   --margin 0.05

# python mine_hard_negatives_faiss.py \
#   --train_triplets /path/to/train_triplets.jsonl \
#   --out_jsonl /path/to/hard_negatives.jsonl \
#   --model_path /path/to/FINAL-fine_tuned_model \
#   --index_type ivf --ivf_nprobe 32 \
#   --use_gpu --gpu_device 0 \
#   --k_neighbors 256 \
#   --n_neg_per_anchor 4 \
#   --semi_hard --margin 0.05
