# Once we have a trained model, we can use it to help train a better model...
# Least that's the hope of this experiment. We'll use the trained model to
# create embeddings and then create triplet groups that are "hard negatives"
# that is negative examples that are close to their positive example.
# this training set can then be used to help further push negatives away

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

# * `Ncat = torch.cat(cand_embs, dim=0)  # [Nc, D], CPU`
#   → a 2-D tensor with `Nc` candidate vectors, each `D`-dim, stored on CPU.
# * `emb[i:i+1]  # [1, D]` → a single row (kept 2-D) for matmul.
# * `a @ Ncat.T  # [Nc]` → cosine scores for the anchor against all `Nc` candidates.

import argparse
import json
import os
import pickle
import random
from multiprocessing import get_context
from typing import Dict, List, Optional, Tuple

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# -----------------------------------------------------------------------------
# global constant to restrict the number of hard negatives we find
MAX_NEGATIVES_PER_PAIR = 10

# type alias for readability
GroupEmbeddings = Dict[str, Tuple[List[str], torch.Tensor]]

# Globals in worker processes
_GROUPS: Optional[GroupEmbeddings] = None  # dict[group] = (names, emb_cpu)
_DEVICE: str = "cpu"


# -----------------------------------------------------------------------------
def _init_worker(pkl_path: str, device: str):
    global _GROUPS, _DEVICE
    _DEVICE = device
    with open(pkl_path, "rb") as f:
        _GROUPS = pickle.load(f)  # loaded once per worker


# -----------------------------------------------------------------------------
def set_seed(seed=42):
    # random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)


# -----------------------------------------------------------------------------
# def generate_embeddings(model, triplets):

#     # collect all the unique name entries and group ids
#     grouped_names = defaultdict(list)
#     for triplet in triplets:
#         grouped_names[triplet["anchor_group"]].extend([triplet["anchor"], triplet["positive"]])

#     # remove duplicate names from each group
#     for group in grouped_names:
#         grouped_names[group] = list(set(grouped_names[group]))

#     # encode all the names
#     group_embeddings = {}
#     progress = tqdm(grouped_names.items(), initial=0, desc="Embedding", leave=True)
#     for group, names in progress:
#         embeddings = model.encode(names, convert_to_tensor=True, show_progress_bar=False)
#         group_embeddings[group] = (names, embeddings)


#     return group_embeddings
def generate_embeddings(model: SentenceTransformer, triplets):
    """
    Returns: dict[group_id] = (names: List[str], emb: torch.FloatTensor [G, D], L2-normalized, on CPU)
    """
    from collections import defaultdict

    grouped_names = defaultdict(set)  # set -> unique per group

    # collect unique anchor/positive per group (negatives belong to other groups)
    for t in triplets:
        g = str(t["anchor_group"])
        grouped_names[g].add(t["anchor"])
        grouped_names[g].add(t["positive"])

    group_embeddings = {}
    model.eval()
    with torch.no_grad():
        progress = tqdm(grouped_names.items(), desc="Embedding", leave=True)
        for group, names_set in progress:
            names = list(names_set)
            # encode -> tensor on CPU; normalize for cosine
            emb = model.encode(
                names,
                convert_to_tensor=True,
                normalize_embeddings=True,  # ensures unit length
                show_progress_bar=False,
            )
            emb = emb.to("cpu", dtype=torch.float32)
            group_embeddings[group] = (names, emb)
    model.train()
    return group_embeddings


# -----------------------------------------------------------------------------
def load_triplets(filename):

    triplets = []
    with open(filename, "r", encoding="utf-8") as f:
        triplets = [json.loads(line) for line in f]

    return triplets


# -----------------------------------------------------------------------------
# def mine_group_hard_negatives(args):

#     group_id, margin, sample_size = args
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # load group embeddings
#     with open('group_embeddings.pkl', 'rb') as f:
#         group_embeddings = pickle.load(f)

#     names, group_tensor = group_embeddings[group_id]
#     group_tensor = group_tensor.to(device)
#     n = len(names)
#     # need to have at least one positive with the anchor
#     if n < 2:
#         return []

#     hard_triplets = []

#     # pre-compute negative group candidates
#     group_id_set = set(group_embeddings.keys())
#     group_ids_other = list(group_id_set - {group_id})
#     random.shuffle(group_ids_other)

#     for i in range(n):
#         for j in range(i + 1, n):
#             anchor = names[i]
#             positive = names[j]
#             emb_anchor = group_tensor[i].unsqueeze(0)
#             emb_positive = group_tensor[j].unsqueeze(0)

#             # compute anchor-positive cosine similarity
#             sim_ap = torch.nn.functional.cosine_similarity(emb_anchor, emb_positive)

#             # sample a number of random groups
#             negative_group_ids = random.sample(group_ids_other, k=sample_size)

#             for neg_group in negative_group_ids:
#                 neg_names, emb_negative = group_embeddings[neg_group]
#                 emb_negative = emb_negative.to(device)

#                 # compute anchor-negative cosine similarity
#                 sim_an = torch.nn.functional.cosine_similarity(emb_anchor, emb_negative)

#                 # find hard negatives
#                 mask = sim_an > (sim_ap - margin)
#                 if torch.any(mask):
#                     neg_indicies = torch.nonzero(mask, as_tuple=False).squeeze(1).tolist()
#                     hard_negatives = []
#                     for idx in neg_indicies:
#                         hard_triplets.append({
#                             "texts": [anchor, positive, neg_names[idx], group_id]
#                         })
#                     random.shuffle(hard_negatives)
#                     hard_triplets.extend(hard_negatives[:MAX_NEGATIVES_PER_PAIR])

#     return hard_triplets


def mine_group_hard_negatives(args):
    """
    args = (group_id:str, margin:float, sample_size:int, semi_hard:bool)
    Uses globals: _GROUPS (loaded by _init_worker)
    """

    global _GROUPS
    if _GROUPS is None:
        raise RuntimeError("Worker globals non initialized. did you pass initializer=_init_worker?")
    groups = _GROUPS
    group_id, margin, sample_size, semi_hard = args

    # access preloaded embeddings (CPU)
    names, emb_g = groups[group_id]  # emb_g: [Ng, D] on CPU, normalized
    n = len(names)
    if n < 2:
        return []

    # candidate negative groups once
    all_groups = list(_GROUPS.keys())
    all_groups.remove(group_id)

    hard_triplets = []

    # for every unordered pair (i, j) in this group
    with torch.no_grad():
        for i in range(n):
            for j in range(i + 1, n):
                anchor = names[i]
                positive = names[j]
                a = emb_g[i].unsqueeze(0)  # [1, D]
                p = emb_g[j].unsqueeze(0)  # [1, D]

                # cosine since normalized -> dot product
                sim_ap = torch.sum(a * p, dim=1)  # scalar tensor

                # sample a small pool of negative groups
                if sample_size >= len(all_groups):
                    neg_groups = all_groups
                else:
                    neg_groups = random.sample(all_groups, k=sample_size)

                # gather candidates from sampled groups
                cand_names: List[str] = []
                cand_embs: List[torch.Tensor] = []
                for g in neg_groups:
                    n_names, n_emb = _GROUPS[g]
                    if len(n_names) == 0:
                        continue
                    cand_names.extend(n_names)
                    cand_embs.append(n_emb)
                if not cand_embs:
                    continue
                Ncat = torch.cat(cand_embs, dim=0)  # [Nc, D], CPU

                # cosine with all candidates: [Nc]
                sim_an = (a @ Ncat.T).squeeze(0)

                # semi-hard filter: sim_ap - margin <= sim_an < sim_ap
                if semi_hard:
                    mask = (sim_an >= (sim_ap - margin)) & (sim_an < sim_ap)
                else:
                    # "hard" but not same group: allow sim_an >= sim_ap - margin (may include > sim_ap)
                    mask = sim_an >= (sim_ap - margin)

                if not torch.any(mask):
                    continue

                # select top-K hardest negatives under the mask
                masked = sim_an[mask]
                # indices back to full candidate set
                cand_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)

                k = min(MAX_NEGATIVES_PER_PAIR, masked.numel())
                # topk by similarity (hardest)
                top_vals, top_pos = torch.topk(masked, k=k, largest=True, sorted=True)
                top_idx = cand_idx[top_pos].tolist()

                for idx in top_idx:
                    hard_triplets.append(
                        {"texts": [anchor, positive, cand_names[idx], group_id]}
                    )

    return hard_triplets


# -----------------------------------------------------------------------------
# def find_hard_triplets(group_embeddings, output_file, workers, margin=0.3, sample_size=10):

#     group_ids = list(group_embeddings.keys())

#     # pre-move all shared embeddings to the cpu
#     for group in group_embeddings:
#         names, emb = group_embeddings[group]
#         group_embeddings[group] = (names, emb.cpu())

#     # save the group embeddings so they'll be accessible to workers
#     with open('group_embeddings.pkl', 'wb') as f:
#         pickle.dump(group_embeddings, f)

#     all_hard_triplets = []
#     arg_list = [(group_id, margin, sample_size) for group_id in group_ids]

#     ctx = get_context("spawn")

#     # multi-process across groups
#     with ctx.Pool(processes=workers) as pool:
#         with open(output_file, 'a', encoding='utf-8') as f:
#             for triplets in tqdm(pool.imap_unordered(mine_group_hard_negatives, arg_list), total=len(group_ids)):
#                 for t in triplets:
#                     assert len(t["texts"]) == 4
#                     json.dump({
#                         "anchor": t["texts"][0],
#                         "positive": t["texts"][1],
#                         "negative": t["texts"][2],
#                         "anchor_group": t["texts"][3],
#                     }, f)
#                     f.write('\n')

#     return all_hard_triplets


def find_hard_triplets(
    group_embeddings: Dict[str, Tuple[list, torch.Tensor]],
    output_file: str,
    workers: int,
    margin: float = 0.3,
    sample_size: int = 10,
    semi_hard: bool = True,
):
    """
    Streams triplets to `output_file` as JSONL.
    """
    # move all group emb to CPU (already done) and save once
    pkl_path = os.path.join(os.path.dirname(output_file) or ".", "group_embeddings.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(group_embeddings, f)

    group_ids = list(group_embeddings.keys())
    arg_list = [(gid, margin, sample_size, semi_hard) for gid in group_ids]

    # spawn is safer with tokenizers/threads
    ctx = get_context("spawn")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with ctx.Pool(
        processes=workers, initializer=_init_worker, initargs=(pkl_path, "cpu")
    ) as pool:
        with open(output_file, "w", encoding="utf-8") as f:
            for triplets in tqdm(
                pool.imap_unordered(mine_group_hard_negatives, arg_list),
                total=len(arg_list),
                desc="Mining hard negatives",
            ):
                for t in triplets:
                    assert len(t["texts"]) == 4
                    json.dump(
                        {
                            "anchor": t["texts"][0],
                            "positive": t["texts"][1],
                            "negative": t["texts"][2],
                            "anchor_group": t["texts"][3],
                        },
                        f,
                        ensure_ascii=False,
                    )
                    f.write("\n")


# -----------------------------------------------------------------------------
def save_triplets_to_jsonl(examples, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for ex in examples:
            print(ex)
            assert len(ex["texts"]) == 4
            json.dump(
                {
                    "anchor": ex["texts"][0],
                    "positive": ex["texts"][1],
                    "negative": ex["texts"][2],
                    "anchor_group": ex["texts"][3],
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")


# =============================================================================
if __name__ == "__main__":
    cpus = os.cpu_count()
    workers = 1
    if cpus is not None:
        workers = cpus // 2

    parser = argparse.ArgumentParser(
        prog="mine_hard_negative",
        description="Mines a training set along with a previously trained model for 'hard negative' training samples.",
    )
    parser.add_argument(
        "--triplet_path",
        type=str,
        required=True,
        help="Path to the triplet JSONL file.",
    )
    parser.add_argument(
        "--out_path", type=str, required=True, help="Path to output files."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine tuned sentence transormer model.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        required=False,
        default=0.1,
        help="Defines the margin between groups. Default: 0.1",
    )
    parser.add_argument(
        "--workers",
        type=int,
        required=False,
        default=workers,
        help=f"Defines the number of worker to use to process groups. Default: {workers}",
    )
    args = parser.parse_args()

    triplet_path = args.triplet_path
    os.makedirs(args.out_path, exist_ok=True)
    analysis_path = os.path.join(args.out_path, "analysis")
    os.makedirs(analysis_path, exist_ok=True)

    print(f"📌 output path = {args.out_path}")
    print(f"📌 analysis path = {analysis_path}")
    print(f"📌 margin = {args.margin}")
    print(f"📌 workers = {args.workers}")

    # check for CUDA
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print(f"📌 Using device: {device}")

    # initialize the random number generators to something repeatable
    set_seed()

    print(f"⏳ Loading sentence-transformer model ({args.model_path})...")
    model = SentenceTransformer(args.model_path)
    model.to(device)

    print(f"⏳ Loading triplets from {triplet_path}...")
    triplets = load_triplets(triplet_path)
    print(f"... Number of triplets {len(triplets)}")

    print("🚧 Creating embeddings...")
    embeddings = generate_embeddings(model, triplets)
    print(f"... Number of embeddings {len(embeddings)}")

    triplet_data_file = os.path.join(args.out_path, "hard_training_triplets.jsonl")
    print(f"⚙️ Finding hard negative triplets and streaming to {triplet_data_file}...")
    find_hard_triplets(embeddings, triplet_data_file, args.workers, margin=args.margin)

# python mine_hard_negatives.py --triplet_path output/20250718/test_triplets.jsonl --out_path output/20250718-test --model_path output/20250717/FINAL-fine_tuned_model
