import argparse
import json
import os
import random
from collections import Counter
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import umap
from matplotlib.lines import Line2D
from sentence_transformers import InputExample, SentenceTransformer, util
from sklearn.manifold import TSNE
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# -----------------------------------------------------------------------------
class TripletDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        return self.triplets[index]


# -----------------------------------------------------------------------------
def load_triplets_from_jsonl(filename):
    triplets = []
    with open(filename, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            print(f"⏳ Reading line {index:,}", end='\r')
            data = json.loads(line)
            triplets.append(InputExample(texts=[data["anchor"], data["positive"], data["negative"], data["anchor_group"]]))
    print("\n... ✅ done reading.")
    return triplets


# -----------------------------------------------------------------------------
def plot_f1_vs_threshold(positive_sims: np.ndarray, negative_sims: np.ndarray, path):

    thresholds = np.arange(0.0, 1.01, 0.01)
    true_labels = np.concatenate([np.ones(len(positive_sims)), np.zeros(len(negative_sims))])
    all_sims = np.concatenate([positive_sims, negative_sims])

    f1_scores = []
    precisions = []
    recalls = []

    best_f1 = 0.0
    best_threshold = 0.0

    for t in thresholds:
        preds = (all_sims > t).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average='binary', zero_division=0)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    # plot F1 vs threshold
    plt.figure(figsize=(10, 7))
    plt.plot(thresholds, precisions, label='Precision', color='blue')
    plt.plot(thresholds, recalls, label='Recall', color='green')
    plt.plot(thresholds, f1_scores, label='F1 Score', color='orange')
    plt.axvline(x=float(best_threshold), color='red', linestyle='--', label=f'Best threshold = {best_threshold:.2f}')
    plt.title("Precision, Recall, and F1 Score vs Cosine Similarity Threshold")
    plt.xlabel("Cosine Similarity Threshold")
    plt.ylabel("Scores")
    plt.grid(True)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncols=3)
    plt.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    plt.savefig(os.path.join(path, 'f1_vs_cosine_similarity.png'), bbox_inches='tight', dpi=150)
    plt.close()
    return best_threshold, best_f1


# -----------------------------------------------------------------------------
def plot_similarity_histogram(positive_sims: np.ndarray, negative_sims: np.ndarray, threshold, path):

    sns.histplot(positive_sims, color='green', label='Positive pairs', stat='density', bins=50)
    sns.histplot(negative_sims, color='red', label='Negative pairs', stat='density', bins=50)
    # plt.axvline(x=threshold, color='gray', linestyle='--', label=f'Best Threshold {threshold:.2f}')
    plt.axvline(x=threshold, color='gray', linestyle='--')
    _, y_max = plt.ylim()
    plt.text(threshold - 0.01, y_max * 0.5, f'Best Threshold {threshold:.2f}', rotation=90, va='center', ha='right')
    plt.title("Cosine Similarity Distributions")
    plt.xlabel("Cosine Similarity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'cosine_similarity_distributions.png'), bbox_inches='tight', dpi=150)
    plt.close()


# -----------------------------------------------------------------------------
# evaluate the model: F1, average gap, violation rate, tau, etc
def evaluate_triplet_model(model, dataloader, margin, path):
    model.eval()
    device = model.device
    total_gap = 0.0
    correct = 0
    total = 0
    margin_violations = 0
    categories = Counter()

    all_positive_sims = []
    all_negative_sims = []

    with torch.no_grad():
        progress = tqdm(dataloader, initial=0, desc="Progress", leave=True)
        for batch in progress:
            anchors = [ex.texts[0] for ex in batch]
            positives = [ex.texts[1] for ex in batch]
            negatives = [ex.texts[2] for ex in batch]
            emb_anchors = model.encode(anchors, convert_to_tensor=True, device=device)
            emb_positives = model.encode(positives, convert_to_tensor=True, device=device)
            emb_negatives = model.encode(negatives, convert_to_tensor=True, device=device)

            positive_sims = util.cos_sim(emb_anchors, emb_positives).diagonal()
            negative_sims = util.cos_sim(emb_anchors, emb_negatives).diagonal()

            all_positive_sims.append(positive_sims.cpu())
            all_negative_sims.append(negative_sims.cpu())

            # Triplet accuracy
            correct += (positive_sims > negative_sims).sum().item()

            # Cosine gap
            total_gap += torch.sum(positive_sims - negative_sims).item()

            # Margin violations
            margin_violations += (positive_sims < (negative_sims + margin)).sum().item()

            total += len(batch)

            # calculate confusion
            for sp, sn in zip(positive_sims, negative_sims):
                sp = sp.item()
                sn = sn.item()
                if sp > sn:
                    if sp >= sn + margin:
                        categories["Correct & Satisfied"] += 1
                    else:
                        categories["Correct but Violated"] += 1
                else:
                    categories["Incorrect"] += 1

    accuracy = correct / total
    avg_gap = total_gap / total
    violations_rate = margin_violations / total
    model.train()

    all_pos = torch.cat(all_positive_sims).numpy()
    all_neg = torch.cat(all_negative_sims).numpy()
    best_threshold, best_f1 = plot_f1_vs_threshold(all_pos, all_neg, path)
    print(f"📊 Best F1 Score: {best_f1:.4f} at threshold(tau) {best_threshold:.4f}")
    plot_similarity_histogram(all_pos, all_neg, best_threshold, path)

    return accuracy, avg_gap, violations_rate, categories, best_threshold


# -----------------------------------------------------------------------------
# plot t-SNE for each triplet
def plot_triplet_tsne(model, triplets, path, sample_size=1000):
    device = model.device
    model.eval()

    if sample_size < len(triplets):
        triplets = random.sample(triplets, sample_size)
        # triplets = triplets[:sample_size]

    anchors = [ex.texts[0] for ex in triplets]
    positives = [ex.texts[1] for ex in triplets]
    negatives = [ex.texts[2] for ex in triplets]

    all_texts = anchors + positives + negatives
    roles = (["anchor"] * len(anchors) + ["positive"] * len(positives) + ["negative"] * len(negatives))

    embeddings = model.encode(all_texts, convert_to_tensor=False, device=device)

    # changed perplexity from 30 -> 100 added early_exaggerations
    tsne = TSNE(n_components=2, init="random", random_state=42, perplexity=100, early_exaggeration=12)
    reduced = tsne.fit_transform(embeddings)

    # Plot
    color_map = {"anchor": "blue", "positive": "green", "negative": "red"}
    colors = [color_map[r] for r in roles]

    plt.figure(figsize=(10, 8))

    # 1. plot points
    plt.scatter(reduced[:, 0], reduced[:, 1], c=colors, alpha=0.6, s=10)

    # 2. draw connecting lines
    N = len(triplets)
    for i in range(N):
        a, p, n = reduced[i], reduced[i + N], reduced[i + 2 * N]
        plt.plot([a[0], p[0]], [a[1], p[1]], 'g-', alpha=0.1)
        plt.plot([a[0], n[0]], [a[1], n[1]], 'r-', alpha=0.1)

    # 3. add title and legend
    plt.title("t-SNE of Triplet Embeddings")
    plt.legend(handles=[
        Line2D([0], [0], marker='o', color='w', label='Anchor', markerfacecolor='blue', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Positive', markerfacecolor='green', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Negative', markerfacecolor='red', markersize=8),
    ])
    plt.tight_layout()
    plt.savefig(os.path.join(path, 't-SNE_triplet_embeddings.png'), bbox_inches='tight', dpi=150)
    plt.close()
    model.train()


# -----------------------------------------------------------------------------
# plot UMAP for each triplet
def plot_triplet_umap(model, triplets, path, sample_size=1000):
    device = model.device
    model.eval()

    if sample_size < len(triplets):
        triplets = random.sample(triplets, sample_size)
        # triplets = triplets[:sample_size]

    anchors = [ex.texts[0] for ex in triplets]
    positives = [ex.texts[1] for ex in triplets]
    negatives = [ex.texts[2] for ex in triplets]

    all_texts = anchors + positives + negatives
    roles = (["anchor"] * len(anchors) + ["positive"] * len(positives) + ["negative"] * len(negatives))

    embeddings: np.ndarray = model.encode(all_texts, convert_to_tensor=False, device=device)

    # removed random_state to avoid runtime warning
    # reducer = umap.UMAP(n_components=2, random_state=42)
    reducer = umap.UMAP(n_components=2)
    reduced = cast(np.ndarray, reducer.fit_transform(embeddings))

    # Plot
    color_map = {"anchor": "blue", "positive": "green", "negative": "red"}
    colors = [color_map[r] for r in roles]

    plt.figure(figsize=(10, 8))

    # 1. plot points
    plt.scatter(reduced[:, 0], reduced[:, 1], c=colors, alpha=0.6, s=10)

    # 2. draw connecting lines
    N = len(triplets)
    for i in range(N):
        a, p, n = reduced[i], reduced[i + N], reduced[i + 2 * N]
        plt.plot([a[0], p[0]], [a[1], p[1]], 'g-', alpha=0.1)
        plt.plot([a[0], n[0]], [a[1], n[1]], 'r-', alpha=0.1)

    # 3. add title and legend
    plt.title("UMAP of Triplet Embeddings")
    plt.legend(handles=[
        Line2D([0], [0], marker='o', color='w', label='Anchor', markerfacecolor='blue', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Positive', markerfacecolor='green', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Negative', markerfacecolor='red', markersize=8),
    ])
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'UMAP_triplet_embeddings.png'), bbox_inches='tight', dpi=150)
    plt.close()
    model.train()


# -----------------------------------------------------------------------------
# Build pair list of validation triplets
def build_val_pairs_from_triplets(val_triplets):
    # positive and negatives for pairwise evaluation
    # anchor - texts[0]
    # positive - texts[1]
    # negative - texts[2]
    pos = [(t.texts[0], t.texts[1], 1) for t in val_triplets]
    neg = [(t.texts[0], t.texts[2], 0) for t in val_triplets]
    # TODO: de-dup?
    pairs = list({(a, b, y) for (a, b, y) in pos + neg})
    return pairs


# -----------------------------------------------------------------------------
# embed unique strings once
def embed_unique(model, texts, batch_size=2048):
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32, copy=False)
    return {t: e for t, e in zip(texts, embeddings)}


# -----------------------------------------------------------------------------
# Pairwise metrics: ROC-AUC, PR-AUC, Accuracy/F1 at τ
#   ROC-AUC: Receiver Operating Characteristic Area Under the Curve
#   PR-AUC:  Precision-Recall Area Under the Curve
#   Accuracy at τ: proportion of correct pairs
#   F1 at τ:  harmonic mean of precision and recall
def evaluate_pairs(model, pairs, batch_size=2048, tau=None, select_tau_mode="max_f1", target_precision=0.995):
    # pairs - (anchor, name, 1 or 0) 1 = positive, 0 = negative
    texts = sorted({t for a, b, _ in pairs for t in (a, b)})
    embeddings = embed_unique(model, texts, batch_size=batch_size)
    y_true, y_score = [], []
    for a, b, y in pairs:
        y_true.append(y)
        y_score.append(float(np.dot(embeddings[a], embeddings[b])))
    y_true = np.asarray(y_true, np.int32)
    y_score = np.asarray(y_score, np.float32)

    roc = roc_auc_score(y_true, y_score)
    pr = average_precision_score(y_true, y_score)

    # choose tau on validation if not provided
    if tau is None:
        prec, rec, th = precision_recall_curve(y_true, y_score)
        if select_tau_mode == "max_f1":
            f1 = (2 * prec * rec) / np.clip(prec + rec, 1e-12, None)
            i = int(np.nanargmax(f1))
            tau = float(th[i - 1]) if i > 0 else float(np.median(y_score))
        elif select_tau_mode == "precision_at":
            idx = np.where(prec[:-1] >= target_precision)[0]
            tau = float(th[idx[0]]) if len(idx) else float(th[-1])
        else:
            raise ValueError("select_tau_mode must be 'max_f1' or 'precision_at'")

    y_pred = (y_score >= tau).astype(np.int32)
    acc = float((y_pred == y_true).mean())
    f1_at_tau = f1_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    npv = 0.0
    if (tn + fn) == 0:
        npv = 0.0  # Avoid division by zero if there are no negative predictions
    else:
        npv = tn / (tn + fn)
    ppv = 0.0
    if (tp + fp) == 0:
        ppv = 0.0  # Avoid division by zero if there are no negative predictions
    else:
        ppv = tp / (tp + fp)

    return {"roc_auc": roc,
            "pr_auc": pr,
            "acc_at_tau": acc,
            "f1_at_tau": f1_at_tau,
            "tau": tau,
            "npv": npv,
            "ppv": ppv,
            "n_pairs": int(len(pairs))
            }


# -----------------------------------------------------------------------------
# Evaluate retrieval: R@1, R@Ks
#   R@1, R@5, R@10, etc: recall at 1 , 5, 10, etc
#       percentage of times relevant names are in the the first X items
def evaluate_retrieval_faiss(model, val_triplets, Ks=(1, 5, 10), batch_query=8192):

    import faiss
    faiss.omp_set_num_threads(os.cpu_count() or 8)

    # Build a name to group id dictionary
    from collections import defaultdict
    name2groups = defaultdict(set)  # why didn't the global import work for this?
    for t in val_triplets:
        g = str(t.texts[3])
        name2groups[str(t.texts[0])].add(g)
        name2groups[str(t.texts[1])].add(g)
        # no need for negatives

    # define our queries and gallery of names
    anchors = np.array([str(t.texts[0]) for t in val_triplets], dtype=object)
    gallery = sorted({str(t.texts[0]) for t in val_triplets}
                     | {str(t.texts[1]) for t in val_triplets}
                     | {str(t.texts[2]) for t in val_triplets})

    # Deduplicate queries for speed
    uniq_q = np.unique(anchors)   # uniq_q[inv_q] == anchors
    # uniq_q, inv_q = np.unique(anchors, return_inverse=True)   # uniq_q[inv_q] == anchors
    # uniq_q: unique query strings
    # inv_q: maps each original anchor to its uniq_q row

    # Embed unique strings once, normalized so cosine sim == inner product
    to_embed = sorted(set(gallery) | set(uniq_q))
    emb_map = model.encode(
        to_embed, batch_size=int(batch_query / 2), convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32, copy=False)
    tok2row = {t: i for i, t in enumerate(to_embed)}  # text to row in emb_map

    # build matrices for FAISS input
    G = np.stack([emb_map[tok2row[n]] for n in gallery], axis=0)   # [Ng, D]
    Q = np.stack([emb_map[tok2row[n]] for n in uniq_q], axis=0)    # [Nq, D]
    d = G.shape[1]

    # Build FAISS index (Inner Product for cosine with normalized vectors)
    index = faiss.IndexFlatIP(d)  # exact search, should it be IVF?
    index.add(G)  # type: ignore[call-arg]                # add gallery vectors to the index

    # Precompute self-match indices (mask out exact string matches)
    name_to_gallery_idx = defaultdict(list)
    for j, n in enumerate(gallery):
        name_to_gallery_idx[n].append(j)

    # prep array for top-K neighbor indices per unique query
    maxK = max(Ks)
    sel_idx = np.full((len(uniq_q), maxK), -1, dtype=np.int64)

    # Batched nearest neighbor search
    extra = 5  # fetch a few extra to survive filtering
    for start in tqdm(range(0, len(uniq_q), batch_query), initial=0, desc="Progress", leave=True):
        end = min(start + batch_query, len(uniq_q))
        # Distances, Indices are returned from index.search.
        _, Indices = index.search(Q[start:end], maxK + extra)  # type: ignore[call-arg]  # [B,K+extra] scores/indices

        # for each query in the batch, filter out exact string self-matches
        for r in range(end - start):
            qname = uniq_q[start + r]
            forbid = set(name_to_gallery_idx.get(qname, []))  # drop exact-string self
            take = []
            for idx in Indices[r]:
                if idx < 0:
                    continue
                if idx in forbid:
                    continue
                take.append(idx)
                if len(take) == maxK:
                    break
            # pad with -1 when not enough neighbors survived filtering
            while len(take) < maxK:
                take.append(-1)
            sel_idx[start + r] = take

    # Compute Recall@K over unique queries (same value per duplicate)
    hits = {K: 0 for K in Ks}
    for qi, qname in enumerate(uniq_q):
        q_groups = name2groups.get(qname, set())  # groups for this query name
        nbrs = sel_idx[qi]  # top-K neighbor indices into the gallery
        for K in Ks:
            ok = False
            for j in nbrs[:K]:
                if j < 0:
                    continue
                # hit if neighbor share any group with the query
                if name2groups.get(gallery[j], set()) & q_groups:
                    ok = True
                    break
            if ok:
                hits[K] += 1

    # return recalls computed over the unique queries
    Quniq = len(uniq_q)
    return {f"R@{K}": hits[K] / max(1, Quniq) for K in Ks} | {"queries": Quniq, "gallery": len(gallery)}


# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="test_triplets", description="Tests a fine tuned model with test set saved in JSONL file."
    )
    parser.add_argument('--jsonl_path', type=str, required=True, help='Path to JSONL test examples file.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to fine tuned sentence transormer model.')
    parser.add_argument('--out_path', type=str, required=True, help='Path to save files for further analysis, will create an "analysis directory".')
    parser.add_argument('--device', type=str, required=False, help='Device to run on (cpu/cuda). Default: auto')
    parser.add_argument('--batch_size', type=int, required=False, help='Training batch size. Default: auto tune')
    parser.add_argument('--margin', type=float, required=False, default=0.3, help='Defines the margin between groups. Default: 0.3')
    parser.add_argument('--tau', type=float, required=False, default=None, help='Pick a tau for reporting stats. Default: None')
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)
    analysis_path = os.path.join(args.out_path, "analysis")
    os.makedirs(analysis_path, exist_ok=True)
    print(f"📌 output path = {args.out_path}")
    print(f"📌 analysis path = {analysis_path}")

    jsonl_path = args.jsonl_path

    batch_size = args.batch_size
    if not batch_size:
        batch_size = 32

    device = args.device
    if not device:
        # check for CUDA
        device = "cpu"
        if torch.cuda.is_available():
            batch_size = 64 if not args.batch_size else args.batch_size
            device = "cuda"
    print(f"📌 Using device: {device}")
    print(f"📌 Batch size: {batch_size}")

    print("⏳ Loading model...")
    model = SentenceTransformer(args.model_path)

    print(f"📖 Reading samples from: {jsonl_path}")
    triplets = load_triplets_from_jsonl(jsonl_path)
    print("⏳ Building pairs for pair-wise evaluation...")
    validation_pairs = build_val_pairs_from_triplets(triplets)

    dataset = TripletDataset(triplets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

    print("⏳ Evaluating tests...")
    print("⏳   Evaluating pairs...")
    pair_stats = evaluate_pairs(model, validation_pairs, batch_size=(batch_size * 4), tau=args.tau, select_tau_mode="max_f1")
    print("⏳   Evaluating retrieval statistics...")
    retrieval_stats = evaluate_retrieval_faiss(model, triplets, Ks=(1, 5, 10), batch_query=int(batch_size * 16))
    print("⏳   Evaluating triplet model...")
    accuracy, avg_gap, violations_rate, categories, best_threshold = evaluate_triplet_model(model, dataloader, args.margin, analysis_path)
    for key in categories:
        print(f"\t{key:24}: {categories[key]} ({categories[key] / sum(categories.values()):.2%})")
    plot_triplet_tsne(model, triplets, analysis_path)
    plot_triplet_umap(model, triplets, analysis_path)
    # calculate other stats
    print(f"📊 Test accuracy: {accuracy:.4f} | Avg cosine similarity gap: {avg_gap:.4f} | Margin violation rate: {violations_rate:.4f} | Best threshold: {best_threshold:.2f}")
    print(
        "📊 "
        f"ROC-AUC={pair_stats['roc_auc']:.4f} | "
        f"PR-AUC={pair_stats['pr_auc']:.4f} | "
        f"PPV={pair_stats['ppv']:.4f} | "
        f"NPV={pair_stats['npv']:.4f} | "
        f"f1@τ={pair_stats['f1_at_tau']:.4f} | "
        f"ACC@τ={pair_stats['acc_at_tau']:.4f} | "
        f"τ={pair_stats['tau']:.4f} | "
        f"R@1={retrieval_stats['R@1']:.3f} R@5={retrieval_stats['R@5']:.3f} R@10={retrieval_stats['R@10']:.3f} | "
        f"(pairs={pair_stats['n_pairs']:,}, queries={retrieval_stats['queries']:,}, gallery={retrieval_stats['gallery']:,})"
    )

# python test_triplets.py --model_path output/20250714/FINAL-fine_tuned_model --jsonl_path output/20250714/test_triplets.jsonl --device cuda --batch_size 128 --out_path output/20250714
