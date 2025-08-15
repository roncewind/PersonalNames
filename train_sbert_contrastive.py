# initial training:
# CUDA_LAUNCH_BLOCKING=1 python train_sbert_contrastive.py --training_path output/20250811/training_triplets.jsonl --validation_path output/20250811/validation_triplets.jsonl --out_path output/20250722
# hard negative training:
# CUDA_LAUNCH_BLOCKING=1 python train_sbert_contrastive.py --training_path output/20250719/training_triplets.jsonl --validation_path output/20250719/validation_triplets.jsonl --hard_path output/20250719/hard_training_triplets.jsonl --out_path output/20250722  --model_path output/20250717/FINAL-fine_tuned_model

import argparse
import itertools
import json
import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from sklearn.manifold import TSNE
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import autocast
from torch.amp import GradScaler  # type: ignore

# from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.optimization import get_scheduler

from dataset import TripletDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
CHECKPOINT_FILENAME_LATEST = "latest_checkpoint.pt"
CHECKPOINT_FILENAME_BEST = "best_checkpoint.pt"
GPU_MONITORING_ENABLED = False
GPU_MONITORING_HANDLE = None


# -----------------------------------------------------------------------------
# helper class to do norm stats during training
class GradNormMeter:
    def __init__(self, clip_norm: float):
        self.clip_norm = float(clip_norm)
        self.reset()

    def reset(self):
        self.pre_vals = []     # store pre-clip norms for this epoch
        self.clipped = 0       # how many steps exceeded clip_norm
        self.n_nonfinite = 0

    def update(self, pre_clip_norm: float):
        if not np.isfinite(pre_clip_norm):
            # skip adding to stats, but count it
            self.n_nonfinite += 1
            return
        self.pre_vals.append(pre_clip_norm)
        if pre_clip_norm > self.clip_norm:
            self.clipped += 1

    def summary(self):
        raw = np.asanyarray(self.pre_vals, dtype=np.float64)
        steps = int(raw.size + self.n_nonfinite)
        if raw.size:
            mean = float(raw.mean())
            std = float(raw.std(ddof=0))
            p95 = float(np.percentile(raw, 95))
        else:
            mean = float("nan")
            std = float("nan")
            p95 = float("nan")
        clip_rate = 100.0 * self.clipped / max(1, steps)
        return {
            "steps": steps,
            "mean": mean,
            "std": std,
            "p95": p95,
            "clip_rate_pct": clip_rate,
            "nonfinite_steps": int(self.n_nonfinite),
        }


# -----------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# -----------------------------------------------------------------------------
def initialize_gpu_monitoring():
    try:
        import pynvml
        pynvml.nvmlInit()
        global GPU_MONITORING_ENABLED
        global GPU_MONITORING_HANDLE
        # only monitor gpu 0 now
        GPU_MONITORING_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
        GPU_MONITORING_ENABLED = True
    except (ImportError, pynvml.NVMLError):
        GPU_MONITORING_ENABLED = False
        GPU_MONITORING_HANDLE = None


# -----------------------------------------------------------------------------
def get_gpu_usage():
    if not GPU_MONITORING_ENABLED:
        return "⚠️ GPU monitoring not enabled."
    try:
        import pynvml
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(GPU_MONITORING_HANDLE)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(GPU_MONITORING_HANDLE)
        return f"GPU Util: {utilization.gpu}% | Mem: {meminfo.used / 1e6:.1f}MB / {meminfo.total / 1e6:.1f}MB"  # type: ignore
    except pynvml.NVMLError as e:
        return f"NVML Error: {e}"


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
            # print(f'triplet: {triplet}')
            # skip dups
            if triplet in triplet_set:
                continue

            triplet_set.add(triplet)

    return triplet_set


# -----------------------------------------------------------------------------
def visualize_embeddings(model, groups, path, file_prefix, max_groups=10, title="t-SNE of fine-tuned name embeddings"):
    os.makedirs(path, exist_ok=True)
    model.eval()
    words, labels, embeddings = [], [], []
    for i, group in enumerate(groups[:max_groups]):
        group = list(set(group))
        for name in group:
            try:
                emb = model.encode(name)
                embeddings.append(emb)
                labels.append(f"group_{i}")
                words.append(name)
            except Exception as e:
                print(e)
                continue
    if not embeddings:
        print("No embeddings to visualize.")
        return
    tsne = TSNE(n_components=2, random_state=42)
    vectors = np.array(embeddings)
    reduced = tsne.fit_transform(vectors)
    df = pd.DataFrame({'x': reduced[:, 0].tolist(), 'y': reduced[:, 1].tolist(), 'label': labels, 'word': words})
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='x', y='y', hue='label', style='label', s=60)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save_path = os.path.join(path, file_prefix + '.png')
    print(f"📝 Save image to {save_path}")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    model.train()


# -----------------------------------------------------------------------------
def plot_tsne(model, group_dict, path, max_groups=10, epoch=None):
    os.makedirs(path, exist_ok=True)
    title = "t-SNE"
    file_stem = title
    if epoch is not None:
        if int(epoch) < 0:
            title = title + " - pre-trained only"
            file_stem = file_stem + "-PRE"
        else:
            title = title + f" - Epoch {epoch}"
            file_stem = file_stem + f"-E{epoch:03d}"
    else:
        title = title + " - Final fine-tuned name embeddings"
        file_stem = file_stem + "-FINAL"
    visualize_embeddings(model, group_dict, path, file_stem, max_groups=max_groups, title=title)


# -----------------------------------------------------------------------------
# Save an interim checkpoint such that we can restart from this point
def save_checkpoint(model, optimizer, scheduler, epoch, step, global_step, path, file_name):
    os.makedirs(path, exist_ok=True)
    tmp = os.path.join(path, file_name + ".tmp")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'step': step,
        'global_step': global_step,
        # TODO: do we care about saving RNG state?
        # 'rng_python': random.getstate(),
        # 'rng_numpy': np.random.get_state(),
        # 'rng_torch': torch.get_rng_state(),
        # 'rng_cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    }, tmp)
    # atomic replace, at least in POSIX
    os.replace(tmp, os.path.join(path, file_name))


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# --- Validation metrics
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Build pair list of validation triplets
def build_val_pairs_from_triplets(val_triplets):
    # positive and negatives for pairwise evaluation
    pos = [(t["anchor"], t["positive"], 1) for t in val_triplets]
    neg = [(t["anchor"], t["negative"], 0) for t in val_triplets]
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

    return {"roc_auc": roc,
            "pr_auc": pr,
            "acc_at_tau": acc,
            "f1_at_tau": f1_at_tau,
            "tau": tau,
            "n_pairs": int(len(pairs))
            }


# -----------------------------------------------------------------------------
# Evaluate retrieval: R@1, R@Ks
#   R@1, R@5, R@10, etc: recall at 1 , 5, 10, etc
#       percentage of times relevant names are in the the first X items
# This blows up thanks to the similarities calculation... so...
def evaluate_retrieval(model, val_triplets, Ks=(1, 5, 10), batch_size=4096):
    # build mapping: name -> set of group ids
    name2groups = defaultdict(set)
    for t in val_triplets:
        g = str(t["anchor_group"])
        name2groups[str(t["anchor"])].add(g)
        name2groups[str(t["positive"])].add(g)
        # no negatives, on purpose

    anchors = [str(t["anchor"]) for t in val_triplets]
    gallery = sorted({str(t["anchor"]) for t in val_triplets}
                     | {str(t["positive"]) for t in val_triplets}
                     | {str(t["negative"]) for t in val_triplets})

    uniq = sorted(set(anchors) | set(gallery))
    emb_map = embed_unique(model, uniq, batch_size=batch_size)

    A = np.stack([emb_map[a] for a in anchors], axis=0)  # [Qa, D]
    G = np.stack([emb_map[g] for g in gallery], axis=0)  # [Ng, D]
    similarities = A @ G.T  # cosine similarity since normalized

    # Avoid self matches when the exact same string appears in the gallery
    anchor_to_gallery_idx = defaultdict(list)
    token2idx = {g: i for i, g in enumerate(gallery)}
    for qi, a in enumerate(anchors):
        if a in token2idx:
            anchor_to_gallery_idx[qi].append(token2idx[a])

    recall = {K: 0 for K in Ks}
    for qi in range(len(anchors)):
        row = similarities[qi].copy()

        # mask out exact-string self if present
        for self_j in anchor_to_gallery_idx.get(qi, []):
            row[self_j] = -np.inf

        top_idx = np.argpartition(-row, kth=max(Ks) - 1)[:max(Ks)]
        top_idx = top_idx[np.argsort(-row[top_idx])]

        a_groups = name2groups.get(anchors[qi], set())
        top_names = [gallery[j] for j in top_idx]
        for K in Ks:
            hit = any(len(name2groups.get(n, set()) & a_groups) > 0 for n in top_names[:K])
            recall[K] += int(hit)

    Q = len(anchors)
    return {f"R@{K}": recall[K] / max(1, Q) for K in Ks
            | {"queries": Q, "gallery": len(gallery)}}


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
        g = str(t["anchor_group"])
        name2groups[str(t["anchor"])].add(g)
        name2groups[str(t["positive"])].add(g)
        # no need for negatives

    # define our queries and gallery of names
    anchors = np.array([str(t["anchor"]) for t in val_triplets], dtype=object)
    gallery = sorted({str(t["anchor"]) for t in val_triplets}
                     | {str(t["positive"]) for t in val_triplets}
                     | {str(t["negative"]) for t in val_triplets})

    # Deduplicate queries for speed
    uniq_q = np.unique(anchors)   # uniq_q[inv_q] == anchors
    # uniq_q, inv_q = np.unique(anchors, return_inverse=True)   # uniq_q[inv_q] == anchors
    # uniq_q: unique query strings
    # inv_q: maps each original anchor to its uniq_q row

    # Embed unique strings once, normalized so cosine sim == inner product
    to_embed = sorted(set(gallery) | set(uniq_q))
    emb_map = model.encode(
        to_embed, batch_size=4096, convert_to_numpy=True, normalize_embeddings=True
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
    for start in tqdm(range(0, len(uniq_q), batch_query), initial=0, desc="Evaluating", leave=True):
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


# -----------------------------------------------------------------------------
# evaluate the model while training
def evaluate_triplet_model(model, dataloader, margin):
    model.eval()
    device = model.device
    total_gap = 0.0
    correct = 0
    total = 0
    margin_violations = 0
    total_losses = 0

    with torch.no_grad():
        evaluate = tqdm(dataloader, initial=0, desc="Evaluating", leave=True)
        for batch in evaluate:
            anchors = [ex.texts[0] for ex in batch]
            positives = [ex.texts[1] for ex in batch]
            negatives = [ex.texts[2] for ex in batch]
            batch_len = len(anchors)
            texts = anchors + positives + negatives
            inputs = model.tokenize(texts)
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
            embeddings = model.forward(inputs)['sentence_embedding']

            # Get embeddings
            anchor_emb, positive_emb, negative_emb = embeddings[: batch_len], embeddings[batch_len: 2 * batch_len], embeddings[2 * batch_len:]

            positive_sims = util.cos_sim(anchor_emb, positive_emb).diagonal()
            negative_sims = util.cos_sim(anchor_emb, negative_emb).diagonal()

            # calculate total loss
            triplet_loss = torch.relu(negative_sims - positive_sims + margin)
            total_losses += triplet_loss.sum().item()

            # Triplet accuracy
            correct += (positive_sims > negative_sims).sum().item()

            # Cosine gap
            total_gap += torch.sum(positive_sims - negative_sims).item()

            # Margin violations
            margin_violations += (positive_sims < (negative_sims + margin)).sum().item()

            total += len(batch)

    accuracy = correct / total
    avg_gap = total_gap / total
    violations_rate = margin_violations / total
    triplet_losses = total_losses / total
    model.train()
    return accuracy, avg_gap, violations_rate, triplet_losses


# -----------------------------------------------------------------------------
def save_triplets_to_jsonl(examples, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for ex in examples:
            assert len(ex.texts) == 4
            json.dump({
                "anchor": ex.texts[0],
                "positive": ex.texts[1],
                "negative": ex.texts[2],
                "anchor_group": ex.texts[3],
            }, f)
            f.write('\n')


# -----------------------------------------------------------------------------
# .fit() didn't allow us to do gradient clipping, save t-SNE images, etc
# this should be a manual loop replacement
def train(model, training_dataloader, validation_dataloader, validation_set, groups, output_path, epochs=10, margin=0.3, lr=1e-5, tsne_every=1, clip_norm=5.0, log_every=1000, save_every=2500, resume=True):
    model.train()
    # Get the device of the model
    device = model.device
    validation_pairs = build_val_pairs_from_triplets(validation_set)

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # decay every epoch
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
    steps_per_epoch = len(training_dataloader)
    num_training_steps = epochs * steps_per_epoch
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    scaler = GradScaler()
    grad_norm_meter = GradNormMeter(clip_norm)

    start_epoch = 0
    start_step = 0
    global_step = 0
    best_val_loss = float('inf')
    checkpoint_dir = os.path.join(output_path, 'checkpoints')

    # Load from a check point, if resuming
    if resume:
        latest = os.path.join(checkpoint_dir, CHECKPOINT_FILENAME_LATEST)
        # best = os.path.join(checkpoint_dir, CHECKPOINT_FILENAME_BEST)
        # use = best if os.path.exists(best) else latest
        use = latest
        if os.path.exists(use):
            ckpt = torch.load(use, map_location="cpu")
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt.get('epoch', 0)
            start_step = ckpt.get('step')
            global_step = ckpt.get('global_step', start_epoch * len(training_dataloader) + start_step)
            # TODO: restore RNG states
            print(f'📌 Resuming from {use}: epoch={start_epoch}, step={start_step}, global={global_step}')

    for epoch in range(start_epoch, epochs):
        running_loss = 0
        grad_norm_meter.reset()
        progress = tqdm(training_dataloader, initial=0, desc=f"Epoch {epoch + 1}", leave=True)
        for step, batch in enumerate(progress):
            if epoch == start_epoch and step < start_step:
                # skip completed steps
                continue

            # Extract texts from InputExample
            anchor_texts = [ex.texts[0] for ex in batch]
            positive_texts = [ex.texts[1] for ex in batch]
            negative_texts = [ex.texts[2] for ex in batch]
            batch_len = len(anchor_texts)

            # Tokenize all of out inputs at once
            texts = anchor_texts + positive_texts + negative_texts
            inputs = model.tokenize(texts)

            # Make sure everything is moved to the model's device
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

            optimizer.zero_grad(set_to_none=True)

            # autocast to use faster fp16 for many ops; uses fp32 for sensitive ops.
            # # TODO: try bfloat16? does this gpu support bfloat16? 'torch.cuda.is_bf16_supported()'
            with autocast(device_type="cuda", dtype=torch.float16):
                embeddings = model.forward(inputs)['sentence_embedding']

                # Get embeddings
                anchor_emb, positive_emb, negative_emb = embeddings[: batch_len], embeddings[batch_len: 2 * batch_len], embeddings[2 * batch_len:]
                anchor_norm = anchor_emb.norm(dim=1).mean().item()
                positive_norm = positive_emb.norm(dim=1).mean().item()
                negative_norm = negative_emb.norm(dim=1).mean().item()
                # tqdm.write(f"    📊 anchor: {anchor_norm:.4f}, positive: {positive_norm:.4f}, negative: {negative_norm:.4f}")
                # check for NaNs/infs in the loss, protect against CUDA crash
                if not torch.isfinite(anchor_emb).all():
                    tqdm.write(f"❌ Non-finite anchor embedding detected: {anchor_norm:.4f}")
                    continue
                if not torch.isfinite(positive_emb).all():
                    tqdm.write(f"❌ Non-finite positive embedding detected: {positive_norm:.4f}")
                    continue
                if not torch.isfinite(negative_emb).all():
                    tqdm.write(f"❌ Non-finite negative embedding detected: {negative_norm:.4f}")
                    continue

                # Compute cosine similarity
                cos_sim_ap = F.cosine_similarity(anchor_emb, positive_emb)
                cos_sim_an = F.cosine_similarity(anchor_emb, negative_emb)

                with torch.no_grad():
                    pos_mean = cos_sim_ap.detach().mean().item()
                    neg_mean = cos_sim_an.detach().mean().item()
                    gap_mean = (cos_sim_ap - cos_sim_an).detach().mean().item()

                    # margin violation
                    viol_rate = ((cos_sim_an - cos_sim_ap + margin) > 0).float().mean().item()

                # Compute triplet loss
                triplet_loss = F.relu(cos_sim_an - cos_sim_ap + margin).mean()

            # check for NaNs/infs in the loss, protect against CUDA crash
            if not torch.isfinite(triplet_loss).item():
                tqdm.write(f"❌ Non-finite loss detected: {triplet_loss.item():.4f}")
                continue

            # scale the loss to avoid fp16 underflow, remove scaler if using bf16
            scaler.scale(triplet_loss).backward()

            # unscale the gradiants back to "real" values BEFORE clipping
            scaler.unscale_(optimizer)

            # clip unscaled gradients
            pre_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            pre_clip_norm = float(pre_clip_norm)

            # update running stats
            grad_norm_meter.update(pre_clip_norm)

            if not np.isfinite(pre_clip_norm):
                # overflow/NaN grads detected. Skip this update cleanly
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue

            # step the optimizer with scaled grads
            scaler.step(optimizer)

            # update the scaler factor (does auto downscale on overflow, upscale on stability)
            scaler.update()

            scheduler.step()
            global_step += 1
            loss_value = triplet_loss.detach().item()
            running_loss = 0.9 * running_loss + 0.1 * loss_value

            # add loss to our progress bar, on reflection, not sure this has much utility
            # progress.set_postfix(loss=f'{loss_value:.4f}')

            # -----------------------------------------------------------------
            # save checkpoints every so often so that we can restart
            if (global_step % save_every) == 0:
                save_checkpoint(model, optimizer, scheduler, epoch, step, global_step, os.path.join(output_path, "checkpoints"), CHECKPOINT_FILENAME_LATEST)
                tqdm.write(f"    📝 Save checkpoint @ global step: {global_step:,}")

            # Monitor key metrics as we go
            if (global_step % log_every) == 0:
                gpu_stats = ""
                # log GPU stats
                if GPU_MONITORING_ENABLED:
                    gpu_stats = get_gpu_usage()
                current_lr = optimizer.param_groups[0]['lr']
                tqdm.write(
                    f"    Global step {global_step:,} | Loss: {triplet_loss.item():.4f} | "
                    f"pos={pos_mean:.3f} neg={neg_mean:.3f} gap={gap_mean:.3f} | violation={viol_rate:.3f} | "
                    f"Grad norm(pre): {pre_clip_norm:.2f} | LR: {current_lr:.2e} | {gpu_stats}"
                )

        # ---------------------------------------------------------------------
        # Do a whole bunch of post Epoch validation and statistics
        current_lr = optimizer.param_groups[0]['lr']
        summary = grad_norm_meter.summary()
        tqdm.write(
            f"\nEpoch {epoch + 1}/{epochs} | Loss: {running_loss:.4f} | "
            f"Grad Norm(steps={summary['steps']} mean={summary['mean']:.2f} std={summary['std']:.2f} p95={summary['p95']:.2f} clipped={summary['clip_rate_pct']:.1f} nonfinite_steps={summary['nonfinite_steps']}) | "
            f"LR: {current_lr:.2e}"
        )
        accuracy, avg_gap, violations_rate, avg_val_loss = evaluate_triplet_model(model, validation_dataloader, margin)
        tqdm.write(f"    📊 Validation accuracy: {accuracy:.4f} | Avg cosine similarity gap: {avg_gap:.4f} | Margin violation rate: {violations_rate:.4f} | Validation loss: {avg_val_loss:.4f}\n")
        # save a checkpoint if this is the best so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, step, global_step, os.path.join(output_path, "checkpoints"), CHECKPOINT_FILENAME_BEST)
        # calculate other stats
        pair_stats = evaluate_pairs(model, validation_pairs, batch_size=2048, tau=None, select_tau_mode="max_f1")
        retrieval_stats = evaluate_retrieval_faiss(model, validation_set, Ks=(1, 5, 10))
        tqdm.write(
            "    📊 Validation "
            f"ROC-AUC={pair_stats['roc_auc']:.4f} "
            f"PR-AUC={pair_stats['pr_auc']:.4f} "
            f"f1@τ={pair_stats['f1_at_tau']:.4f} "
            f"ACC@τ={pair_stats['acc_at_tau']:.4f} "
            f"τ={pair_stats['tau']:.4f} "
            f"R@1={retrieval_stats['R@1']:.3f} R@5={retrieval_stats['R@5']:.3f} R@10={retrieval_stats['R@10']:.3f} "
            f"(pairs={pair_stats['n_pairs']}, queries={retrieval_stats['queries']}, gallery={retrieval_stats['gallery']})"
        )
        if (epoch + 1) % tsne_every == 0:
            plot_tsne(model, groups, path=os.path.join(output_path, "analysis"), epoch=epoch)

        tqdm.write(f"    📝 Saving interim model to Epoch-{epoch}-fine_tuned_model...")
        model.save(os.path.join(output_path, f"Epoch-{epoch:03d}-fine_tuned_model"))

    return model


# -----------------------------------------------------------------------------
# just return the list as-is
def input_example_collate_func(batch):
    return batch


# -----------------------------------------------------------------------------
def load_triplets(filename):

    triplets = []
    with open(filename, 'r', encoding='utf-8') as f:
        triplets = [json.loads(line) for line in f]

    return triplets


# -----------------------------------------------------------------------------
def cap_triplets_by_anchor_and_group(triplets, anchor_cap=100, group_cap=1000):

    group_buckets = defaultdict(list)
    for triplet in triplets:
        group_id = triplet["anchor_group"]
        if group_id:
            group_buckets[group_id].append(triplet)

    # cap per group
    capped_by_group = []
    for group_id, triplet_list in group_buckets.items():
        if len(triplet_list) > group_cap:
            capped_by_group.extend(random.sample(triplet_list, group_cap))
        else:
            capped_by_group.extend(triplet_list)

    # next cap per anchor
    anchor_buckets = defaultdict(list)

    # group triplets per anchor
    for triplet in capped_by_group:
        anchor = triplet["anchor"]
        anchor_buckets[anchor].append(triplet)

    # cap per anchor
    capped_triplets = []
    for anchor, triplet_list in anchor_buckets.items():
        if len(triplet_list) > anchor_cap:
            capped_triplets.extend(random.sample(triplet_list, anchor_cap))
        else:
            capped_triplets.extend(triplet_list)

    return capped_triplets


# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="train_sbert_constrastive", description="Trains a contrastive model of biznames."
    )
    # parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV file.')
    parser.add_argument('--training_path', type=str, required=True, help='Path to training triplets in a JSONL file.')
    parser.add_argument('--hard_path', type=str, required=False, help='Path to hard negative training triplets in a JSONL file.')
    parser.add_argument('--validation_path', type=str, required=True, help='Path validation triplets in a JSONL file.')
    parser.add_argument('--out_path', type=str, required=True, help='Path to output files.')
    parser.add_argument('--model_path', type=str, required=False, default="distiluse-base-multilingual-cased-v1", help='Sentence transformer model. Default: \'distiluse-base-multilingual-cased-v1\'')
    parser.add_argument('--batch_size', type=int, required=False, help='Training batch size. Default: auto tune')
    parser.add_argument('--epochs', type=int, required=False, default=5, help='Number of epochs to train. Default: 5')
    parser.add_argument('--margin', type=float, required=False, default=0.3, help='Defines the margin between groups. Default: 0.3')
    parser.add_argument('--learning_rate', type=float, required=False, default=1e-5, help='Defines the learning rate. Default: 1e-5')
    parser.add_argument('--hard_ratio', type=float, required=False, default=1.0, help='Defines the percentage of hard negatives to use in training. Default: 1.0')
    args = parser.parse_args()

    # csv_path = args.csv_path
    os.makedirs(args.out_path, exist_ok=True)
    analysis_path = os.path.join(args.out_path, "analysis")
    os.makedirs(analysis_path, exist_ok=True)
    checkpoint_path = os.path.join(args.out_path, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)

    print(f"📌 output path = {args.out_path}")
    print(f"📌 analysis path = {analysis_path}")
    print(f"📌 checkpoint path = {checkpoint_path}")

    # check for CUDA
    device = "cpu"
    batch_size = args.batch_size
    pin_memory = False
    if not batch_size:
        batch_size = 32
    if torch.cuda.is_available():
        print(f"📌 CUDA version: {torch.version.cuda}")  # type: ignore
        print(f"📌 CUDNN version: {torch.backends.cudnn.version()}")
        batch_size = 64 if not args.batch_size else args.batch_size
        device = "cuda"
        pin_memory = True
        # Disable TF32 (TensorFloat32) to avoid precision instability
        # torch.set_float32_matmul_precision("highest")
        # torch.backends.cuda.matmul.allow_tf32 = False
        # torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.fp32_precision = "tf32"  # "ieee"
        torch.backends.cudnn.conv.fp32_precision = "tf32"  # "ieee" # type: ignore
        print("📌 matmul fp32_precision: ", torch.backends.cuda.matmul.fp32_precision)
        print("📌 cudnn conv fp32_precision: ", torch.backends.cudnn.conv.fp32_precision)  # type: ignore

        initialize_gpu_monitoring()

    print(f"📌 Using device: {device}")
    print(f"📌 Batch size: {batch_size}")
    print(f"📌 Pin memory: {pin_memory}")
    print(f"📌 GPU monitoring enabled: {GPU_MONITORING_ENABLED}")

    # initialize the random number generators to something repeatable
    set_seed()

    print(f"⏳ Loading training triplet set ({args.training_path})...")
    training_set = load_triplets(args.training_path)
    print(f"... number of training samples: {len(training_set):,}")
    # when we have the hard negative triplets available add them to the training set
    if args.hard_path:
        print(f"⏳ Loading hard negative triplet set ({args.hard_path})...")
        hard_negative_set = load_triplets(args.hard_path)
        print(f"... number of hard negative training samples, pre-cap: {len(hard_negative_set):,}")
        anchor_cap = 100
        group_cap = 1000
        hard_negative_set = cap_triplets_by_anchor_and_group(hard_negative_set, anchor_cap, group_cap)
        print(f"... capped hard negative training samples at {anchor_cap} samples per anchor and {group_cap} anchors per group")
        print(f"... number of hard negative training samples, post-cap: {len(hard_negative_set):,}")
        keep = int(len(hard_negative_set) * args.hard_ratio)
        print(f"... keeping ~{args.hard_ratio * 100:.1f}% of hard negative training samples: {keep:,} samples")
        mined_sample = random.sample(hard_negative_set, k=min(keep, len(hard_negative_set)))
        training_set.extend(mined_sample)
        print(f"... number of training samples with hard negatives: {len(training_set):,}")
        random.shuffle(training_set)
    training_dataset = TripletDataset(training_set)

    print(f"⏳ Loading validation triplet set ({args.validation_path})...")
    validation_set = load_triplets(args.validation_path)
    validation_dataset = TripletDataset(validation_set)

    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=pin_memory, persistent_workers=True, prefetch_factor=2, collate_fn=input_example_collate_func)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, collate_fn=input_example_collate_func)

    print(f"⏳ Loading sentence-transformer model ({args.model_path})...")
    model = SentenceTransformer(args.model_path)
    model.to(device)
    model.max_seq_length = 32

    print("⚙️  Grouping training data for visualization...")
    training_df = pd.DataFrame(training_set)
    # print(training_df.head())
    training_group = training_df.groupby('anchor_group')['anchor'].apply(list)
    print("🎨 Visualizing pre-trained only embeddings...")
    plot_tsne(model, training_group, path=analysis_path, epoch=-1)

    print("⚙️  Training...")
    model = train(model, training_dataloader, validation_dataloader, validation_set, training_group, args.out_path, margin=args.margin, epochs=args.epochs, lr=args.learning_rate)
    print(f"📝 Saving final model to {args.out_path}/FINAL-fine_tuned_model...")
    model.save(os.path.join(args.out_path, "FINAL-fine_tuned_model"))

    print("🎨 Visualizing final embeddings...")
    plot_tsne(model, training_group, path=analysis_path)
