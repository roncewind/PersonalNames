
# ANN method selection

- Approximate Nearest Neighbor (ANN) method depends on performance characteristics expected and maching configuration.
  - **IVFFLAT**: Inverted Flat File
  - **HNSW**: Hierarchical Navigable Small World Graph

## Quick Reference Table

| Feature                    | IVFFLAT                               | HNSW                                         |
| -------------------------- | ------------------------------------- | -------------------------------------------- |
| **Speed (Query)**          | Very fast (when tuned well)           | Very fast (often faster than IVFFLAT)        |
| **Accuracy**               | Good (depends on `lists`/`probes`)    | Excellent (near-exact)                       |
| **Index Build Time**       | Fast                                  | Slower                                       |
| **Insert Time**            | Fast                                  | Slower (due to graph maintenance)            |
| **Index Size**             | Smaller                               | Larger (due to graph structure)              |
| **Tunability**             | Easy: `lists`, `probes`               | Complex: `m`, `ef_construction`, `ef_search` |
| **Dynamic Insert Support** | Yes (very good for streaming inserts) | Not ideal (best with bulk static data)       |
| **Best for**               | Large, frequently updated datasets    | Smaller static datasets, high-accuracy use   |

## **IVFFLAT**: Inverted Flat File
  * Pros
    * Fast query times with proper tuning
    * Index builds quickly
    * Good for **large** datasets
    * Good for **frequently updated** tables
    * Simple to tune: `lists` (index build), `probes` (query-time)

  * Cons
    * Needs `ANALYZE` after inserts to perform well
    * Accuracy depends on `lists` and `probes` (may miss close neighbors if not tuned well)

  * 📏 Rules of Thumb

| Parameter | Default | Typical Starting Value              |
| --------- | ------- | ----------------------------------- |
| `lists`   | —       | \~√(num\_rows), e.g., 100 for 10k   |
| `probes`  | 1       | 10–20 for balance of speed/accuracy |

```sql
CREATE INDEX ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
SET ivfflat.probes = 10;
```

## **HNSW**: Hierarchical Navigable Small World Graph
  * Pros
    * **Very high recall**, near-exact
    * Very fast queries
    * No need to `ANALYZE`
    * Great for **read-heavy workloads** where accuracy matters

  * Cons
    * Slower to build
    * Slower inserts (graph structure must be updated)
    * Not ideal for **frequently updated** or streaming datasets

  * 📏 Rules of Thumb

| Parameter         | Default | Typical Starting Value                          |
| ----------------- | ------- | ----------------------------------------------- |
| `m`               | 16      | 16–32 (controls graph connections)              |
| `ef_construction` | 64      | 100–200 (affects index build time and accuracy) |
| `ef_search`       | 40      | 100+ (affects query recall/speed tradeoff)      |

```sql
CREATE INDEX ON embeddings USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 100);
SET hnsw.ef_search = 100;
```

---

## When to Use Which?

| Scenario                                                   | Recommendation                        |
| ---------------------------------------------------------- | ------------------------------------- |
| You have frequent inserts                                  | ✅ **IVFFLAT**                         |
| You want max accuracy, static data                         | ✅ **HNSW**                            |
| You care about low latency + okay with approximate results | ✅ **IVFFLAT**, well-tuned             |
| You're running batch jobs or infrequent updates            | ✅ **HNSW**                            |
| You're just getting started                                | ✅ Use IVFFLAT first — simpler to tune |

## Combo Strategy?

Some teams:

* Use **IVFFLAT for dev / iterative testing**
* Switch to **HNSW for production search** once the data stabilizes

Bottom line, do your research and tune them appropriately

---

# Using **τ (tau)** with PostgreSQL/pgvector.

 The only trick is remembering that pgvector's **cosine operator** returns a **distance**, not similarity.

## Key mapping

* Let **similarity** $s = \cos(\mathbf{q}, \mathbf{x}) \in [-1, 1]$.
* pgvector's **cosine distance** operator `<=>` returns
  $d = 1 - s \in [0, 2]$.
* Your decision rule "match if $s \ge \tau$" becomes
  **"match if $d \le 1 - \tau$"**.

Example: if $\tau = 0.3016$, then accept when `cosine_distance ≤ 1 - 0.3016 = 0.6984`.

---

## Option A: Top-K then apply τ in SQL (simple & index-friendly)

This is the common pattern with IVFFlat/HNSW indexes: get the nearest **K** and filter by τ.

```sql
-- $1 = query vector, $2 = top_k, $3 = tau
WITH params AS (
  SELECT $1::vector AS qv, $2::int AS k, $3::float8 AS tau
)
SELECT
  e.id,
  e.group_id,
  e.name,
  e.language,
  (e.embedding <=> p.qv)       AS cosine_distance,
  1 - (e.embedding <=> p.qv)   AS cosine_similarity
FROM embeddings e, params p
ORDER BY e.embedding <=> p.qv
LIMIT p.k;
```

Then in your app, keep only rows with `cosine_similarity >= tau`.
(You can also add a `WHERE` clause here; see Option B.)

**Why this is nice:** `ORDER BY embedding <=> q LIMIT k` is exactly what pgvector's approximate indexes are built to accelerate.

---

## Option B: Apply τ inside SQL (distance threshold)

Same query, but **also** filter by your τ → distance threshold $1-\tau$:

```sql
-- $1 = query vector, $2 = (1 - tau) as a distance cutoff, $3 = top_k
WITH params AS (
  SELECT $1::vector AS qv, $2::float8 AS max_dist, $3::int AS k
)
SELECT
  e.id,
  e.group_id,
  e.name,
  e.language,
  (e.embedding <=> p.qv)       AS cosine_distance,
  1 - (e.embedding <=> p.qv)   AS cosine_similarity
FROM embeddings e, params p
WHERE (e.embedding <=> p.qv) <= p.max_dist
ORDER BY e.embedding <=> p.qv
LIMIT p.k;
```

For $\tau = 0.3016$, pass `max_dist = 0.6984`.

**Note:** With IVFFlat/HNSW, the index is primarily used for the `ORDER BY … LIMIT`. The `WHERE` cutoff is applied as a filter; performance is still good in practice, but the pure top-K pattern (Option A) is the most index-friendly.

---

## Python example (psycopg)

```python
import numpy as np
import psycopg

TAU = 0.3016
TOP_K = 10

# qvec must be a 1-D float list the same length as your column's dimension
qvec = model.encode([query_name], normalize_embeddings=True)[0].astype(np.float32).tolist()

sql = """
WITH params AS (
  SELECT %s::vector AS qv, %s::float8 AS max_dist, %s::int AS k
)
SELECT id, group_id, name, language,
       (embedding <=> p.qv) AS cosine_distance,
       1 - (embedding <=> p.qv) AS cosine_similarity
FROM embeddings e, params p
WHERE (embedding <=> p.qv) <= p.max_dist
ORDER BY embedding <=> p.qv
LIMIT p.k;
"""

with psycopg.connect(conninfo) as conn, conn.cursor() as cur:
    cur.execute(sql, (qvec, 1.0 - TAU, TOP_K))
    rows = cur.fetchall()
    # rows already honor tau; if you used Option A, filter here instead
```

---

## Indexing tips (pgvector)

* Use cosine ops in the index:

  ```sql
  -- Fast approximate
  CREATE INDEX ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

  -- Or HNSW (often great out-of-the-box)
  CREATE INDEX ON embeddings USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200);
  ```

* Tune probes (IVFFlat):

  ```sql
  SET ivfflat.probes = 10;   -- try 5–20 for recall/speed tradeoff
  ```

* Store **float4/float8** vectors; you **don't need** to pre-normalize for `<=>` to work, but normalizing in your app makes cosine consistent across systems and lets you switch to inner-product (`IndexFlatIP`) elsewhere without surprises.

---

## PostgreSQL bottom-line

* pgvector's `<=>` returns **cosine distance**, so threshold by **`1 - τ`**.
* **Top-K order** then filter by τ is standard; adding a distance `WHERE` is fine too.
* Keep your embeddings normalized and your τ versioned with the model, and you're good to go.

