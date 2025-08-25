
# Setting up Senzing database for vector operations

## References:

- https://senzing.com/docs/4_beta/quickstart_linux/
- https://senzing.zendesk.com/hc/en-us/articles/360041965973-Setup-with-PostgreSQL some names changes for v4

## Postgres vector database

Install a Postgres vector database and import training data embeddings into it using a fine-tuned model.
Further testing can be done using cosine distance and cosine similarity implemented in the Postgres
vector database.

### Install PostgreSQL

```
sudo apt update
sudo apt install postgresql postgresql-contrib
```

#### Switch to the postgres user and create a new db user/pwd if needed

dbuser is the user your programs will use to access the database.
Grant privs as appropriate, but the dbuser needs to be able to read the tables.

```
sudo -i -u postgres
psql

CREATE USER dbuser WITH PASSWORD 'dbpassword';
CREATE DATABASE embeddings_db OWNER dbuser;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO dbuser;

\q
```

#### Exit out to sudo user

```
exit
sudo apt install postgresql-server-dev-all
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

#### Launch psql again

```
sudo -u postgres psql -d embeddings_db
CREATE EXTENSION vector;
\q
```

#### give linux user permissions... just an example, optional

```
sudo -u postgres psql
ALTER USER username WITH CREATEDB;
```

#### Add Senzing schema

```
psql -U <user> -d <database> -h <server> -W
\i <senzing_project_path>/resources/schema/szcore-schema-postgresql-create.sql
```
---

#### ANN method selection

- Approximate Nearest Neighbor (ANN) method depends on performance characteristics expected and maching configuration.
  - **IVFFLAT**: Inverted Flat File
  - **HNSW**: Hierarchical Navigable Small World Graph

##### Quick Reference Table

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

##### **IVFFLAT**: Inverted Flat File
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

##### **HNSW**: Hierarchical Navigable Small World Graph
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

##### When to Use Which?

| Scenario                                                   | Recommendation                        |
| ---------------------------------------------------------- | ------------------------------------- |
| You have frequent inserts                                  | ✅ **IVFFLAT**                         |
| You want max accuracy, static data                         | ✅ **HNSW**                            |
| You care about low latency + okay with approximate results | ✅ **IVFFLAT**, well-tuned             |
| You're running batch jobs or infrequent updates            | ✅ **HNSW**                            |
| You're just getting started                                | ✅ Use IVFFLAT first — simpler to tune |

##### Combo Strategy?

Some teams:

* Use **IVFFLAT for dev / iterative testing**
* Switch to **HNSW for production search** once the data stabilizes

Bottom line, do your research and tune them appropriately

---

#### Create tables and indexes for embeddings:

- Note: "SEMANTIC_VALUE" is the table that is already defined in the g2config.json.
it can be named how ever one likes, but you must update the g2config.json file to match.

- Use the SEMANTIC_VALUE table that is already configured for BizName embeddings:

```
CREATE TABLE SEMANTIC_VALUE (LIB_FEAT_ID BIGINT NOT NULL, EMBEDDING VECTOR(512), PRIMARY KEY(LIB_FEAT_ID));
CREATE INDEX ON SEMANTIC_VALUE USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 100);
SET hnsw.ef_search = 100;
 --or--
CREATE INDEX ON SEMANTIC_VALUE USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
SET ivfflat.probes = 10;
```

- Creating a new table and configuration for Peronal name embeddings:

```
CREATE TABLE NAME_EMBEDDING (LIB_FEAT_ID BIGINT NOT NULL, EMBEDDING VECTOR(512), PRIMARY KEY(LIB_FEAT_ID));
CREATE INDEX ON NAME_EMBEDDING USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 100);
SET hnsw.ef_search = 100;
 --or--
CREATE INDEX ON NAME_EMBEDDING USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
SET ivfflat.probes = 10;
```

#### Senzing config for embeddings

- license string with embedding feature enabled.

- *g2config.json* IF not using the pre-defined table for embedding, which is "SEMANTIC_VALUE"

- create a copy of the g2config.json file in ./resources/templates.
- Copy items with FTYPE_CODE = SEMANTIC_VALUE, create new attr or ftype id, ftype_code to match table name

CFG_ATTR: the ATTR_CODE maps directly to the JSON data file attribute name.

```
...
  "CFG_ATTR":[
    ...
    {
        "ATTR_ID": 2817,
        "ATTR_CODE": "NAME_EMBEDDING",
        "ATTR_CLASS": "IDENTIFIER",
        "FTYPE_CODE": "NAME_EMBEDDINGS",
        "FELEM_CODE": "EMBEDDING",
        "FELEM_REQ": "No",
        "DEFAULT_VALUE": null,
        "INTERNAL": "No"
    },
    {
        "ATTR_ID": 2818,
        "ATTR_CODE": "SEMANTIC_ALGORITHM",
        "ATTR_CLASS": "IDENTIFIER",
        "FTYPE_CODE": "NAME_EMBEDDINGS",
        "FELEM_CODE": "ALGORITHM",
        "FELEM_REQ": "No",
        "DEFAULT_VALUE": null,
        "INTERNAL": "Yes"
    }
  ]
...
  "CFG_CFBOM":[
    ...
    {
        "CFCALL_ID": 68,
        "FTYPE_ID": 100,
        "FELEM_ID": 127,
        "EXEC_ORDER": 1
    },
    {
        "CFCALL_ID": 68,
        "FTYPE_ID": 100,
        "FELEM_ID": 128,
        "EXEC_ORDER": 2
    }
  ],
  ...
  "CFG_CFCALL":[
    ...
    {
        "CFCALL_ID": 68,
        "FTYPE_ID": 100,
        "CFUNC_ID": 15
    }
  ],
  ...
  "CFG_FBOM":[
    ...
    {
        "FTYPE_ID": 100,
        "FELEM_ID": 127,
        "EXEC_ORDER": 1,
        "DISPLAY_LEVEL": 1,
        "DISPLAY_DELIM": null,
        "DERIVED": "No"
    },
    {
        "FTYPE_ID": 100,
        "FELEM_ID": 128,
        "EXEC_ORDER": 2,
        "DISPLAY_LEVEL": 1,
        "DISPLAY_DELIM": null,
        "DERIVED": "No"
    }
  ],
  ...
  "CFG_FTYPE":[
    ...
    {
        "FTYPE_ID": 99,
        "FTYPE_CODE": "SEMANTIC_VALUE",
        "FTYPE_DESC": "Semantic value",
        "FCLASS_ID": 7,
        "FTYPE_FREQ": "FF",
        "FTYPE_EXCL": "No",
        "FTYPE_STAB": "No",
        "PERSIST_HISTORY": "Yes",
        "USED_FOR_CAND": "Yes",
        "DERIVED": "No",
        "RTYPE_ID": 0,
        "ANONYMIZE": "No",
        "VERSION": 1,
        "SHOW_IN_MATCH_KEY": "Yes"
    },
    {
        "FTYPE_ID": 100,
        "FTYPE_CODE": "NAME_EMBEDDINGS",
        "FTYPE_DESC": "Peronal name embeddings",
        "FCLASS_ID": 7,
        "FTYPE_FREQ": "FF",
        "FTYPE_EXCL": "No",
        "FTYPE_STAB": "No",
        "PERSIST_HISTORY": "Yes",
        "USED_FOR_CAND": "Yes",
        "DERIVED": "No",
        "RTYPE_ID": 0,
        "ANONYMIZE": "No",
        "VERSION": 1,
        "SHOW_IN_MATCH_KEY": "Yes"
    }
  ]
...
```

- Note that `"USED_FOR_CAND": "Yes",` was changed from No to Yes for SEMANTIC_VALUE too.

- run

```
./bin/sz_configtool

importFromFile <filename>
save
quit
```

#### Data for Senzing

The test this is prepared for is on the Open Sanctions data. In this data there are two
"RECORD_TYPES": ORGANIZATION and PERSON. Using different models to create each
embedding Senzing has been configured with two tables and separate attribues to
capture the data and embeddings.

As such, in Senzing JSON, there are fields based on our configuration above.

- SEMANTIC_EMBEDDING: used for Business name embeddings
- NAME_EMBEDDING: used for Personal name embeddings

```
{
  "DATA_SOURCE": "TEST",
  "RECORD_ID": "1A",
  "RECORD_TYPE":"PERSON",
  "NAME_FULL": "Jane Smith",
  "PHONE_NUMBER": "+15551212",
  "NAME_EMBEDDING": "[-0.021743419,...]"
}
```

if multiple:
```
{
  "DATA_SOURCE": "TEST",
  "RECORD_ID": "1A",
  "RECORD_TYPE":"PERSON",
  "NAMES":[{"NAME_TYPE":"PRIMARY","NAME_FULL": "Jane Smith"},{{"NAME_TYPE":"ALIAS","NAME_FULL": "Jannie Smith"}}]
  "PHONE_NUMBER": "+15551212",
  "NAME_EMBEDDINGS": [{"NAME_EMBEDDING": "[-0.021743419,...]"}, {"NAME_EMBEDDING": "[0.521743123,...]"}, ...]
}
```

```
{
  "DATA_SOURCE": "TEST",
  "RECORD_ID": "1A",
  "RECORD_TYPE":"ORGANIZATION",
  "NAME_FULL": "Jane Smith",
  "PHONE_NUMBER": "+15551212",
  "SEMANTIC_EMBEDDING": "[-0.021743419,...]"
}
```

if multiple:
```
{
  "DATA_SOURCE": "TEST",
  "RECORD_ID": "1A",
  "RECORD_TYPE":"ORGANIZATION",
  "NAMES":[{"NAME_TYPE":"PRIMARY","NAME_FULL": "Jane Smith"},{{"NAME_TYPE":"ALIAS","NAME_FULL": "Jannie Smith"}}]
  "PHONE_NUMBER": "+15551212",
  "SEMANTIC_EMBEDDINGS": [{"SEMANTIC_EMBEDDING": "[-0.021743419,...]"}, {"SEMANTIC_EMBEDDING": "[0.521743123,...]"}, ...]
}
```



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

