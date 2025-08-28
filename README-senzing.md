
# Setting up Senzing database for vector operations

Look at [Postgres README](README-postgres.md) for information about using PostgreSQL
for vector operations and some details about which ANN method to use.

## References:

- https://senzing.com/docs/quickstart_linux/
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

CREATE USER <dbuser> WITH PASSWORD '<dbpassword>';
CREATE DATABASE <database> OWNER <dbuser>;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO <dbuser>;

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
sudo -u postgres psql -d <database>
CREATE EXTENSION vector;
\q
```

#### give linux user permissions... just an example, optional

```
sudo -u postgres psql
ALTER USER username WITH CREATEDB;
```

#### make a few performance enhancements for Senzing, optional

```
ALTER DATABASE senzing SET synchronous_commit = OFF;
ALTER DATABASE senzing SET enable_seqscan TO OFF;
ALTER SYSTEM SET wal_writer_delay = '1000ms';
SELECT pg_reload_conf();
```

#### Add Senzing schema

```
psql -U <dbuser> -d <database> -h <server> -W
\i <senzing_project_path>/resources/schema/szcore-schema-postgresql-create.sql
```
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

- Creating a new table and configuration for Peronal name embeddings
- The table name will be used as the FTYPE_CODE in the Senzing configuration below.

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
        "FTYPE_CODE": "NAME_EMBEDDING",
        "FELEM_CODE": "EMBEDDING",
        "FELEM_REQ": "No",
        "DEFAULT_VALUE": null,
        "INTERNAL": "No"
    },
    {
        "ATTR_ID": 2818,
        "ATTR_CODE": "SEMANTIC_ALGORITHM",
        "ATTR_CLASS": "IDENTIFIER",
        "FTYPE_CODE": "NAME_EMBEDDING",
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
        "FTYPE_CODE": "NAME_EMBEDDING",
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

#### Import the new config into Senzing and add the datasource

- run

```
./bin/sz_configtool

importFromFile <filename>
addDataSource OPEN_SANCTIONS
listDataSources
save

```

#### Set the Tau threshold for each embedding model

- Multiply Tau by 100 and set that as the "likelyScore", distribute the rest as
needed.
- Look at similarity distribution histogram and use that to inform score decisions.

```
./bin/sz_configtool

addComparisonThreshold {"function": "SEMANTIC_SIMILARITY_COMP", "returnOrder": 1, "scoreName": "FULL_SCORE", "feature": "SEMANTIC_VALUE", "sameScore": 80, "closeScore": 50, "likelyScore": 30, "plausibleScore": 20, "unlikelyScore": 10}

addComparisonThreshold {"function": "SEMANTIC_SIMILARITY_COMP", "returnOrder": 1, "scoreName": "FULL_SCORE", "feature": "NAME_EMBEDDING", "sameScore": 80, "closeScore": 60, "likelyScore": 43, "plausibleScore": 30, "unlikelyScore": 20}

listComparisonThresholds
save
quit

```


#### Data for Senzing

The test this is prepared for is on the Open Sanctions data. In this data there are two
"RECORD_TYPES": ORGANIZATION and PERSON. Using different models to create each
embedding Senzing has been configured with two tables and separate attribues to
capture the data and embeddings.

As such, in Senzing JSON, there are fields based on our configuration above.
Note that the FTYPE_CODE should be the same for the JSON attribut as well as the table name.

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
