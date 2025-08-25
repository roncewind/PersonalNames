# PersonalNames
Tinkerspace for personal name clustering


Purpose of the experiment is to see if we can created embeddings based on
personal names such that a personal name in any language is similar, using
cosine similarity, to other names for the same person.

## Find biznames in Wikidata extract:

`extract_wikidata_names.py` reads from a wikidata extract and creates a CSV file of personal names. Particularly translated versions.

## Analyze personal names from Wikidata (optional, playground)

`analyze_wikidata_extract.py` analyzes the CSV file created from a wikidata extract and reports some stats.

## Split training data into triplets files for training, validation and testing

`split_training_data.py` creates personal name triplets and splits data in to training, validation, and test sets in JSONL files.

## Train a model using data created

`train_sbert_contrastive.py` fine-tune a sentence transformer model for out specific use case.

## Test trained model

`test_triplets.py` runs the test set of triplets in it's JSONL file against a fine-tuned model and reports with various stats and graphs.

## Create hard negative training set

`mine_hard_negatives.py` mines the training set using a fine-tuned model to create a "hard negative" training set in a JSONL file. This further fine-tunes the model. One should set the margin parameter to less that the what the training loop used.

## Hard negative sanity test

`hard_negative_sanity.py` analyzes the hard negative training set and reports various statistics.

## Re-train with hard negatives

`train_sbert_contrastive.py` same as the training loop, but give it the additional `--hard_path` parameter pointing to the hard negative training set JSONL file.

## Compare a couple of names using a model:

`compare_names.py` simple example on how one might use the model to compare two names against each other.


## Data source, possibilities:

- sigpwned - names by country, with romanization
    - https://github.com/sigpwned/popular-names-by-country-dataset
    - CC0-1.0

- m7medVision - list of names by country
    - https://github.com/m7medVision/top-names
    - unknown license

- ParaNames - NOT USEFUL
    - https://github.com/bltlab/paranames
    - http://arxiv.org/abs/2202.14035
    - MIT? CC BY 4.0?
    - Just code and methodolgy for extracting data from wikidata
    - We can already do this

- ✅ openempi
    - https://github.com/MrCsabaToth/SOEMPI/tree/master/openempi
    - Apache 2.0
    - https://github.com/MrCsabaToth/SOEMPI/blob/master/openempi/conf/name_to_nick.csv
    - https://github.com/MrCsabaToth/SOEMPI/blob/master/openempi/conf/nick_to_name.csv

- ✅ onyxrev
    - https://github.com/onyxrev/common_nickname_csv
    - Public domain

- ✅ carltonnorthern
    - https://github.com/carltonnorthern/nicknames
    - Apache 2.0

- ✅ Deron Meranda
    - https://web.archive.org/web/20181022154748/https://deron.meranda.us/data/nicknames.txt
    - Public domain??

- brianary
    - https://github.com/brianary/Lingua-EN-Nickname/blob/main/nicknames.txt
    - Public domain??
    - Many odd nicknames ending in 'E'. EG Agatha	AddE AggE.
      Not sure why the ending 'E'... could indicate both 'ie' and 'y' variants?

- diminutives.db - NOT USABLE (viral license)
    - https://github.com/HaJongler/diminutives.db
    - https://github.com/HaJongler/diminutives.db/blob/master/female_diminutives.csv
    - https://github.com/HaJongler/diminutives.db/blob/master/male_diminutives.csv
    - GPLv3

- philipperemy Can we use this??? Is it useful?
    - https://github.com/philipperemy/name-dataset
    - Apache 2.0
    - However...
        - License
            - This version was generated from the massive Facebook Leak (533M accounts).
            - Lists of names are not copyrightable, generally speaking, but if you want to be completely sure you should talk to a lawyer. (https://www.justia.com/intellectual-property/copyright/lists-directories-and-databases/)

- useful wikidata properties:
    - nickname: https://www.wikidata.org/wiki/Property:P1449
    - pseudonym: https://www.wikidata.org/wiki/Property:P742
    - given name: https://www.wikidata.org/wiki/Q202444
    - name: https://www.wikidata.org/wiki/Q82799
    - https://w.wiki/4qa2

- Linguistic Data Consortium: american nickname database PAID
    - https://catalog.ldc.upenn.edu/LDC2012T11


## Synthetic data

### Name variants

1. Element variations
    - Data errors
        1. Optical Character Recognition errors
        1. Typos
        1. Truncations
    - Name particles
        1. Segmentation, e.g. Abd Al Rahman ~ Abdal Rahman, De Los Angeles ~ Delosangeles
        1. Omission, e.g. of bin in Arabic names or de in Hispanic names.
    - Short forms
        1. Abbreviations, e.g. Muhammad ~ Mhd
        1. Initials, e.g. John Smith ~ J Smith
    - Spelling variations
        1. Alternate spellings, e.g. Jennifer ~ Jenifer
        1. Transliteration, e.g. Husayn ~ Husein
    - Nicknames and diminutives, e.g. Robert ~ Bob
    - Translation variants, e.g. Joseph ~ Giuseppe
1. Structural variations
    - Additions/deletions, e.g. John Smith ~ John Charles Smith
    - Fielding variation: division of full name into surname and given name, or swapping given name and surname
    - Permutations, e.g. Clara Lucia Garcia ~ Lucia Clara Garcia
    - Placeholders: non-name tokens like FNU, LNU, UNK
    - Element segmentation, e.g. Mohamed Amin ~ Mohammedamin

Ref. https://www.researchgate.net/publication/220746750_A_Ground_Truth_Dataset_for_Matching_Culturally_Diverse_Romanized_Person_Names

## 1) Name variation types to consider:

### **Orthography & Unicode**

  * ✅ Diacritics dropped/added (José → Jose), å/ä/ö→a/o, ё→е (Russian), ğ/ş (Turkish).
  * ✅ **Unicode normalization** issues (NFC/NFKD), ½, zero-width joiners,  full-width vs half-width (Ｊｏｓｅ ↔ Jose).
  * 🚧(some) **Homoglyphs** across scripts (Latin “a” vs Cyrillic “а”, “e” vs “е”, “p” vs “р”, “H” vs Greek “Η”).
  * ✅ All-caps, Title Case, random case, capitalize first letter only.
  * **TODO** - not handled:
    - ß→ss, ı↔i (Turkish), ł (Polish)?  Homoglyphs????
    - curly vs straight apostrophes (O’Connor/O'Connor) and other quotes and such

### **Punctuation, spacing, hyphenation**

  * ✅ Hyphen join/split (Jean-Paul ↔ Jean Paul)
  * apostrophe drop or duplicate (D’Angelo ↔ Dangelo)
  * extra spaces, double spaces, trailing/leading spaces, mid-token spaces (Mo hammed)

### 🚧(some) **Script & transliteration quirks**

  * Multiple romanization standards (Zhāng → Zhang/Chang; Kyiv → Kiev).
  * Arabic/Persian: ta marbūṭa ة rendered as “h”, yaa’/alif maqsūra (ى/ي), hamza placement, vowel omission.
  * Japanese: kanji ↔ kana ↔ romaji; long vowels (ō→ou), name order (surname-first vs given-first).
  * Chinese: hyphenation in given names (Xi-Wei ↔ Xiwei), spacing rules.

#### Transliteration in general

  Several python packages are available all with different options and supported languages.

  Short list:

  * Python packages for transliteration
    - [PyICU](https://pypi.org/project/pyicu/)  - 671 different transliterations `icu.Transliterator.getAvailableIDs()`
      - https://unicode-org.github.io/icu/userguide/transforms/general/
      - https://gist.github.com/dpk/8325992
    - [transliterate](https://pypi.org/project/transliterate/) - Armenian, Bulgarian (beta), Georgian, Greek, Macedonian (alpha), Mongolian (alpha), Russian, Serbian (alpha), Ukrainian (beta)
    - [iuliia](https://pypi.org/project/iuliia/) - 20 Russian transliteration schemas (all major international and national standards). Official Uzbek transliteration schema.
    - [cyrtranslit](https://pypi.org/project/cyrtranslit/) - Bulgarian, Montenegrin, Macedonian, Mongolian, Russian, Serbian, Tajik, and Ukrainian.
    - [iso9](https://github.com/cjolowicz/iso9) - Abkhaz, Altay, Belarusian, Bulgarian, Buryat, Chuvash, Karachay-Balkar, Macedonian, Moldavian, Mongolian, Russian, Rusyn, Serbian, Udmurt, Ukrainian, and all Caucasian languages using páločka.
    - [arabic_buckwalter_transliteration](https://github.com/hayderkharrufa/arabic-buckwalter-transliteration) - arabic
    - [gimeltra](https://github.com/twardoch/gimeltra) - 20 scripts, specializing in those of Semitic origin, offering a simplified, abjad-only transliteration.
    - [pypinyin](https://pypi.org/project/pypinyin/) - Chinese. Does "lazy pinyin" with no diacritics.  `from pypinyin import lazy_pinyin`
    - [cjklib](https://pypi.org/project/cjklib3/) - For Wade-Giles romanization.
    - [ai4bharat-transliteration](https://pypi.org/project/ai4bharat-transliteration/) - 21 major languages of the Indian subcontinent.

  * Unfavorable license
    - [transliter](https://github.com/elibooklover/Transliter) - Korean, Japanese, Russian, Ukrainian, Bulgarian, Macedonian, Mongolian, Montenegrin, Serbian, and Tajiki. GPL 2.0 !!!!
    - [pykakasi](https://codeberg.org/miurahr/pykakasi) - Japanese in all it's flavors, GPL 3.0!!!?
    - [unidecode](https://pypi.org/project/Unidecode/) - not language specific, strict unicode to ascii "romanitization"  GPL 2.0 !!!!

  * Python packages not specific to transliteration, but to writing systems:
    - [alphabetic](https://github.com/Halvani/alphabetic) - retrieving script types of writing systems including alphabets, abjads, abugidas, syllabaries, logographs, featurals as well as Latin script codes. Can be used to classify the language of a given text
    - [GlotScript](https://github.com/cisnlp/GlotScript) - determines the script (writing system) of input text using ISO 15924.

### **Cultural structure**

  * **Mononyms** (Indonesia, parts of India): “Sukarno”.
  * **Patronymics/matronymics**: Russian (-ovich/-ovna), Icelandic (Jónsdóttir), Arabic name chains (kunya Abu…, nasab bin/ibn, nisba al-).
  * 🚧(some) **Particles and nobiliary markers**: van/de/da/di/del/della/du, O’/Mc/Mac; case sensitivity (van Gogh vs Van Gogh).
  * **Iberian double surnames** (paternal + maternal), Brazilian order differences, optional “y/de/del”.

### **Titles & suffixes**

  * Mr./Mrs./Dr./Haji/Hajjah, Esq., Jr./Sr./III, academic/royal/religious honorifics interleaved anywhere.

### **Data-entry / system artifacts**

  * 🚧(QWERY only) **Keyboard adjacency** errors (mobile & desktop layouts), swapped neighboring letters, repeated letters.
  * ✅ **Truncation** at fixed field widths, mid-token clipping, ellipsis, dropped suffixes.
  * 🚧(some) **Field contamination** (name field contains extra tokens like “N/A”, “—”, emails, dates),
    **placeholders** beyond FNU/LNU/UNK (TBD, ???, NONE).

### **Phonetic drift**

  * th↔t/d (Sathya→Satya), v↔w (Venkata↔Wenkata), y/j/i (Yosef/Josef/Iosef), ch/š/ś → various Latinizations, ll↔y (Spanish), dh/ḥ/kh Arabic clusters.

### **Alias classes**

  * **Stage/clerical names** (e.g., “Pope Francis” vs birth name), courtesy names, pen names.
  * **Register-driven initials** (South India: “R. Karthik” where “R” is father’s name), institutional short forms (“Md” for Muhammad, “Ma.” for María in the Philippines).

## 2) Options for generating realistic synthetic data

### A) Rule-based + probabilistic “error channels”

* 🚧 library of transforms started, need to work on per-language weights.
* Using per-locale profiles so, e.g., Turkish dotted-i or Spanish double surnames trigger only where appropriate.
* How many transforms per name to keep noise realistic? (0-3)?
* ✅ **Diacritic folding / casefolding / width folding**
* 🚧(QWERY only) **Keyboard adjacency substitutions** (build layouts for QWERTY, AZERTY, QWERTZ, mobile)
* 🚧(some)**Particle rules** (drop/join/re-case de/da/van/al/bin, split/merge)
* ✅ **Hyphen/apostrophe/space join-split**
* ✅ **Token reorderings** (GN-SN ↔ SN-GN, permutations within 2–3 tokens)
* ✅ **Truncation** policies (head/tail/mid with sensible min lengths)
* **Script transliteration cycle** (Latin→Cyrillic→Latin, Arabic↔Latin with multiple schemes)
* 🚧(mostly English) **Nicknames & diminutives** via curated maps (Robert→Bob/Rob; Joseph→Giuseppe/José/Yosef/Youssef)
* **Placeholders/titles** insertion/removal


### B) Phonetic/transliteration-driven variants (effort?, very useful)

* Convert to **phonetic codes** (Double Metaphone, NYSIIS, Daitch–Mokotoff) then regenerate plausible spellings from the code (or pick from a lookup). For same-sound/different-spelling negatives and positives.
* **G2P → perturb phonemes → P2G** to induce sound-preserving misspellings.
* **Transliteration cycling** with multiple standards to create realistic cross-script drift.

### C) Dictionary-driven alias expansion (medium effort, to source nicknames in other languages)

* 🚧(mostly English) Curate nickname/diminutive/translation tables per language.
* Tag each mapping with confidence to avoid aggressive expansion?
* Add culture-specific patterns (Arabic kunya, Russian patronymic expansion, South Indian initial expansion).

### D) Learned corruption models (large effort?, seems flexible)

Need to look into how to use ML to corrupt data realistically.

* **Seq2seq “noiser”**: train a small transformer to map clean name → corrupted name using your rule-based outputs as supervision; then sample from it. This gives you diversity without hand-tuning every rule.
* **Denoising autoencoder**: train to reconstruct clean from corrupted; during generation, run clean → latent → decode with noise to sample corruptions.
* **Masked-LM** tuned on names: randomly mask characters/subtokens and let it predict plausible replacements (constrained to same script).

### E) Rendering-based OCR corruption (big effort to generate data)

Render names with varied fonts/blur/noise, run OCR (e.g. Tesseract/pytesseract ) to capture truly realistic OCR mistakes?

  *  🚧(some) **OCR confusions**: rn↔m, l↔1, O↔0, B↔8, cl↔d, I↔l, c↔e, t↔f, é→e, diacritic loss, “.”→“,”, random insertions.


### F) Online augmentation during triplet mining (training-aware)

When forming triplets:

* Generate **positives** by applying light, label-preserving transforms to the anchor (diacritic folding, particle join, nickname, transliteration cycle).
* Generate **semi-hard negatives** by:
  * same surname, nearby given name (Levenshtein 1–2),
  * same initials, different given/surname,
  * phonetic collision but different dictionary identity,
  * transliteration collisions.
* Periodically **re-mine hard negatives** from the current embedding space with constraints (must not be same entity or alias cluster).

---

# TODO:

* Tag each synthetic example with the applied transform list; you can later weight or filter during training.
* Keep the **noise rate modest** (e.g., ≤30–40% of examples augmented, 0–2 transforms each). Too much noise can collapse the embedding geometry.
* **Language-aware profiles**: choose transform sets and weights by script/locale.
* Build **gold “same-person” clusters** from Wikidata IDs/aliases; generate positives within cluster, negatives across clusters but near by string/phonetics.
* Consider **Supervised Contrastive Loss** in addition to triplet loss for stability with many positives per anchor.
* For hard negatives, add a small **margin schedule** and periodically re-mine neighbors to avoid overfitting to stale hard negatives.


---
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

