# Business Entity Resolution — Amazon ML Challenge 2026

Find every record of the same business across three messy data sources.

For each business in **Source 1 (S1)**, the task is to find all matching records in **Source 2 (S2)** and **Source 3 (S3)**.
The same business can look very different from source to source:

| S1 | Matching record in S2 / S3 |
|---|---|
| `Potts Desert Kochav LLC, 1819 G Street, Sacramento, CA` | `POTTSDESERT, 819 G ST, SACRAMENTO, CA` |
| `Keystone Inc, 9765 272, Bertram, TX` | `keyst0ne inc, 9765 272, bertram city, texas` |
| `Southern Ventures Limited, No. 17, Kamaraja Colony …, Chennai` | `சதர்ன் வென்சர்ஸ் லிமிடெட், NO. 17, CHENNAI` |

The kinds of noise include ALL-CAPS text, typos, website-style names (`wilfordhancock.com`), legal-form variants (`L.L.C.`, `Pvt Ltd`), truncated addresses and names in Tamil or Hindi script.

---

## Results

| | |
|---|---|
| **Validation macro F0.5** (official metric) | **0.9811** |
| **Leaderboard progression** | 0.942 → 0.944 → 0.958 → 0.961 → v9 (final) |
| **Blocking recall** (true matches present in the candidate list) | **0.980** |
| **Candidates per S1 entity** | 7.57 |
| **Largest model** | multilingual MiniLM-L12, 118M parameters, Apache-2.0 |
| **External data / APIs** | none |
| **Hardware** | one laptop: 8-core CPU, 22 GB RAM, one consumer GPU |

---

## How it works

The pipeline first finds a small set of plausible candidates with very high recall. It then judges each candidate with progressively stronger models, and finally makes a decision tuned for the official metric.

```
 S1 / S2 / S3 records
        │
 ① Normalise          clean text, transliterate, extract legal form, numbers, postal code
        │
 ② Find candidates    (always inside the same country)
        │   • name embeddings          top-5 per source
        │   • address embeddings       top-5 per source
        │   • lexical keyword index    top-20
        │   • fine-tuned retriever     top-3 per source   ← biggest recall gain
        │
 ③ Pair features      57 similarity features per candidate pair
        │
 ④ Prune              LightGBM keeps the best ~5 of ~68 candidates per S1
        │
 ⑤ Re-score           two fine-tuned cross-encoders read both records together
        │
 ⑥ Decide             LightGBM stacker → threshold tuned on macro F0.5
        │              → one-to-one: each S2/S3 record goes to its best S1 only
        ▼
 matching_results.tsv  +  candidate_pairs.tsv
```

### ① Normalise (`layer0.py`)
Each string is kept in two forms:

- **Romanised (`anyascii`):** used for string matching.
- **Original script:** used for the embeddings.

Legal forms (LLC, Inc, Pvt Ltd …), street numbers, postal codes, landmarks and "doing business as" names are pulled out into separate fields.

### ② Find candidates: blocking
Comparing every S1 record with every S2/S3 record would mean about 10¹³ pairs, so we only compare plausible ones. Matches never cross countries, so every search runs within one country.

| Blocker | File | What it catches |
|---|---|---|
| Name embeddings | `embed.py`, `retrieve_dense.py` | spelling variants, transliterations |
| Address embeddings | same | renamed businesses at the same address |
| Lexical IDF index | `retrieve_lexical.py` | exact numbers, postal codes, website-style names |
| **Fine-tuned retriever** | `retr.py` | the hard cases the others miss |

The **fine-tuned retriever** is the same MiniLM model trained on 800k known matches from the training data, so it learns what "the same business" means in this data. It raised blocking recall from **0.943 to 0.980**.

### ③ Pair features (`build_pairs.py`)
Each candidate pair gets 57 features:

- **Name and address similarity** (RapidFuzz): ratio, token-set, partial ratio, Jaro-Winkler.
- **Embedding cosines.**
- **Agreement checks:** street number, postal code and legal form.
- **Context:** how this candidate compares with the other candidates for the same S1.

### ④ Prune and match (`match.py`)
A small LightGBM **pruner** cuts about 68 candidates down to about 5 per S1, at almost no loss. A LightGBM **matcher** gives each remaining pair a first probability.

### ⑤ Cross-encoders (`ce.py`)
The MiniLM model is fine-tuned to read both records at once:
`S1 name | address [SEP] candidate name | address` → match probability.
We train two of them, with different data sizes and seeds, and use them as an ensemble. This was the **largest single gain: +0.016 F0.5**.

### ⑥ Decide (`retr.py stack`)
A LightGBM **stacker** combines all the scores and gives the final probability. A pair is kept if its probability is at least τ, where τ = 0.80 was chosen to maximise the official macro F0.5 on held-out data. Finally, each S2/S3 record is assigned only to the S1 entity that scores it highest.

---

## How we measured progress

- **Held-out validation:** 10% of a 264,818-query training sample, never used to train any model.
- **Realistic conditions:** queries are searched against the **full** S2/S3 training files, just like the test set.
- **Same metric as the leaderboard:** every decision uses the official macro F0.5, counted against the full ground truth.
- **Reliable:** validation matched the leaderboard within about 0.001 on every submission.

| Version | What changed | Valid F0.5 | Leaderboard |
|---|---|---|---|
| v1 | 3 blockers + 57 features + LightGBM | 0.9487 | 0.942 |
| v4 | wider blocking + candidate pruner | 0.9511 | 0.944 |
| v6 | + cross-encoder #1 + stacker | 0.9650 | 0.958 |
| v8 | + cross-encoder #2 (ensemble) | 0.9675 | 0.961 |
| **v9** | **+ fine-tuned retriever** | **0.9811** | **final** |

**Tried and dropped:** expanding candidates from confident matches (+0.002 but 2.3× more candidates), a looser pruner (+0.0005), and simply widening the off-the-shelf blocking (recall stuck at 0.958).

---

## Getting started

### 1. Install
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
You need a CUDA GPU. The base model `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` downloads automatically the first time it's used.

### 2. Put the data in place
```
student_resource/
└── dataset/
    ├── train/  train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
    └── test/   test_source1.tsv   test_source2.tsv   test_source3.tsv
```

### 3. Run everything
```bash
bash run_pipeline.sh
```
This writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`, then runs the official validator.
The full run takes about 14 hours on one laptop GPU. Run it from a plain terminal with `nohup` so it survives a closed window:
```bash
nohup bash run_pipeline.sh > pipeline.log 2>&1 &
tail -f pipeline.log
```

---

## Step-by-step commands

`run_pipeline.sh` runs exactly these steps. `T` is the path to the training ground truth.

```bash
T=student_resource/dataset/train/train_ground_truth.tsv
```

**① Normalise**
```bash
for sp in train test; do for i in 1 2 3; do
  python layer0.py --in student_resource/dataset/$sp/${sp}_source$i.tsv --out work/l0/${sp}_s$i --workers 6
done; done
```

**② Candidates: embeddings + lexical index**
```bash
# name and address vectors (train S1 = 12% query sample)
python embed.py --l0 work/l0/train_s1 --out work/emb_name/train_s1 --field name --frac 0.12
python embed.py --l0 work/l0/train_s2 --out work/emb_name/train_s2 --field name
#   … repeat for train_s3, test_s1/2/3, and for --field addr (folder emb_addr)

python retrieve_dense.py --queries work/emb_name/train_s1 --db2 work/emb_name/train_s2 \
    --db3 work/emb_name/train_s3 --out work/cand/train_name.parquet --k 20 --tag name
#   … same for addr, and for test

python retrieve_lexical.py --q work/l0/train_s1 --db2 work/l0/train_s2 --db3 work/l0/train_s3 \
    --ids-from work/emb_name/train_s1 --out work/cand/train_lex.parquet --k 20
python retrieve_lexical.py --q work/l0/test_s1 --db2 work/l0/test_s2 --db3 work/l0/test_s3 \
    --out work/cand/test_lex.parquet --k 20 --chunks 8
```

**③ Pair features**
```bash
python build_pairs.py --split train --truth $T --k-name 5 --k-addr 5 --k-lex 20 --chunks 20 --mem 4GB
python build_pairs.py --split test           --k-name 5 --k-addr 5 --k-lex 20 --chunks 130 --mem 6GB
```

**④ Matcher + pruner**
```bash
python match.py train --truth $T --fit-frac 0.6
python match.py prune-train --truth $T --rich --max-loss 0.0005
python match.py predict --prune --out-dir output_v4
```

**⑤ Cross-encoders**
```bash
python ce.py data
python ce.py train --max-train 800000                                    # cross-encoder #1
python ce.py score --split valid && python ce.py score --split test
python ce.py train --max-train 0 --epochs 2 --seed 1 --model-dir model2  # cross-encoder #2
python ce.py score --split valid --model-dir model2 --col ce2
python ce.py score --split test  --model-dir model2 --col ce2
```

**②+⑥ Fine-tuned retriever, final stacker, submission**
```bash
python retr.py train --n 800000
python retr.py embed --split train
python retr.py embed --split test
for sp in train test; do
  python retrieve_dense.py --queries work/emb_ft/${sp}_s1 --db2 work/emb_ft/${sp}_s2 \
      --db3 work/emb_ft/${sp}_s3 --out work/cand/${sp}_ft.parquet --k 3 --tag ft
done
python retr.py newpairs --split valid --truth $T
python retr.py score --split valid --only-ce2
python retr.py stack --truth $T
python retr.py newpairs --split test
python retr.py score --split test --only-ce2
python retr.py predict --out-dir output
```

**Validate the submission**
```bash
python student_resource/utils/validate_submission.py \
    -m output/matching_results.tsv -c output/candidate_pairs.tsv -t student_resource/dataset/test
```

---

## Project structure

```
├── layer0.py               ① normalisation
├── embed.py                ② sentence embeddings (name / address)
├── retrieve_dense.py       ② exact GPU top-k search, per country
├── retrieve_lexical.py     ② keyword (IDF) blocker in DuckDB
├── build_pairs.py          ③ candidate union + 57 pair features
├── match.py                ④ LightGBM matcher, pruner, F0.5 decision rule
├── ce.py                   ⑤ cross-encoder training / scoring
├── retr.py                 ②⑥ fine-tuned retriever, final stacker, submission writer
├── expand.py               experiment (cluster expansion), its writer is reused
├── eval_blocking.py        diagnostic: blocking recall
├── inspect_misses.py       diagnostic: why true matches are missed
├── diagnose_transitive.py  diagnostic: cluster structure of misses
├── check_structure.py      diagnostic: checks IDs / row order carry no signal
├── run_pipeline.sh         full pipeline, end to end
├── requirements.txt
└── README.md
```

Generated folders (`work/`, `output*/`, `submissions/`) are not in git; see `.gitignore`.

---

## Rules compliance

- **Models:** only open-source models ≤ 8B parameters. MiniLM-L12 (118M, Apache-2.0); LightGBM, DuckDB and RapidFuzz (MIT); PyTorch and Transformers.
- **Data:** no external data and no external APIs.
- **No leakage:** we checked that IDs and file order carry no information (all correlations ≈ 0), and they are not used anywhere.

## Engineering notes

- Large joins (117M candidate pairs) run in **DuckDB with an on-disk database**, so they never run out of RAM.
- Work is split into **130 chunks by S1 id**, so per-query features stay exact and long GPU jobs can **resume** after a crash.
- All transformer training and inference uses **fp16 and length-bucketed batches**.

## Possible next steps

- Train the retriever with **hard negatives** and more epochs. About 2% of true matches are still never retrieved.
- Score the new retriever candidates with **both** cross-encoders (only one was used, to save time).
