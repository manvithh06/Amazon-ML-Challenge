#!/usr/bin/env python3
"""
Layer 1a - multilingual sentence embeddings of Layer-0 output  (dense blocker input)

Reads a Layer-0 directory (parquet parts), embeds "name | address" in the ORIGINAL script
(name_emb / addr_emb) with paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0, 118M params),
and writes, in row order:
    <out>/emb.npy       float16 (N, 384), L2-normalised  -> cosine == dot product
    <out>/meta.parquet  entity_id, country               (same row order)
Resumable: finished parts are skipped after a crash / Ctrl-C (same arguments required).

    # 0) benchmark first (about 1 minute) and look at rows/s
    python embed.py --l0 work/l0/train_s2 --out work/emb/bench --limit 100000
    # 1) real runs
    python embed.py --l0 work/l0/train_s1 --out work/emb/train_s1 --frac 0.12   # query sample
    python embed.py --l0 work/l0/train_s2 --out work/emb/train_s2
    ...
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DIM = 384


def load_model(name, max_len, device):
    import torch
    from sentence_transformers import SentenceTransformer
    if device == "cuda":
        if not torch.cuda.is_available():
            sys.exit("PyTorch cannot see the GPU. RTX 50-series (Blackwell) needs a CUDA 12.8+ "
                     "build:  pip install torch --index-url https://download.pytorch.org/whl/cu128")
        print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}", flush=True)
    model = SentenceTransformer(name, device=device)
    model.max_seq_length = max_len
    if device == "cuda":
        model.half()
    return model


def encode(model, texts, batch):
    # encode() sorts by length internally, so padding waste stays small
    return model.encode(texts, batch_size=batch, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False)


def make_texts(df, field="both"):
    """Lower-cased on purpose: the model tokenizes ALL-CAPS text very differently, which made
    upper-case S2 records unreachable from mixed-case S1 records."""
    if field == "name":
        return [n.lower() for n in df.name_emb]
    if field == "addr":
        return [a.lower() for a in df.addr_emb]
    return [(n if not a else f"{n} | {a}").lower() for n, a in zip(df.name_emb, df.addr_emb)]


def plan_selection(counts, a):
    """Per-part row indices to embed (None = all rows), deterministic for --frac/--limit."""
    sel, left = [], a.limit
    for i, n in enumerate(counts):
        if a.limit:
            k = min(n, left)
            left -= k
            sel.append(np.arange(k))
        elif a.frac < 1.0:
            rng = np.random.default_rng(a.seed + i)
            sel.append(np.sort(rng.choice(n, int(round(n * a.frac)), replace=False)))
        else:
            sel.append(None)
    return sel


def main(argv=None, model_loader=load_model, encoder=encode):
    ap = argparse.ArgumentParser()
    ap.add_argument("--l0", required=True, help="Layer-0 output directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=1.0, help="random fraction of rows (query sample)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="benchmark: only the first N rows")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--field", choices=["both", "name", "addr"], default="both",
                    help="which text to embed (name-only / address-only vectors retrieve independently)")
    a = ap.parse_args(argv)

    parts = sorted(glob.glob(os.path.join(a.l0, "part-*.parquet")))
    if not parts:
        sys.exit(f"no part-*.parquet in {a.l0}")
    counts = [pq.ParquetFile(p).metadata.num_rows for p in parts]
    sel = plan_selection(counts, a)
    n_sel = [c if s is None else len(s) for c, s in zip(counts, sel)]
    total = int(sum(n_sel))
    offsets = np.concatenate([[0], np.cumsum(n_sel)]).astype(np.int64)
    os.makedirs(a.out, exist_ok=True)

    plan = dict(l0=os.path.abspath(a.l0), frac=a.frac, seed=a.seed, limit=a.limit,
                model=a.model, max_len=a.max_len, total=total, parts=len(parts), field=a.field)
    plan_path = os.path.join(a.out, "plan.json")
    if os.path.exists(plan_path):
        if json.load(open(plan_path)) != plan:
            sys.exit(f"{a.out} was created with different arguments; use another --out or delete it")
    else:
        json.dump(plan, open(plan_path, "w"), indent=1)

    meta_path = os.path.join(a.out, "meta.parquet")
    if not os.path.exists(meta_path):
        metas = []
        for p, s in zip(parts, sel):
            m = pq.read_table(p, columns=["entity_id", "country"]).to_pandas()
            metas.append(m if s is None else m.iloc[s])
        pd.concat(metas, ignore_index=True).to_parquet(meta_path, index=False)

    emb_path = os.path.join(a.out, "emb.npy")
    mode = "r+" if os.path.exists(emb_path) else "w+"
    emb = np.lib.format.open_memmap(emb_path, mode=mode, dtype=np.float16, shape=(total, DIM))

    todo = [i for i in range(len(parts)) if not os.path.exists(os.path.join(a.out, f".done-{i:05d}"))]
    print(f"{total:,} rows, {len(parts)} parts, {len(todo)} to do -> {a.out}", flush=True)
    if not todo:
        return
    model = model_loader(a.model, a.max_len, a.device)

    t0, done_rows = time.time(), 0
    for i in todo:
        if n_sel[i] == 0:                      # e.g. parts beyond --limit
            open(os.path.join(a.out, f".done-{i:05d}"), "w").close()
            continue
        df = pq.read_table(parts[i], columns=["name_emb", "addr_emb"]).to_pandas()
        if sel[i] is not None:
            df = df.iloc[sel[i]]
        texts = make_texts(df, a.field)
        nz = [j for j, t in enumerate(texts) if t.strip()]
        vecs = np.zeros((len(texts), DIM), np.float16)     # empty text -> zero vector (never retrieved)
        if nz:
            v = np.asarray(encoder(model, [texts[j] for j in nz], a.batch))
            assert v.shape == (len(nz), DIM), v.shape
            vecs[nz] = v.astype(np.float16)
        emb[offsets[i]:offsets[i + 1]] = vecs
        emb.flush()
        open(os.path.join(a.out, f".done-{i:05d}"), "w").close()
        done_rows += len(df)
        rate = done_rows / (time.time() - t0)
        left = sum(n_sel[j] for j in todo if j > i)
        print(f"part {i + 1}/{len(parts)}  {rate:,.0f} rows/s  ETA {left / rate / 60:,.1f} min", flush=True)

    # sanity: rows that are all zeros are either empty texts (expected) or unwritten rows (bug)
    bad = 0
    for s in range(0, total, 500_000):
        bad += int((np.abs(emb[s:s + 500_000]).sum(axis=1) == 0).sum())
    print(f"done in {(time.time() - t0) / 60:.1f} min | all-zero rows: {bad} (empty texts get zero vectors)")


if __name__ == "__main__":
    main()
