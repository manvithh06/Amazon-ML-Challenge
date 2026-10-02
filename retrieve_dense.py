#!/usr/bin/env python3
"""
Layer 1b - exact dense retrieval on the GPU (semantic blocker)

For every S1 query, finds the top-k most similar S2 records and top-k most similar S3 records
INSIDE THE SAME COUNTRY (country is an open set: whatever labels the query file has are used;
labels missing from the database simply yield no candidates).  Exact search = brute-force
blocked matrix products in fp16 - no FAISS needed at this size.

Output parquet columns (ids are the numeric part of e.g. "S2-192345572"):
    q  int64 | src int8 (2 or 3) | c int64 | <tag>_score float32 (cosine) | <tag>_rank int16
    (<tag> defaults to "dense"; use --tag name / --tag addr for the separate name / address vectors)

    # train (query sample vs FULL train S2/S3)
    python retrieve_dense.py --queries work/emb/train_s1 --db2 work/emb/train_s2 \
        --db3 work/emb/train_s3 --out work/cand/train_dense.parquet --k 20
    # test
    python retrieve_dense.py --queries work/emb/test_s1 --db2 work/emb/test_s2 \
        --db3 work/emb/test_s3 --out work/cand/test_dense.parquet --k 20
    # verify the GPU code path against numpy in a few seconds
    python retrieve_dense.py --selftest
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd


def num_ids(series):
    return series.str.slice(3).astype("int64").to_numpy()


def topk_search(q, db, k, q_chunk=1024, db_block=400_000, device="cuda"):
    """q (Nq,D) and db (Nd,D) float16 numpy, rows L2-normalised.
    Returns (idx int32 (Nq,ke), score float32 (Nq,ke)) sorted by descending similarity."""
    import torch
    dt = torch.float16 if device == "cuda" else torch.float32
    nq, nd = len(q), len(db)
    ke = min(k, nd)
    dbt = torch.from_numpy(np.ascontiguousarray(db)).to(device=device, dtype=dt)
    out_i = np.empty((nq, ke), np.int32)
    out_v = np.empty((nq, ke), np.float32)
    for s in range(0, nq, q_chunk):
        qt = torch.from_numpy(np.ascontiguousarray(q[s:s + q_chunk])).to(device=device, dtype=dt)
        bv = bi = None
        for b in range(0, nd, db_block):
            sc = qt @ dbt[b:b + db_block].T
            v, i = sc.topk(min(ke, sc.shape[1]), dim=1)
            i = i + b
            if bv is None:
                bv, bi = v, i
            else:
                v = torch.cat([bv, v], 1)
                i = torch.cat([bi, i], 1)
                bv, sel = v.topk(ke, dim=1)
                bi = i.gather(1, sel)
        out_i[s:s + q_chunk] = bi.cpu().numpy()
        out_v[s:s + q_chunk] = bv.float().cpu().numpy()
    del dbt
    if device == "cuda":
        torch.cuda.empty_cache()
    return out_i, out_v


def run(a, search=topk_search):
    mq = pd.read_parquet(os.path.join(a.queries, "meta.parquet"))
    eq = np.load(os.path.join(a.queries, "emb.npy"), mmap_mode="r")
    q_ids = num_ids(mq.entity_id)
    frames, t0 = [], time.time()
    for src, path in ((2, a.db2), (3, a.db3)):
        md = pd.read_parquet(os.path.join(path, "meta.parquet"))
        ed = np.load(os.path.join(path, "emb.npy"), mmap_mode="r")
        d_ids = num_ids(md.entity_id)
        for c in sorted(mq.country.unique()):
            qi = np.flatnonzero(mq.country.values == c)
            di = np.flatnonzero(md.country.values == c)
            if a.max_queries:
                qi = qi[:a.max_queries]
            if len(di) == 0:
                print(f"[S{src}] country '{c}': {len(qi):,} queries but NO database rows - skipped")
                continue
            t1 = time.time()
            idx, val = search(np.asarray(eq[qi]), np.asarray(ed[di]),
                              a.k, a.q_chunk, a.db_block, a.device)
            ke = idx.shape[1]
            df = pd.DataFrame({
                "q": np.repeat(q_ids[qi], ke),
                "src": np.int8(src),
                "c": d_ids[di][idx.ravel()],
                f"{a.tag}_score": val.ravel(),
                f"{a.tag}_rank": np.tile(np.arange(1, ke + 1, dtype=np.int16), len(qi)),
            })
            df = df[df[f"{a.tag}_score"] >= a.min_score]      # also drops zero-vector (empty text) rows
            frames.append(df)
            print(f"[S{src}] {c:>10s}: {len(qi):>9,} queries x {len(di):>9,} db rows "
                  f"in {time.time() - t1:6.1f}s  (median top1 {np.median(val[:, 0]):.3f})", flush=True)
    out = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out.to_parquet(a.out, index=False)
    print(f"wrote {len(out):,} candidate rows for {out.q.nunique():,} queries in "
          f"{(time.time() - t0) / 60:.1f} min -> {a.out}")


def selftest(device):
    rng = np.random.default_rng(0)
    db = rng.normal(size=(20_000, 384)).astype(np.float32)
    db /= np.linalg.norm(db, axis=1, keepdims=True)
    q = db[rng.choice(20_000, 700, replace=False)] + 0.3 * rng.normal(size=(700, 384)).astype(np.float32) / 20
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q16, db16 = q.astype(np.float16), db.astype(np.float16)
    idx, val = topk_search(q16, db16, 10, q_chunk=256, db_block=6_000, device=device)   # 4 blocks
    ref = q16.astype(np.float32) @ db16.astype(np.float32).T
    ref_i = np.argsort(-ref, axis=1)[:, :10]
    ref_v = np.take_along_axis(ref, ref_i, 1)
    overlap = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(idx, ref_i)])
    err = np.abs(val - ref_v).max()
    print(f"device={device}  top-10 overlap with numpy: {overlap:.3f}  max score error: {err:.4f}")
    ok = overlap > 0.97 and err < 5e-3
    print("SELFTEST", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries")
    ap.add_argument("--db2")
    ap.add_argument("--db3")
    ap.add_argument("--out")
    ap.add_argument("--k", type=int, default=20, help="candidates per query per source")
    ap.add_argument("--min-score", type=float, default=0.05)
    ap.add_argument("--tag", default="dense", help="column prefix: <tag>_score / <tag>_rank")
    ap.add_argument("--q-chunk", type=int, default=1024)
    ap.add_argument("--db-block", type=int, default=400_000)
    ap.add_argument("--max-queries", type=int, default=0, help="debug: queries per country")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(a.device)
    if not (a.queries and a.db2 and a.db3 and a.out):
        ap.error("--queries --db2 --db3 --out are required")
    run(a)


if __name__ == "__main__":
    main()
