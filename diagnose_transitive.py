#!/usr/bin/env python3
"""
Diagnostic: could CLUSTER EXPANSION recover the true pairs that blocking misses?

Every S1 entity has ~3.5 true matches (several S2 + several S3 records of the same business).
For a missed record m of query q, we check how similar m is to the OTHER true records of q that
blocking DID find ("anchors").  If misses look like their anchors (near-duplicates) but not like
the S1 text, then "find one match, then pull in its near-duplicates" raises recall.

    python diagnose_transitive.py            (uses work/ defaults; ~2-4 min)
"""
import argparse
import glob

import duckdb
import numpy as np
import pandas as pd

from build_pairs import EmbLookup
from eval_blocking import load_truth


def cos_pairs(lk_a, lk_b, a, b):
    ea, eb = lk_a.gather(a), lk_b.gather(b)
    d = np.einsum("ij,ij->i", ea, eb)
    d[(np.abs(ea).sum(1) == 0) | (np.abs(eb).sum(1) == 0)] = np.nan
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--truth", default="student_resource/dataset/train/train_ground_truth.tsv")
    ap.add_argument("--n", type=int, default=25, help="examples to print")
    a = ap.parse_args()
    W = a.work

    meta = pd.read_parquet(f"{W}/emb_name/train_s1/meta.parquet", columns=["entity_id"])
    sample = set(meta.entity_id.str.slice(3).astype("int64").tolist())
    truth = load_truth(a.truth, sample)
    cand = pd.concat([pd.read_parquet(p, columns=["q", "src", "c"])
                      for p in sorted(glob.glob(f"{W}/pairs/train/part-*.parquet"))], ignore_index=True)
    cand["found"] = True
    t = truth.merge(cand, on=["q", "src", "c"], how="left")
    t["found"] = t.found.eq(True)
    del cand
    miss, anc = t[~t.found], t[t.found]
    print(f"{len(t):,} true pairs | found by v4 blocking {t.found.mean():.4f} | missed {len(miss):,}")
    has_anchor = miss.q.isin(set(anc.q.tolist()))
    print(f"missed pairs whose S1 has >=1 FOUND true match (expandable): {has_anchor.mean():.4f}")
    print(f"missed pairs whose S1 has NO found match at all:            {1 - has_anchor.mean():.4f}\n")

    lk = {f: {s: EmbLookup(f"{W}/emb_{f}/train_s{s}") for s in (1, 2, 3)} for f in ("name", "addr")}

    # similarity of each miss to its own S1 query
    m = miss.reset_index(drop=True)
    for f in ("name", "addr"):
        v = np.full(len(m), np.nan, np.float32)
        for s in (2, 3):
            k = (m.src == s).to_numpy()
            v[k] = cos_pairs(lk[f][1], lk[f][s], m.q.to_numpy()[k], m.c.to_numpy()[k])
        m[f"q_{f}"] = v

    # similarity of each miss to each anchor of the same query (all source combinations)
    x = m[["q", "src", "c"]].merge(anc[["q", "src", "c"]].rename(columns={"src": "asrc", "c": "ac"}), on="q")
    for f in ("name", "addr"):
        v = np.full(len(x), np.nan, np.float32)
        for s in (2, 3):
            for s2 in (2, 3):
                k = ((x.src == s) & (x.asrc == s2)).to_numpy()
                if k.any():
                    v[k] = cos_pairs(lk[f][s], lk[f][s2], x.c.to_numpy()[k], x.ac.to_numpy()[k])
        x[f"a_{f}"] = v
    x["a_best"] = x[["a_name", "a_addr"]].max(axis=1)
    best = x.sort_values("a_best", ascending=False).drop_duplicates(["q", "src", "c"])
    m = m.merge(best[["q", "src", "c", "asrc", "ac", "a_name", "a_addr", "a_best"]], on=["q", "src", "c"], how="left")
    m["q_best"] = m[["q_name", "q_addr"]].max(axis=1)

    print("share of MISSED pairs with similarity >= threshold   (to S1 query | to best anchor)")
    for th in (0.95, 0.9, 0.85, 0.8, 0.7):
        print(f"   >= {th:.2f}:   S1 {np.mean(m.q_best >= th):.3f}   |   anchor {np.mean(m.a_best >= th):.3f}"
              f"   (name {np.mean(m.a_name >= th):.3f}, addr {np.mean(m.a_addr >= th):.3f})")
    same = (m.src == m.asrc)
    print(f"\nbest anchor is in the SAME source as the miss: {same[m.a_best.notna()].mean():.3f}")
    print(f"miss source: S2 {np.mean(m.src == 2):.3f}  S3 {np.mean(m.src == 3):.3f}")
    gain = np.mean(m.a_best >= 0.85) * len(m) / len(t)
    print(f"\nrough recall gain if near-duplicates (>=0.85) of found matches were added: +{gain:.4f} "
          f"(0.{int(t.found.mean() * 10000):04d} -> {t.found.mean() + gain:.4f})")

    # examples: S1 text | anchor text | missed text
    con = duckdb.connect()
    ex = m[m.a_best.notna()].sample(min(a.n, int(m.a_best.notna().sum())), random_state=1)
    con.register("ex", ex)

    def text(src, col):
        return (f"SELECT CAST(substr(entity_id,4) AS BIGINT) AS id, name_emb || ' | ' || addr_emb AS txt "
                f"FROM read_parquet('{W}/l0/train_s{src}/part-*.parquet') WHERE CAST(substr(entity_id,4) AS BIGINT) "
                f"IN (SELECT {col} FROM ex)")
    tx = {}
    for s, col in ((1, "q"), (2, "c"), (3, "c"), (2, "ac"), (3, "ac")):
        tx.update({(s, i): v for i, v in con.execute(text(s, col)).fetchall()})
    print(f"\n=== {len(ex)} random missed pairs: S1 | best found anchor | MISSED record")
    for r in ex.itertuples():
        print(f"S1        : {tx.get((1, r.q), '?')[:110]}")
        print(f"anchor S{int(r.asrc)} : {tx.get((int(r.asrc), int(r.ac)), '?')[:110]}   (cos to miss {r.a_best:.2f})")
        print(f"MISSED S{r.src} : {tx.get((r.src, r.c), '?')[:110]}   (cos to S1 {r.q_best:.2f})\n")


if __name__ == "__main__":
    main()
