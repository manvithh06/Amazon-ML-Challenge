#!/usr/bin/env python3
"""
Diagnostic: is there structure in IDs / file order that links matching records?
(A 0.99 leaderboard score needs ~99% blocking recall, which text similarity alone rarely gives.)

    python check_structure.py --train student_resource/dataset/train
"""
import argparse
import os

import numpy as np
import pandas as pd


def ids(path):
    s = pd.read_csv(path, sep="\t", usecols=["entity_id"], dtype=str, quoting=3, keep_default_na=False).entity_id
    return pd.DataFrame({"id": s.str.slice(3).astype("int64").to_numpy(), "pos": np.arange(len(s))})


def spearman(a, b):
    return float(np.corrcoef(pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy())[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="student_resource/dataset/train")
    a = ap.parse_args()
    d = a.train
    s1, s2, s3 = (ids(os.path.join(d, f"train_source{i}.tsv")) for i in (1, 2, 3))
    print(f"rows: S1 {len(s1):,}  S2 {len(s2):,}  S3 {len(s3):,}")
    for n, s in (("S1", s1), ("S2", s2), ("S3", s3)):
        print(f"{n}: id range {s.id.min():,}..{s.id.max():,} | ids sorted in file: {bool(np.all(np.diff(s.id.to_numpy()) > 0))}"
              f" | corr(id, row) {spearman(s.id, s.pos):+.3f}")

    t = pd.read_csv(os.path.join(d, "train_ground_truth.tsv"), sep="\t", dtype=str, keep_default_na=False, quoting=3)
    t["q"] = t.source1_entity_id.str.slice(3).astype("int64")
    m = t.set_index("q").matched_entity_ids.str.split(",").explode()
    m = m[m != ""]
    pr = pd.DataFrame({"q": m.index.to_numpy(), "src": m.str.slice(1, 2).astype(int).to_numpy(),
                       "c": m.str.slice(3).astype("int64").to_numpy()})
    pr = pr.merge(s1.rename(columns={"id": "q", "pos": "qpos"}), on="q")
    print(f"\n{len(pr):,} true pairs")
    for src, s in ((2, s2), (3, s3)):
        p = pr[pr.src == src].merge(s.rename(columns={"id": "c", "pos": "cpos"}), on="c")
        print(f"\n--- S1 -> S{src}: {len(p):,} pairs")
        print(f"  spearman(S1 id, S{src} id)   {spearman(p.q, p.c):+.4f}")
        print(f"  spearman(S1 row, S{src} row) {spearman(p.qpos, p.cpos):+.4f}")
        # if order is informative, the matched record sits near the 'expected' position
        exp = p.qpos / len(s1) * len(s)
        dev = np.abs(p.cpos - exp) / len(s)
        print(f"  |row deviation| as share of file: median {np.median(dev):.4f}  (random ~0.25)")
        for w in (10, 100, 1000):
            idd = np.abs(p.c - p.q)
            print(f"  |S{src} id - S1 id| <= {w:>4}: {np.mean(idd <= w):.4f}")
        print(p.sample(8, random_state=0)[["q", "c", "qpos", "cpos"]].to_string(index=False))
    # records of one cluster: are S2 and S3 ids of the same S1 related to each other?
    both = pr.pivot_table(index="q", columns="src", values="c", aggfunc="min").dropna()
    if len(both):
        print(f"\nS1 with both S2 and S3 matches: {len(both):,} | spearman(S2 id, S3 id) {spearman(both[2], both[3]):+.4f}")


if __name__ == "__main__":
    main()
