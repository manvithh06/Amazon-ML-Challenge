#!/usr/bin/env python3
"""
Blocking quality on the training query sample: what fraction of TRUE pairs do the candidates contain?
(This is the recall ceiling of the whole pipeline - later stages can only choose among candidates.)

    python eval_blocking.py --queries work/emb_name/train_s1 \
        --truth student_resource/dataset/train/train_ground_truth.tsv \
        --cand name=work/cand/train_name.parquet=name_rank \
        --cand addr=work/cand/train_addr.parquet=addr_rank \
        --cand lex=work/cand/train_lex.parquet=lex_rank
    (--cand LABEL=PATH=RANKCOLUMN; a candidate counts as found when its rank <= k)
"""
import argparse
import os

import numpy as np
import pandas as pd


def load_truth(path, sample_ids):
    t = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)
    t["q"] = t.source1_entity_id.str.slice(3).astype("int64")
    t = t[t.q.isin(sample_ids)]
    ids = t.set_index("q").matched_entity_ids.str.split(",").explode()
    ids = ids[ids != ""]
    return pd.DataFrame({"q": ids.index.to_numpy(),
                         "src": ids.str.slice(1, 2).astype("int8").to_numpy(),
                         "c": ids.str.slice(3).astype("int64").to_numpy()})


def report(name, truth, cand, n_queries):
    keys = cand[["q", "src", "c"]].drop_duplicates()
    m = truth.merge(keys, on=["q", "src", "c"], how="left", indicator=True)
    found = (m._merge == "both")
    per_q = found.groupby(m.q).all()
    by_src = found.groupby(m.src).mean()
    print(f"{name:<22s} pair recall {found.mean():.4f} | S2 {by_src.get(2, np.nan):.4f} "
          f"S3 {by_src.get(3, np.nan):.4f} | queries fully covered {per_q.mean():.4f} | "
          f"cand/query {len(keys) / n_queries:5.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True, help="embedding dir of the S1 sample (meta.parquet)")
    ap.add_argument("--truth", required=True)
    ap.add_argument("--cand", action="append", default=[], help="LABEL=PATH=RANKCOLUMN (repeatable)")
    ap.add_argument("--ks", default="1,3,5,10,20")
    ap.add_argument("--union-grid", default="",
                    help="e.g. 'name:3,5,10;addr:3,5,10;lex:5,10,20' -> union recall for every combination")
    a = ap.parse_args()

    meta = pd.read_parquet(os.path.join(a.queries, "meta.parquet"))
    sample = meta.entity_id.str.slice(3).astype("int64").to_numpy()
    truth = load_truth(a.truth, set(sample.tolist()))
    n_q = len(sample)
    print(f"{n_q:,} query entities | {len(truth):,} true S2/S3 pairs among them "
          f"| singletons {1 - truth.q.nunique() / n_q:.3f}\n")

    if a.union_grid:
        import itertools
        cands = {}
        for spec in a.cand:
            label, path, col = spec.split("=")
            c = pd.read_parquet(path, columns=["q", "src", "c", col])
            cands[label] = c.rename(columns={col: "rank"})
        grid = [(g.split(":")[0], [int(x) for x in g.split(":")[1].split(",")]) for g in a.union_grid.split(";")]
        for combo in itertools.product(*[ks for _, ks in grid]):
            u = pd.concat([cands[l][cands[l]["rank"] <= k][["q", "src", "c"]] for (l, _), k in zip(grid, combo)])
            report(" ".join(f"{l}{k}" for (l, _), k in zip(grid, combo)), truth, u, n_q)
        return

    parts = []
    for spec in a.cand:
        label, path, col = spec.split("=")
        c = pd.read_parquet(path, columns=["q", "src", "c", col])
        parts.append(c[c[col] <= max(int(x) for x in a.ks.split(","))][["q", "src", "c"]])
        for k in [int(x) for x in a.ks.split(",")]:
            report(f"{label} top-{k}", truth, c[c[col] <= k], n_q)
        print()
    if len(parts) > 1:
        report("UNION (all candidates)", truth, pd.concat(parts), n_q)


if __name__ == "__main__":
    main()
