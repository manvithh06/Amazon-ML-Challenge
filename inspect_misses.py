#!/usr/bin/env python3
"""
Diagnostic: WHY are true matches missing from the candidate set?

Joins every true (S1, S2/S3) pair of the query sample with its Layer-0 fields, marks whether the
dense / lexical blocker found it, then prints
  (1) recall broken down by country, source, script, address availability, shared name words,
      shared street numbers ...
  (2) random examples of pairs that NEITHER blocker found, and a few that were found, side by side.

    python inspect_misses.py --queries work/emb/train_s1 \
        --truth student_resource/dataset/train/train_ground_truth.tsv \
        --l0-s1 work/l0/train_s1 --l0-s2 work/l0/train_s2 --l0-s3 work/l0/train_s3 \
        --dense work/cand/train_dense.parquet --lex work/cand/train_lex.parquet --n 30
"""
import argparse
import os

import duckdb
import pandas as pd

from eval_blocking import load_truth


def cut(s, n=105):
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[:n - 1] + "…"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--l0-s1", required=True)
    ap.add_argument("--l0-s2", required=True)
    ap.add_argument("--l0-s3", required=True)
    ap.add_argument("--dense", required=True)
    ap.add_argument("--lex", required=True)
    ap.add_argument("--k", type=int, default=20, help="only candidates ranked <= k count as found")
    ap.add_argument("--n", type=int, default=30, help="examples to print")
    ap.add_argument("--seed", type=float, default=0.42)
    a = ap.parse_args()

    meta = pd.read_parquet(os.path.join(a.queries, "meta.parquet"))
    truth = load_truth(a.truth, set(meta.entity_id.str.slice(3).astype("int64")))
    for label, path, col in (("dense", a.dense, "dense_rank"), ("lex", a.lex, "lex_rank")):
        c = pd.read_parquet(path, columns=["q", "src", "c", col])
        c = c[c[col] <= a.k][["q", "src", "c"]].drop_duplicates()
        c[f"found_{label}"] = True
        truth = truth.merge(c, on=["q", "src", "c"], how="left")
        truth[f"found_{label}"] = truth[f"found_{label}"].eq(True)
    truth["found_any"] = truth.found_dense | truth.found_lex
    print(f"{len(truth):,} true pairs | dense {truth.found_dense.mean():.3f} | lex {truth.found_lex.mean():.3f} "
          f"| either {truth.found_any.mean():.3f}\n")

    con = duckdb.connect()
    con.register("t", truth)
    cols = "name_core, name_emb, addr_emb, addr_words, streetnums, has_nonlatin"
    con.execute(f"""CREATE VIEW s1 AS SELECT CAST(substr(entity_id,4) AS BIGINT) AS id, country, {cols}
                    FROM read_parquet('{a.l0_s1}/part-*.parquet')""")
    con.execute(f"""CREATE VIEW cd AS
        SELECT 2 AS src, CAST(substr(entity_id,4) AS BIGINT) AS id, dba_core, addr_missing, {cols}
          FROM read_parquet('{a.l0_s2}/part-*.parquet')
        UNION ALL
        SELECT 3, CAST(substr(entity_id,4) AS BIGINT), dba_core, addr_missing, {cols}
          FROM read_parquet('{a.l0_s3}/part-*.parquet')""")
    con.execute("""CREATE TABLE j AS
        SELECT t.q, t.src, t.c, t.found_dense, t.found_lex, t.found_any, s1.country,
               s1.name_core AS q_name, s1.name_emb AS q_name_emb, s1.addr_emb AS q_addr_emb,
               s1.streetnums AS q_nums, s1.addr_words AS q_words,
               cd.name_core AS c_name, cd.dba_core AS c_dba, cd.name_emb AS c_name_emb,
               cd.addr_emb AS c_addr_emb, cd.streetnums AS c_nums, cd.addr_words AS c_words,
               cd.has_nonlatin AS c_nonlatin, cd.addr_missing AS c_addr_missing
        FROM t JOIN s1 ON s1.id = t.q JOIN cd ON cd.src = t.src AND cd.id = t.c""")
    n_join = con.execute("SELECT count(*) FROM j").fetchone()[0]
    print(f"joined {n_join:,} of {len(truth):,} true pairs with their Layer-0 rows\n")

    con.execute("""ALTER TABLE j ADD COLUMN name_ov INTEGER; ALTER TABLE j ADD COLUMN num_ov INTEGER;
                   ALTER TABLE j ADD COLUMN word_ov INTEGER;
        UPDATE j SET
          name_ov = coalesce(len(list_intersect(string_split(nullif(q_name,''),' '),
                                  string_split(nullif(trim(c_name || ' ' || c_dba),''),' '))), 0),
          num_ov  = coalesce(len(list_intersect(string_split(nullif(q_nums,''),'|'),
                                  string_split(nullif(c_nums,''),'|'))), 0),
          word_ov = coalesce(len(list_intersect(string_split(nullif(q_words,''),' '),
                                  string_split(nullif(c_words,''),' '))), 0)""")

    def table(title, expr):
        df = con.execute(f"""SELECT {expr} AS bucket, count(*) AS pairs,
                   round(avg(found_dense::INT), 3) AS dense, round(avg(found_lex::INT), 3) AS lex,
                   round(avg(found_any::INT), 3) AS either
                 FROM j GROUP BY 1 ORDER BY 1""").df()
        df["share"] = (df.pairs / df.pairs.sum()).round(3)
        print(f"--- recall by {title}\n{df.to_string(index=False)}\n")

    table("country", "country")
    table("source", "src")
    table("candidate has non-Latin script", "c_nonlatin")
    table("candidate address missing", "c_addr_missing")
    table("shared name words (S1 vs candidate)",
          "CASE WHEN name_ov = 0 THEN '0 shared' WHEN name_ov = 1 THEN '1 shared' ELSE '2+ shared' END")
    table("street numbers",
          "CASE WHEN q_nums = '' OR c_nums = '' THEN 'a side has none' WHEN num_ov > 0 THEN 'share a number' ELSE 'both have, none shared' END")
    table("shared address words",
          "CASE WHEN word_ov = 0 THEN '0 shared' WHEN word_ov <= 2 THEN '1-2 shared' ELSE '3+ shared' END")

    con.execute(f"SELECT setseed({a.seed})")
    for title, cond, n in (("NOT found by either blocker", "NOT found_any", a.n),
                           ("found (for contrast)", "found_any", max(5, a.n // 4))):
        print(f"=== random pairs: {title}")
        for r in con.execute(f"""SELECT src, country, q_name_emb, q_addr_emb, c_name_emb, c_addr_emb
                                 FROM j WHERE {cond} ORDER BY random() LIMIT {n}""").fetchall():
            print(f"[S{r[0]} {r[1]}]\n   S1 : {cut(r[2])} | {cut(r[3])}\n   S{r[0]} : {cut(r[4])} | {cut(r[5])}")
        print()


if __name__ == "__main__":
    main()
