#!/usr/bin/env python3
"""
Layer 2 - candidate union + pair features  (input of the LightGBM matcher)

For one split (train | test):
  1. candidates = union of  name-dense top-K1  U  address-dense top-K2  U  lexical top-K3
     (this union is EXACTLY what the matching model scores -> it is what candidate_pairs.tsv holds)
  2. joins the Layer-0 fields of both records,
  3. computes ~50 similarity features per pair (string similarity, exact embedding cosines,
     street-number / postal overlap, blocker scores, "how does this candidate compare with the
     query's other candidates" group features),
  4. train split only: label = 1 if the pair is in train_ground_truth.tsv.
Written as parquet parts:  work/pairs/<split>/part-XXX.parquet   (processed in query chunks)

Directory conventions (as produced by the earlier steps):
  work/l0/<split>_s{1,2,3}            Layer 0
  work/emb_{name,addr}/<split>_s{1,2,3}  embeddings
  work/cand/<split>_{name,addr,lex}.parquet  blockers

    python build_pairs.py --split train --truth student_resource/dataset/train/train_ground_truth.tsv
    python build_pairs.py --split test --chunks 24
"""
import argparse
import os
import shutil
import sys
import time

import duckdb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

L0_COLS = ("entity_id, name_norm, name_core, name_sorted, dba_core, legal, addr_norm, addr_words, "
           "landmark, postal, streetnums, has_dba, has_nonlatin, name_missing, addr_missing")


# ----------------------------------------------------------------------------- embeddings
class EmbLookup:
    """entity numeric id -> row of emb.npy, for one (field, split, source)."""

    def __init__(self, d):
        meta = pd.read_parquet(os.path.join(d, "meta.parquet"), columns=["entity_id"])
        ids = meta.entity_id.str.slice(3).astype("int64").to_numpy()
        self.order = np.argsort(ids, kind="stable")
        self.sorted_ids = ids[self.order]
        self.emb = np.load(os.path.join(d, "emb.npy"), mmap_mode="r")

    def rows(self, ids):
        pos = np.searchsorted(self.sorted_ids, ids)
        pos = np.clip(pos, 0, len(self.sorted_ids) - 1)
        ok = self.sorted_ids[pos] == ids
        return self.order[pos], ok

    def gather(self, ids):
        r, ok = self.rows(ids)
        o = np.argsort(r, kind="stable")                     # sorted reads are much faster on a memmap
        out = np.empty((len(r), self.emb.shape[1]), np.float32)
        out[o] = np.asarray(self.emb[r[o]], dtype=np.float32)
        out[~ok] = 0
        return out


def cosines(qlk, clk2, clk3, q, src, c, batch=400_000):
    res = np.full(len(q), np.nan, np.float32)
    for s in range(0, len(q), batch):
        sl = slice(s, s + batch)
        qe = qlk.gather(q[sl])
        ce = np.zeros_like(qe)
        m2, m3 = src[sl] == 2, src[sl] == 3
        if m2.any():
            ce[m2] = clk2.gather(c[sl][m2])
        if m3.any():
            ce[m3] = clk3.gather(c[sl][m3])
        dot = np.einsum("ij,ij->i", qe, ce)
        empty = (np.abs(qe).sum(1) == 0) | (np.abs(ce).sum(1) == 0)
        dot[empty] = np.nan                                  # empty text: unknown, not "dissimilar"
        res[sl] = dot
    return res


# ----------------------------------------------------------------------------- string features
def sim(a, b, scorer, workers):
    return process.cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32) / 100.0


def string_features(df, workers):
    f = {}
    qn, cn = df.q_name_core.tolist(), df.c_name_core.tolist()
    f["n_ratio"] = sim(qn, cn, fuzz.ratio, workers)
    f["n_tset"] = sim(qn, cn, fuzz.token_set_ratio, workers)
    f["n_tsort"] = sim(qn, cn, fuzz.token_sort_ratio, workers)
    f["n_partial"] = sim(qn, cn, fuzz.partial_ratio, workers)
    f["n_jw"] = process.cpdist(qn, cn, scorer=JaroWinkler.normalized_similarity, workers=workers,
                               dtype=np.float32)
    qns = [s.replace(" ", "") for s in qn]
    cns = [s.replace(" ", "") for s in cn]
    f["n_nospace_ratio"] = sim(qns, cns, fuzz.ratio, workers)
    f["n_nospace_partial"] = sim(qns, cns, fuzz.partial_ratio, workers)
    f["n_norm_ratio"] = sim(df.q_name_norm.tolist(), df.c_name_norm.tolist(), fuzz.ratio, workers)
    cdba = df.c_dba_core.tolist()
    f["n_dba_tset"] = sim(qn, cdba, fuzz.token_set_ratio, workers)
    f["n_dba_tset"][df.c_dba_core.values == ""] = np.nan
    qa, ca = df.q_addr_norm.tolist(), df.c_addr_norm.tolist()
    f["a_ratio"] = sim(qa, ca, fuzz.ratio, workers)
    f["a_tset"] = sim(qa, ca, fuzz.token_set_ratio, workers)
    f["a_tsort"] = sim(qa, ca, fuzz.token_sort_ratio, workers)
    f["a_partial"] = sim(qa, ca, fuzz.partial_ratio, workers)
    f["aw_tset"] = sim(df.q_addr_words.tolist(), df.c_addr_words.tolist(), fuzz.token_set_ratio, workers)
    empty_a = (df.q_addr_norm.values == "") | (df.c_addr_norm.values == "")
    for k in ("a_ratio", "a_tset", "a_tsort", "a_partial", "aw_tset"):
        f[k][empty_a] = np.nan
    f["lm_tset"] = sim(df.q_landmark.tolist(), df.c_landmark.tolist(), fuzz.token_set_ratio, workers)
    f["lm_tset"][(df.q_landmark.values == "") | (df.c_landmark.values == "")] = np.nan
    f["q_name_len"] = df.q_name_core.str.len().to_numpy(np.float32)
    f["c_name_len"] = df.c_name_core.str.len().to_numpy(np.float32)
    f["q_name_words"] = df.q_name_core.str.count(" ").to_numpy(np.float32) + 1
    f["c_name_words"] = df.c_name_core.str.count(" ").to_numpy(np.float32) + 1
    f["c_addr_len"] = df.c_addr_norm.str.len().to_numpy(np.float32)
    return f


GROUP_COLS = ["cos_name", "cos_addr", "n_tset", "a_tset", "cos_sum"]


def group_features(out):
    """Relative position of a candidate among all candidates of the same query."""
    g = out.groupby("q", sort=False)
    for col in GROUP_COLS:
        v = out[col].fillna(-1)
        mx = v.groupby(out.q).transform("max")
        out[f"{col}_gap"] = (mx - v).astype(np.float32)
        out[f"{col}_rk"] = v.groupby(out.q).rank(ascending=False, method="min").astype(np.float32)
        # same, restricted to the same source (S2 vs S3 lists are independent)
        mx2 = v.groupby([out.q, out.src]).transform("max")
        out[f"{col}_gap_src"] = (mx2 - v).astype(np.float32)
    out["n_cand"] = g.q.transform("size").astype(np.float32)
    out["n_cand_src"] = out.groupby(["q", "src"]).q.transform("size").astype(np.float32)
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--work", default="work")
    ap.add_argument("--truth", help="train_ground_truth.tsv (train split only)")
    ap.add_argument("--k-name", type=int, default=3)
    ap.add_argument("--k-addr", type=int, default=3)
    ap.add_argument("--k-lex", type=int, default=10)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--only-chunk", type=int, default=-1, help="debug: build a single chunk")
    ap.add_argument("--workers", type=int, default=-1, help="rapidfuzz threads (-1 = all cores)")
    ap.add_argument("--threads", type=int, default=8, help="DuckDB threads")
    ap.add_argument("--mem", default="6GB")
    ap.add_argument("--cand-file", help="use exactly these (q, src, c) pairs instead of the blocker union "
                                        "(cluster expansion, see expand.py)")
    ap.add_argument("--out-dir", help="default: <work>/pairs/<split>")
    a = ap.parse_args()
    if a.split == "train" and not a.truth:
        sys.exit("--truth is required for --split train")

    W, S = a.work, a.split
    out_dir = a.out_dir or os.path.join(W, "pairs", S)
    if a.only_chunk < 0:
        shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(W, "duck_tmp")
    os.makedirs(tmp, exist_ok=True)

    t0 = time.time()

    def step(msg):
        print(f"[{time.time() - t0:7.0f}s] {msg}", flush=True)

    # on-disk database: the candidate/Layer-0 tables live in a file and RAM is only a cache,
    # so large test runs (100M+ pairs) cannot hit DuckDB's memory limit
    db_file = os.path.join(tmp, f"build_{S}{'_x' if a.cand_file else ''}.duckdb")
    for f in (db_file, db_file + ".wal"):
        if os.path.exists(f):
            os.remove(f)
    con = duckdb.connect(db_file)
    con.execute(f"SET threads={a.threads}; SET memory_limit='{a.mem}'; SET temp_directory='{tmp}'; "
                f"SET preserve_insertion_order=false;")
    for tag, k, rank in (("name", a.k_name, "name_rank"), ("addr", a.k_addr, "addr_rank")):
        con.execute(f"""CREATE TABLE c_{tag} AS SELECT q, src, c, {tag}_score, {rank}
                        FROM read_parquet('{W}/cand/{S}_{tag}.parquet') WHERE {rank} <= {k}""")
    con.execute(f"""CREATE TABLE c_lex AS SELECT q, src, c, lex_name, lex_addr, lex_rank
                    FROM read_parquet('{W}/cand/{S}_lex.parquet') WHERE lex_rank <= {a.k_lex}""")
    if a.cand_file:
        con.execute(f"""CREATE TABLE cand AS SELECT DISTINCT q, CAST(src AS TINYINT) AS src, c
                        FROM read_parquet('{a.cand_file}')""")
    else:
        con.execute("""CREATE TABLE cand AS SELECT DISTINCT q, src, c FROM (
                     SELECT q, src, c FROM c_name UNION ALL SELECT q, src, c FROM c_addr
                     UNION ALL SELECT q, src, c FROM c_lex)""")
    n_pairs, n_q = con.execute("SELECT count(*), count(DISTINCT q) FROM cand").fetchone()
    step(f"candidate union: {n_pairs:,} pairs for {n_q:,} queries ({n_pairs / max(n_q, 1):.1f}/query)")

    con.execute(f"""CREATE TABLE s1 AS SELECT CAST(substr(entity_id,4) AS BIGINT) AS id, {L0_COLS}
                    FROM read_parquet('{W}/l0/{S}_s1/part-*.parquet')
                    WHERE CAST(substr(entity_id,4) AS BIGINT) IN (SELECT DISTINCT q FROM cand)""")
    con.execute(f"""CREATE TABLE cd AS
        SELECT 2 AS src, CAST(substr(entity_id,4) AS BIGINT) AS id, {L0_COLS}
          FROM read_parquet('{W}/l0/{S}_s2/part-*.parquet')
          WHERE CAST(substr(entity_id,4) AS BIGINT) IN (SELECT c FROM cand WHERE src = 2)
        UNION ALL
        SELECT 3, CAST(substr(entity_id,4) AS BIGINT), {L0_COLS}
          FROM read_parquet('{W}/l0/{S}_s3/part-*.parquet')
          WHERE CAST(substr(entity_id,4) AS BIGINT) IN (SELECT c FROM cand WHERE src = 3)""")
    step("Layer-0 rows loaded")

    if a.truth:
        t = pd.read_csv(a.truth, sep="\t", dtype=str, keep_default_na=False, quoting=3)
        t["q"] = t.source1_entity_id.str.slice(3).astype("int64")
        ids = t.set_index("q").matched_entity_ids.str.split(",").explode()
        ids = ids[ids != ""]
        truth = pd.DataFrame({"q": ids.index.to_numpy(), "src": ids.str.slice(1, 2).astype("int8").to_numpy(),
                              "c": ids.str.slice(3).astype("int64").to_numpy()})
        con.register("truth_df", truth)
        con.execute("CREATE TABLE truth AS SELECT * FROM truth_df WHERE q IN (SELECT DISTINCT q FROM cand)")
        lab = "EXISTS (SELECT 1 FROM truth t WHERE t.q = u.q AND t.src = u.src AND t.c = u.c)"
    else:
        lab = "NULL"

    emb = {}
    for field in ("name", "addr"):
        emb[field] = (EmbLookup(f"{W}/emb_{field}/{S}_s1"), EmbLookup(f"{W}/emb_{field}/{S}_s2"),
                      EmbLookup(f"{W}/emb_{field}/{S}_s3"))
    step("embedding indexes ready")

    chunks = range(a.chunks) if a.only_chunk < 0 else [a.only_chunk]
    for i in chunks:
        df = con.execute(f"""
            WITH u AS (SELECT * FROM cand WHERE q % {a.chunks} = {i})
            SELECT u.q, u.src, u.c,
                   n.name_score, n.name_rank, ad.addr_score, ad.addr_rank,
                   l.lex_name, l.lex_addr, l.lex_rank,
                   s.name_norm AS q_name_norm, s.name_core AS q_name_core, s.addr_norm AS q_addr_norm,
                   s.addr_words AS q_addr_words, s.landmark AS q_landmark,
                   d.name_norm AS c_name_norm, d.name_core AS c_name_core, d.dba_core AS c_dba_core,
                   d.addr_norm AS c_addr_norm, d.addr_words AS c_addr_words, d.landmark AS c_landmark,
                   CAST(d.has_dba AS FLOAT) AS c_has_dba, CAST(d.has_nonlatin AS FLOAT) AS c_nonlatin,
                   CAST(d.name_missing AS FLOAT) AS c_name_missing, CAST(d.addr_missing AS FLOAT) AS c_addr_missing,
                   -- street / unit numbers: 1 shared, 0.5 one side has none, 0 both have but none shared
                   CASE WHEN s.streetnums = '' OR d.streetnums = '' THEN 0.5
                        WHEN len(list_intersect(string_split(s.streetnums,'|'), string_split(d.streetnums,'|'))) > 0
                        THEN 1.0 ELSE 0.0 END AS num_match,
                   CASE WHEN s.streetnums = '' OR d.streetnums = '' THEN NULL
                        ELSE len(list_intersect(string_split(s.streetnums,'|'), string_split(d.streetnums,'|')))::FLOAT
                             / len(list_distinct(list_concat(string_split(s.streetnums,'|'), string_split(d.streetnums,'|'))))
                        END AS num_jacc,
                   CASE WHEN s.streetnums = '' OR d.streetnums = '' THEN NULL
                        ELSE (string_split(s.streetnums,'|')[1] = string_split(d.streetnums,'|')[1])::FLOAT END AS num_first_eq,
                   CASE WHEN s.postal = '' OR d.postal = '' THEN 0.5
                        WHEN len(list_intersect(string_split(s.postal,'|'), string_split(d.postal,'|'))) > 0
                        THEN 1.0 ELSE 0.0 END AS postal_match,
                   CASE WHEN s.legal = '' OR d.legal = '' THEN NULL
                        ELSE len(list_intersect(string_split(s.legal,' '), string_split(d.legal,' ')))::FLOAT
                             / len(list_distinct(list_concat(string_split(s.legal,' '), string_split(d.legal,' '))))
                        END AS legal_jacc,
                   {lab} AS label
            FROM u
            JOIN s1 s ON s.id = u.q
            JOIN cd d ON d.src = u.src AND d.id = u.c
            LEFT JOIN c_name n ON n.q = u.q AND n.src = u.src AND n.c = u.c
            LEFT JOIN c_addr ad ON ad.q = u.q AND ad.src = u.src AND ad.c = u.c
            LEFT JOIN c_lex l ON l.q = u.q AND l.src = u.src AND l.c = u.c
        """).df()
        if df.empty:
            continue
        for col in ("q_name_norm", "q_name_core", "q_addr_norm", "q_addr_words", "q_landmark", "c_name_norm",
                    "c_name_core", "c_dba_core", "c_addr_norm", "c_addr_words", "c_landmark"):
            df[col] = df[col].fillna("").astype(str)
        feats = string_features(df, a.workers)
        q, src, c = df.q.to_numpy(np.int64), df.src.to_numpy(np.int8), df.c.to_numpy(np.int64)
        feats["cos_name"] = cosines(*emb["name"], q, src, c)
        feats["cos_addr"] = cosines(*emb["addr"], q, src, c)
        drop = [x for x in df.columns if x.startswith(("q_", "c_name_norm", "c_name_core", "c_dba_core",
                                                      "c_addr_norm", "c_addr_words", "c_landmark"))]
        out = df.drop(columns=drop)
        for k, v in feats.items():
            out[k] = v
        out["cos_sum"] = out.cos_name.fillna(0) + out.cos_addr.fillna(0)
        out["src"] = out.src.astype(np.int8)
        for col in ("name_rank", "addr_rank", "lex_rank"):
            out[col] = out[col].astype(np.float32)
        for col in ("name_score", "addr_score", "lex_name", "lex_addr", "num_match", "num_jacc",
                    "num_first_eq", "postal_match", "legal_jacc"):
            out[col] = out[col].astype(np.float32)
        out = group_features(out)
        if a.split == "train":
            out["label"] = out.label.astype(np.int8)
        else:
            out = out.drop(columns=["label"])
        out.to_parquet(os.path.join(out_dir, f"part-{i:03d}.parquet"), index=False)
        extra = f" | positives {int(out.label.sum()):,}" if a.split == "train" else ""
        step(f"chunk {i + 1}/{a.chunks}: {len(out):,} pairs{extra}")
    con.close()
    for f in (db_file, db_file + ".wal"):
        if os.path.exists(f):
            os.remove(f)
    step(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
