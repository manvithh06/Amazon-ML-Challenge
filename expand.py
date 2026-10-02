#!/usr/bin/env python3
"""
Layer 4 - CLUSTER EXPANSION (second-pass blocking + expansion matcher)

Every S1 business has ~3.5 true S2/S3 records.  Some copies look nothing like the S1 text
("POTTSDESERT | 819 G ST") but are near-duplicates of another copy we DID match
("Potts Desert Kochav L.L.C. | 819 G STREET").  Diagnostic on train: 94% of blocking misses belong to
an S1 with at least one found match, and 58% of misses have cosine >= 0.85 to such a match.

  anchors   : the v4 pipeline's predicted matches (pruner + matcher + F0.5 rule + one-to-one)
  neighbors : for every anchor record, GPU top-k nearest records in S2 and S3 (same country,
              name vectors and address vectors)
  pairs     : new candidates (S1 query, neighbour of its anchor) that v4 never scored and that no other
              S1 already claimed, with expansion features (anchor->candidate cosines, anchor score ...)
              -> then build_pairs.py --cand-file computes the usual S1<->candidate features
  train     : LightGBM on the expansion pairs; tunes its threshold JOINTLY with v4 on the valid queries
              (official macro F0.5) and reports the gain over v4
  predict   : v4 test predictions + expansion predictions -> one-to-one -> submission files

    python expand.py anchors   --split train --truth student_resource/dataset/train/train_ground_truth.tsv
    python expand.py neighbors --split train
    python expand.py pairs     --split train
    python build_pairs.py --split train --truth ... --cand-file work/exp/train_cand.parquet --out-dir work/exp/pairs_train
    python expand.py train     --truth student_resource/dataset/train/train_ground_truth.tsv
    (test: anchors / neighbors / pairs / build_pairs with --split test, then  python expand.py predict)
"""
import argparse
import glob
import json
import os
import shutil
import time

import duckdb
import numpy as np
import pandas as pd

from match import ID_COLS, decide, feature_cols, load_truth, macro_f05, prune_keep

EXP_FEATS = ["exp_max", "exp_max_name", "exp_max_addr", "exp_sum", "exp_hits", "exp_anchors",
             "exp_best_p", "exp_min_rank", "exp_same_src", "exp_n_anchor_q"]
VALID_HASH = 2654435761


def is_valid(q, permille=100):
    return (np.asarray(q) * VALID_HASH % 1000) < permille


def log(t0, msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------- 1. anchors
def cmd_anchors(a):
    """v4 predictions = anchors.  Also stores every pair v4 scored (these are not re-proposed)."""
    import lightgbm as lgb
    t0 = time.time()
    W, S = a.work, a.split
    os.makedirs(f"{W}/exp", exist_ok=True)
    pj = json.load(open(f"{W}/model/prune.json"))
    tau = pj["tau"]
    if S == "test":
        scored = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{W}/preds_test/part-*.parquet"))],
                           ignore_index=True)
    else:
        model = lgb.Booster(model_file=f"{W}/model/lgbm.txt")
        pruner = lgb.Booster(model_file=f"{W}/model/pruner.txt")
        dec = json.load(open(f"{W}/model/decision.json"))
        parts = []
        for pth in sorted(glob.glob(f"{W}/pairs/train/part-*.parquet")):
            df = pd.read_parquet(pth)
            pp = pruner.predict(df[pj["features"]].astype(np.float32))
            df = df[prune_keep(df.q.to_numpy(), pp, pj["top_n"], pj["min_pp"])]
            o = df[ID_COLS + ["label"]].copy()
            o["p"] = model.predict(df[dec["features"]].astype(np.float32), num_iteration=dec.get("best_iteration"))
            parts.append(o)
        scored = pd.concat(parts, ignore_index=True)
        log(t0, f"scored {len(scored):,} pruned train pairs with the v4 model")
    scored = scored.reset_index(drop=True)
    scored["src"] = scored.src.astype(np.int8)
    scored.to_parquet(f"{W}/exp/{S}_scored.parquet", index=False)
    keep = decide(scored, tau, pj["rho"])
    anc = scored.loc[keep, ID_COLS + ["p"]].reset_index(drop=True)
    anc.to_parquet(f"{W}/exp/{S}_anchors.parquet", index=False)
    log(t0, f"{len(anc):,} anchors (tau {tau}) for {anc.q.nunique():,} S1 entities "
            f"({len(anc) / max(anc.q.nunique(), 1):.2f} per entity)")
    if S == "train" and a.truth:
        nq = scored.q.unique()
        print(f"   anchor precision on train sample: {scored.loc[keep, 'label'].mean():.4f}")


# ----------------------------------------------------------------------------- 2. neighbours
def cmd_neighbors(a):
    from retrieve_dense import topk_search
    t0 = time.time()
    W, S = a.work, a.split
    anc = pd.read_parquet(f"{W}/exp/{S}_anchors.parquet", columns=["src", "c"]).drop_duplicates()
    out_dir = f"{W}/exp/{S}_nbr"
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    n_part = 0
    for field in ("name", "addr"):
        meta, emb = {}, {}
        for s in (2, 3):
            m = pd.read_parquet(f"{W}/emb_{field}/{S}_s{s}/meta.parquet")
            meta[s] = (m.entity_id.str.slice(3).astype("int64").to_numpy(), m.country.to_numpy())
            emb[s] = np.load(f"{W}/emb_{field}/{S}_s{s}/emb.npy", mmap_mode="r")
        for s in (2, 3):
            ids_s, ctry_s = meta[s]
            order = np.argsort(ids_s, kind="stable")
            want = anc.c[anc.src == s].to_numpy()
            pos = np.clip(np.searchsorted(ids_s[order], want), 0, len(ids_s) - 1)
            ok = ids_s[order][pos] == want
            rows_s = np.sort(order[pos[ok]])
            for t in (2, 3):
                ids_t, ctry_t = meta[t]
                k = a.k + (1 if s == t else 0)
                for ctry in pd.unique(ctry_s[rows_s]):
                    qi = rows_s[ctry_s[rows_s] == ctry]
                    di = np.flatnonzero(ctry_t == ctry)
                    if len(qi) == 0 or len(di) == 0:
                        continue
                    t1 = time.time()
                    qe = np.asarray(emb[s][qi])
                    nonzero = np.abs(qe.astype(np.float32)).sum(1) > 0           # empty text -> no neighbours
                    qi, qe = qi[nonzero], qe[nonzero]
                    if len(qi) == 0:
                        continue
                    idx, val = topk_search(qe, np.asarray(emb[t][di]), k, a.q_chunk, a.db_block, a.device)
                    ke = idx.shape[1]
                    df = pd.DataFrame({"as": np.int8(s), "ac": np.repeat(ids_s[qi], ke), "ts": np.int8(t),
                                       "tc": ids_t[di][idx.ravel()], "score": val.ravel().astype(np.float32),
                                       "rank": np.tile(np.arange(1, ke + 1, dtype=np.int16), len(qi)),
                                       "field": np.int8(0 if field == "name" else 1)})
                    df = df[(df.score >= a.min_score) & ~((df["as"] == df.ts) & (df.ac == df.tc))]
                    df.to_parquet(f"{out_dir}/part-{n_part:04d}.parquet", index=False)
                    n_part += 1
                    log(t0, f"{field} S{s}->S{t} {ctry:>8s}: {len(qi):>9,} anchors x {len(di):>9,} rows "
                            f"-> {len(df):,} neighbours ({time.time() - t1:.0f}s)")
    log(t0, f"done -> {out_dir}")


# ----------------------------------------------------------------------------- 3. expansion candidates
def cmd_pairs(a):
    t0 = time.time()
    W, S = a.work, a.split
    tmp = f"{W}/duck_tmp"
    os.makedirs(tmp, exist_ok=True)
    dbf = f"{tmp}/expand_{S}.duckdb"
    for f in (dbf, dbf + ".wal"):
        if os.path.exists(f):
            os.remove(f)
    con = duckdb.connect(dbf)
    con.execute(f"SET threads={a.threads}; SET memory_limit='{a.mem}'; SET temp_directory='{tmp}'; "
                f"SET preserve_insertion_order=false;")
    con.execute(f"CREATE TABLE anc AS SELECT q, CAST(src AS TINYINT) AS src, c, p FROM read_parquet('{W}/exp/{S}_anchors.parquet')")
    con.execute(f"CREATE TABLE scored AS SELECT q, CAST(src AS TINYINT) AS src, c FROM read_parquet('{W}/exp/{S}_scored.parquet')")
    con.execute(f"""CREATE TABLE hits AS
        SELECT a.q, n.ts AS src, n.tc AS c, n.field, n.score, n.rank, a.c AS ac, a.src AS asrc, a.p
        FROM anc a JOIN read_parquet('{W}/exp/{S}_nbr/part-*.parquet') n ON n."as" = a.src AND n.ac = a.c
        WHERE n.rank <= {a.k}""")
    log(t0, f"anchor->neighbour hits: {con.execute('SELECT count(*) FROM hits').fetchone()[0]:,}")
    con.execute(f"""CREATE TABLE cand AS
        WITH g AS (
          SELECT q, src, c,
                 max(score) AS exp_max,
                 max(CASE WHEN field = 0 THEN score END) AS exp_max_name,
                 max(CASE WHEN field = 1 THEN score END) AS exp_max_addr,
                 sum(score) AS exp_sum, count(*) AS exp_hits, count(DISTINCT asrc * 10000000000 + ac) AS exp_anchors,
                 max(p) AS exp_best_p, min(rank) AS exp_min_rank,
                 max(CASE WHEN asrc = src THEN 1 ELSE 0 END) AS exp_same_src
          FROM hits GROUP BY q, src, c),
        f AS (   -- never re-propose a pair v4 already scored, nor a record some S1 already owns
          SELECT g.* FROM g
          WHERE g.exp_max >= {a.min_score}
            AND NOT EXISTS (SELECT 1 FROM scored s WHERE s.q = g.q AND s.src = g.src AND s.c = g.c)
            AND NOT EXISTS (SELECT 1 FROM anc x WHERE x.src = g.src AND x.c = g.c)),
        r AS (SELECT *, row_number() OVER (PARTITION BY q, src ORDER BY exp_max DESC, exp_sum DESC, c) AS rk FROM f)
        SELECT r.q, r.src, r.c, exp_max, exp_max_name, exp_max_addr, exp_sum, exp_hits, exp_anchors,
               exp_best_p, exp_min_rank, exp_same_src, n.n AS exp_n_anchor_q
        FROM r JOIN (SELECT q, count(*) AS n FROM anc GROUP BY q) n USING (q)
        WHERE rk <= {a.per_src}""")
    n, nq = con.execute("SELECT count(*), count(DISTINCT q) FROM cand").fetchone()
    con.execute(f"COPY (SELECT * FROM cand) TO '{W}/exp/{S}_cand.parquet' (FORMAT parquet)")
    n_s1 = con.execute(f"SELECT count(*) FROM read_parquet('{W}/emb_name/{S}_s1/meta.parquet')").fetchone()[0]
    log(t0, f"expansion candidates: {n:,} pairs for {nq:,} S1 entities "
            f"(+{n / n_s1:.2f} per S1 entity) -> {W}/exp/{S}_cand.parquet")
    if S == "train" and a.truth:
        from eval_blocking import load_truth as lt
        truth = lt(a.truth, set(con.execute("SELECT q FROM anc").df().q.tolist()) |
                   set(pd.read_parquet(f"{W}/emb_name/train_s1/meta.parquet").entity_id.str.slice(3).astype("int64")))
        con.register("truth", truth)
        tot = len(truth)
        old = con.execute("SELECT count(*) FROM truth t JOIN scored s USING (q, src, c)").fetchone()[0]
        old_all = con.execute(f"""SELECT count(*) FROM truth t JOIN
            (SELECT DISTINCT q, src, c FROM read_parquet('{W}/pairs/train/part-*.parquet')) s USING (q, src, c)""").fetchone()[0]
        new = con.execute("SELECT count(*) FROM truth t JOIN cand s USING (q, src, c)").fetchone()[0]
        print(f"   true pairs: {tot:,} | v4 blocking {old_all / tot:.4f} | v4 after pruner {old / tot:.4f} | "
              f"+ expansion {(old + new) / tot:.4f}  (expansion precision {new / max(n, 1):.3f})")
    con.close()
    for f in (dbf, dbf + ".wal"):
        if os.path.exists(f):
            os.remove(f)


def load_exp_pairs(W, S):
    df = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{W}/exp/pairs_{S}/part-*.parquet"))],
                   ignore_index=True)
    ef = pd.read_parquet(f"{W}/exp/{S}_cand.parquet")
    ef["src"] = ef.src.astype(np.int8)
    df["src"] = df.src.astype(np.int8)
    df = df.merge(ef, on=ID_COLS, how="left")
    for c in EXP_FEATS:
        df[c] = df[c].astype(np.float32)
    return df


def joint_decide(df, tau_v4, tau_x):
    """df: q, src, c, p, is_x.  v4 pairs use tau_v4, expansion pairs tau_x, then one-to-one by p."""
    p, x = df.p.to_numpy(), df.is_x.to_numpy()
    keep = np.where(x, p >= tau_x, p >= tau_v4)
    sub = df[keep]
    out = np.zeros(len(df), bool)
    if len(sub):
        top = sub.sort_values(["p", "q"], ascending=[False, True]).drop_duplicates(["src", "c"])
        out[df.index.get_indexer(top.index)] = True
    return out


# ----------------------------------------------------------------------------- 4. train
def cmd_train(a):
    import lightgbm as lgb
    t0 = time.time()
    W = a.work
    df = load_exp_pairs(W, "train")
    feats = feature_cols(df)
    val = is_valid(df.q, a.valid_permille)
    print(f"{len(df):,} expansion pairs, {df.label.mean():.3f} positive, {len(feats)} features | "
          f"fit {int((~val).sum()):,} valid {int(val.sum()):,}", flush=True)
    X = df[feats].astype(np.float32)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=a.threads, verbose=-1, metric=["binary_logloss", "auc"])
    m = lgb.train(params, lgb.Dataset(X[~val], df.label[~val]), num_boost_round=2000,
                  valid_sets=[lgb.Dataset(X[val], df.label[val])], valid_names=["valid"],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)])
    os.makedirs(f"{W}/exp/model", exist_ok=True)
    m.save_model(f"{W}/exp/model/lgbm.txt")
    imp = pd.Series(m.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print("\ntop features (gain):\n" + (imp.head(15) / imp.sum()).round(4).to_string())

    xv = df.loc[val, ID_COLS + ["label"]].copy()
    xv["p"] = m.predict(X[val], num_iteration=m.best_iteration)
    xv["is_x"] = True
    base = pd.read_parquet(f"{W}/exp/train_scored.parquet")
    base = base[is_valid(base.q, a.valid_permille)].copy()
    base["is_x"] = False
    both = pd.concat([base, xv], ignore_index=True)
    meta = pd.read_parquet(f"{W}/emb_name/train_s1/meta.parquet", columns=["entity_id"])
    all_q = meta.entity_id.str.slice(3).astype("int64").to_numpy()
    n_true = load_truth(a.truth, all_q[is_valid(all_q, a.valid_permille)])
    tau4 = json.load(open(f"{W}/model/prune.json"))["tau"]
    f0 = macro_f05(both, joint_decide(both, tau4, 9.0), n_true)
    print(f"\nvalid: blocking recall v4 {base.label.sum() / n_true.sum():.4f} -> with expansion "
          f"{both.label.sum() / n_true.sum():.4f}")
    print(f"v4 alone (tau {tau4}): F0.5 {f0[0]:.4f}  prec {f0[1]:.4f}  rec {f0[2]:.4f}\n")
    print(f"{'tau_x':>6} | {'F0.5':>7} {'prec':>6} {'rec':>6} {'F_single':>8} {'F_nonsingle':>11}")
    best = (f0[0], 9.0)
    for tx in np.round(np.arange(0.30, 0.96, 0.05), 2):
        r = macro_f05(both, joint_decide(both, tau4, tx), n_true)
        print(f"{tx:6.2f} | {r[0]:7.4f} {r[1]:6.3f} {r[2]:6.3f} {r[3]:8.3f} {r[4]:11.3f}")
        if r[0] > best[0]:
            best = (r[0], float(tx))
    # candidate-set cut: expansion pairs below this score are not even kept in candidate_pairs
    json.dump({"tau_v4": tau4, "tau_x": best[1], "valid_f05": best[0], "valid_f05_v4": f0[0],
               "best_iteration": m.best_iteration, "features": feats},
              open(f"{W}/exp/model/decision.json", "w"), indent=1)
    print(f"\nBEST: tau_x={best[1]} -> valid F0.5 {best[0]:.4f}  (v4 alone {f0[0]:.4f}, "
          f"gain {best[0] - f0[0]:+.4f})  [{(time.time() - t0) / 60:.1f} min]")


# ----------------------------------------------------------------------------- 5. predict (test)
def cmd_predict(a):
    import lightgbm as lgb
    t0 = time.time()
    W = a.work
    dec = json.load(open(f"{W}/exp/model/decision.json"))
    m = lgb.Booster(model_file=f"{W}/exp/model/lgbm.txt")
    x = load_exp_pairs(W, "test")
    missing = [f for f in dec["features"] if f not in x.columns]
    if missing:
        raise SystemExit(f"test expansion features missing: {missing}")
    xo = x[ID_COLS].copy()
    xo["p"] = m.predict(x[dec["features"]].astype(np.float32), num_iteration=dec.get("best_iteration"))
    del x
    xo["is_x"] = True
    base = pd.read_parquet(f"{W}/exp/test_scored.parquet")[ID_COLS + ["p"]]
    base["is_x"] = False
    both = pd.concat([base, xo], ignore_index=True)
    keep = joint_decide(both, dec["tau_v4"], dec["tau_x"])
    log(t0, f"v4 pairs {len(base):,} + expansion pairs {len(xo):,}; kept {int(keep.sum()):,} "
            f"(of which expansion {int((keep & both.is_x.to_numpy()).sum()):,})")
    kept = both.loc[keep, ID_COLS]
    cand = both[ID_COLS]
    write_submission(W, a.out_dir, kept, cand, a.mem, a.threads)
    log(t0, "done")


def write_submission(W, out_dir, kept, cand, mem, threads):
    con = duckdb.connect()
    con.execute(f"SET threads={threads}; SET memory_limit='{mem}';")
    con.register("kept_df", kept)
    con.register("cand_df", cand)
    con.execute(f"""CREATE TABLE ids AS
        SELECT 1 AS src, CAST(substr(entity_id,4) AS BIGINT) AS id, entity_id FROM read_parquet('{W}/l0/test_s1/part-*.parquet')
        UNION ALL SELECT 2, CAST(substr(entity_id,4) AS BIGINT), entity_id FROM read_parquet('{W}/l0/test_s2/part-*.parquet')
        UNION ALL SELECT 3, CAST(substr(entity_id,4) AS BIGINT), entity_id FROM read_parquet('{W}/l0/test_s3/part-*.parquet')""")
    s1 = con.execute(f"""SELECT entity_id, CAST(substr(entity_id,4) AS BIGINT) AS q
                         FROM read_parquet('{W}/l0/test_s1/part-*.parquet')""").df()
    os.makedirs(out_dir, exist_ok=True)
    for fname, header, table in (("matching_results.tsv", "matched_entity_ids", "kept_df"),
                                 ("candidate_pairs.tsv", "candidate_entity_ids", "cand_df")):
        d = con.execute(f"""SELECT t.q, string_agg(i.entity_id, ',') AS ids FROM {table} t
                            JOIN ids i ON i.src = t.src AND i.id = t.c GROUP BY t.q""").df()
        mp = dict(zip(d.q, d.ids))
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8", newline="\n") as f:
            f.write(f"source1_entity_id\t{header}\n")
            for eid, q in zip(s1.entity_id, s1.q):
                f.write(f"{eid}\t{mp.get(q, '')}\n")
        print(f"wrote {out_dir}/{fname}", flush=True)
    nq = len(s1)
    print(f"candidate set: {len(cand):,} pairs = {len(cand) / nq:.2f} per S1 entity | "
          f"{len(kept):,} predicted matches for {kept.q.nunique():,} entities "
          f"({1 - kept.q.nunique() / nq:.3f} predicted singletons)")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    pa = sp.add_parser("anchors")
    pa.add_argument("--split", required=True, choices=["train", "test"])
    pa.add_argument("--truth")
    pn = sp.add_parser("neighbors")
    pn.add_argument("--split", required=True, choices=["train", "test"])
    pn.add_argument("--k", type=int, default=8, help="neighbours per anchor per target source per field")
    pn.add_argument("--min-score", type=float, default=0.6)
    pn.add_argument("--q-chunk", type=int, default=1024)
    pn.add_argument("--db-block", type=int, default=400_000)
    pn.add_argument("--device", default="cuda")
    pp = sp.add_parser("pairs")
    pp.add_argument("--split", required=True, choices=["train", "test"])
    pp.add_argument("--truth")
    pp.add_argument("--k", type=int, default=8)
    pp.add_argument("--min-score", type=float, default=0.75, help="min anchor->candidate cosine")
    pp.add_argument("--per-src", type=int, default=4, help="max new candidates per S1 per source")
    pp.add_argument("--mem", default="6GB")
    pt = sp.add_parser("train")
    pt.add_argument("--truth", required=True)
    pt.add_argument("--valid-permille", type=int, default=100)
    pr = sp.add_parser("predict")
    pr.add_argument("--out-dir", default="output")
    pr.add_argument("--mem", default="6GB")
    for x in (pa, pn, pp, pt, pr):
        x.add_argument("--work", default="work")
        x.add_argument("--threads", type=int, default=10)
    a = ap.parse_args()
    {"anchors": cmd_anchors, "neighbors": cmd_neighbors, "pairs": cmd_pairs, "train": cmd_train,
     "predict": cmd_predict}[a.cmd](a)


if __name__ == "__main__":
    main()
