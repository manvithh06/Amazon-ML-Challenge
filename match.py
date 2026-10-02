-#!/usr/bin/env python3
"""
Layer 3 + Layer 5 - LightGBM matcher, F0.5-optimal decision rule, submission writer

  train   : trains LightGBM on work/pairs/train (queries split 90/10 by id into fit/valid),
            then searches the decision rule on the valid queries to MAXIMISE the official metric
            (macro F0.5 over S1 entities, singletons included, computed against the FULL ground
            truth - so true pairs the blocker missed count as misses):
                keep pair  if  p >= tau  and  p >= rho * (best p of that query)
                then one-to-one: an S2/S3 record is kept only for the S1 entity that scores it highest
            saves work/model/lgbm.txt + work/model/decision.json
  predict : scores work/pairs/test, applies the saved rule, writes
            output/matching_results.tsv and output/candidate_pairs.tsv
            (candidate_pairs = every pair the model scored = the exact model input)

    python match.py train --truth student_resource/dataset/train/train_ground_truth.tsv
    python match.py prune-train --truth student_resource/dataset/train/train_ground_truth.tsv
    python match.py predict --prune --out-dir output      # stage-1 pruner + matcher
    python3 student_resource/utils/validate_submission.py --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv --test-dir student_resource/dataset/test
"""
import argparse
import glob
import json
import os
import time

import duckdb
import numpy as np
import pandas as pd

ID_COLS = ["q", "src", "c"]


def feature_cols(df):
    return [c for c in df.columns if c not in ("q", "c", "label")]


def load_truth(path, qs):
    t = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)
    t["q"] = t.source1_entity_id.str.slice(3).astype("int64")
    t = t[t.q.isin(qs)]
    n_true = t.set_index("q").matched_entity_ids.map(lambda s: len([x for x in s.split(",") if x]))
    return n_true.reindex(qs).fillna(0).astype(np.int32)       # every valid query, 0 = singleton


def decide(df, tau, rho):
    """df: q, src, c, p -> boolean mask of predicted matches."""
    keep = df.p.to_numpy() >= tau
    if rho > 0:
        best = df.p.groupby(df.q).transform("max").to_numpy()
        keep &= df.p.to_numpy() >= rho * best
    sub = df[keep]
    if len(sub):
        # one-to-one: each S2/S3 record goes to the S1 entity that scores it highest
        top = sub.sort_values(["p", "q"], ascending=[False, True]).drop_duplicates(["src", "c"])
        keep = np.zeros(len(df), bool)
        keep[df.index.get_indexer(top.index)] = True
    return keep


def macro_f05(df, keep, n_true):
    """Official metric: per-S1 F0.5 = 1.25 TP / (0.25 |Y| + |Yhat|), both empty -> 1, averaged."""
    k = df[keep]
    tp = k.label.groupby(k.q).sum().reindex(n_true.index).fillna(0).to_numpy()
    npred = k.groupby("q").size().reindex(n_true.index).fillna(0).to_numpy()
    nt = n_true.to_numpy()
    den = 0.25 * nt + npred
    f = np.where(den == 0, 1.0, 1.25 * tp / np.maximum(den, 1e-12))
    prec = tp.sum() / max(npred.sum(), 1)
    rec = tp.sum() / max(nt.sum(), 1)
    single = nt == 0
    return f.mean(), prec, rec, f[single].mean() if single.any() else np.nan, f[~single].mean()


def cmd_train(a):
    import lightgbm as lgb
    t0 = time.time()
    parts = []
    for p in sorted(glob.glob(f"{a.work}/pairs/train/part-*.parquet")):
        d = pd.read_parquet(p)
        h = d.q.to_numpy() * 2654435761 % 1000
        # keep every valid query; keep only --fit-frac of the fit queries (memory control)
        keep = (h < a.valid_permille) | ((h - a.valid_permille) < (1000 - a.valid_permille) * a.fit_frac)
        parts.append(d[keep])
        del d
    df = pd.concat(parts, ignore_index=True)
    del parts
    feats = feature_cols(df)
    is_val = (df.q.to_numpy() * 2654435761 % 1000) < a.valid_permille
    print(f"{len(df):,} pairs, {df.label.mean():.3f} positive, {len(feats)} features | "
          f"fit {int((~is_val).sum()):,}  valid {int(is_val.sum()):,}", flush=True)

    X = df[feats].astype(np.float32, copy=False)
    dtr = lgb.Dataset(X[~is_val], df.label[~is_val], free_raw_data=True)
    dva = lgb.Dataset(X[is_val], df.label[is_val], reference=dtr)
    params = dict(objective="binary", learning_rate=a.lr, num_leaves=a.leaves, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  max_bin=255, num_threads=a.threads, verbose=-1, metric=["binary_logloss", "auc"])
    model = lgb.train(params, dtr, num_boost_round=a.rounds, valid_sets=[dva], valid_names=["valid"],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    os.makedirs(f"{a.work}/model", exist_ok=True)
    model.save_model(f"{a.work}/model/lgbm.txt")
    imp = pd.Series(model.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print("\ntop features (gain):\n" + (imp.head(20) / imp.sum()).round(4).to_string())

    va = df.loc[is_val, ID_COLS + ["label"]].copy()
    va["p"] = model.predict(X[is_val], num_iteration=model.best_iteration)
    va = va.reset_index(drop=True)
    # the valid query universe = every sampled S1 query in the valid slice (even with no candidates)
    meta = pd.read_parquet(f"{a.work}/emb_name/train_s1/meta.parquet", columns=["entity_id"])
    all_q = meta.entity_id.str.slice(3).astype("int64").to_numpy()
    val_q = all_q[(all_q * 2654435761 % 1000) < a.valid_permille]
    n_true = load_truth(a.truth, val_q)
    print(f"\nvalid: {len(val_q):,} queries, {int(n_true.sum()):,} true pairs, "
          f"{(n_true == 0).mean():.3f} singletons, blocking recall "
          f"{va.label.sum() / max(n_true.sum(), 1):.4f}")

    best = None
    print(f"{'tau':>5} {'rho':>5} | {'F0.5':>7} {'prec':>6} {'rec':>6} {'F_single':>8} {'F_nonsingle':>11}")
    for rho in (0.0, 0.3, 0.5, 0.7):
        for tau in np.round(np.arange(0.20, 0.96, 0.05), 2):
            keep = decide(va, tau, rho)
            f, p, r, fs, fn = macro_f05(va, keep, n_true)
            if best is None or f > best[0]:
                best = (f, float(tau), float(rho))
            if tau in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9) or (best[1] == tau and best[2] == rho):
                print(f"{tau:5.2f} {rho:5.2f} | {f:7.4f} {p:6.3f} {r:6.3f} {fs:8.3f} {fn:11.3f}")
    f, tau, rho = best
    # refine tau around the best value
    for t in np.round(np.arange(max(0.05, tau - 0.05), min(0.99, tau + 0.05) + 1e-9, 0.01), 2):
        ff = macro_f05(va, decide(va, t, rho), n_true)[0]
        if ff > f:
            f, tau = ff, float(t)
    json.dump({"tau": tau, "rho": rho, "valid_f05": f, "best_iteration": model.best_iteration,
               "features": feats}, open(f"{a.work}/model/decision.json", "w"), indent=1)
    va.to_parquet(f"{a.work}/model/valid_preds.parquet", index=False)
    print(f"\nBEST: tau={tau:.2f} rho={rho:.2f} -> valid macro F0.5 = {f:.4f}   "
          f"(saved to {a.work}/model/)  [{(time.time() - t0) / 60:.1f} min]")


PRUNE_FEATS = ["src", "name_score", "name_rank", "addr_score", "addr_rank", "lex_name", "lex_addr", "lex_rank",
               "cos_name", "cos_addr", "cos_sum", "cos_name_gap", "cos_name_rk", "cos_name_gap_src",
               "cos_addr_gap", "cos_addr_rk", "cos_addr_gap_src", "cos_sum_gap", "cos_sum_rk", "cos_sum_gap_src",
               "n_cand", "n_cand_src"]


# optional extra stage-1 inputs: a few cheap string similarities (the main model's strongest signals)
PRUNE_RICH = ["a_tset", "a_tset_gap_src", "a_tset_gap", "n_partial", "n_tset", "n_tset_gap", "n_jw",
              "num_jacc", "num_match"]


def prune_keep(q, pp, top_n, min_pp):
    """Stage-1 rule: per query keep the top_n candidates by pruner score, and only if pp >= min_pp."""
    d = pd.DataFrame({"q": q, "pp": pp})
    rk = d.groupby("q").pp.rank(ascending=False, method="first").to_numpy()
    return (rk <= top_n) & (pp >= min_pp)


def cmd_prune_train(a):
    """Stage-1 candidate pruner: a small LightGBM on CHEAP features only (blocker ranks/scores and embedding
    cosines - no string similarity), so it can run on the full blocked set; only its survivors are passed to
    the full matcher. Picks the rule that shrinks the candidate set most at (almost) no F0.5 cost."""
    import lightgbm as lgb
    t0 = time.time()
    feats = PRUNE_FEATS + (PRUNE_RICH if a.rich else [])
    cols = ID_COLS + ["label"] + [f for f in feats if f not in ID_COLS]
    df = pd.concat([pd.read_parquet(p, columns=cols) for p in sorted(glob.glob(f"{a.work}/pairs/train/part-*.parquet"))],
                   ignore_index=True)
    is_val = (df.q.to_numpy() * 2654435761 % 1000) < a.valid_permille
    X = df[feats].astype(np.float32)
    params = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9,
                  bagging_fraction=0.8, bagging_freq=1, num_threads=a.threads, verbose=-1, metric="binary_logloss")
    m = lgb.train(params, lgb.Dataset(X[~is_val], df.label[~is_val]), num_boost_round=300)
    m.save_model(f"{a.work}/model/pruner.txt")

    va = pd.read_parquet(f"{a.work}/model/valid_preds.parquet")          # q, src, c, label, p (full matcher)
    vf = df.loc[is_val, ID_COLS].copy()
    vf["pp"] = m.predict(X[is_val])
    va = va.merge(vf, on=ID_COLS, how="left").reset_index(drop=True)
    dec = json.load(open(f"{a.work}/model/decision.json"))
    meta = pd.read_parquet(f"{a.work}/emb_name/train_s1/meta.parquet", columns=["entity_id"])
    all_q = meta.entity_id.str.slice(3).astype("int64").to_numpy()
    n_true = load_truth(a.truth, all_q[(all_q * 2654435761 % 1000) < a.valid_permille])
    nq = len(n_true)

    base = macro_f05(va, decide(va, dec["tau"], dec["rho"]), n_true)[0]
    print(f"no pruning: {len(va) / nq:.1f} cand/query, F0.5 {base:.4f}\n")
    print(f"{'top_n':>5} {'min_pp':>7} | {'cand/q':>6} {'pos kept':>8} | {'F0.5':>7} {'best tau':>8}")
    rows = []
    for top_n in (3, 4, 5, 6, 8, 10, 15):
        for min_pp in (0.0, 0.001, 0.005, 0.02, 0.05):
            keep = prune_keep(va.q.to_numpy(), va.pp.to_numpy(), top_n, min_pp)
            sub = va[keep].reset_index(drop=True)
            best = max((macro_f05(sub, decide(sub, t, dec["rho"]), n_true)[0], t)
                       for t in np.round(np.arange(0.4, 0.91, 0.03), 2))
            rows.append((top_n, min_pp, keep.sum() / nq, va.label[keep].sum() / va.label.sum(), best[0], best[1]))
            print(f"{top_n:5d} {min_pp:7.3f} | {rows[-1][2]:6.2f} {rows[-1][3]:8.4f} | {best[0]:7.4f} {best[1]:8.2f}")
    # smallest candidate set whose F0.5 is within --max-loss of the unpruned model
    if a.top_n is not None:                              # rule chosen by hand
        pick = [r for r in rows if r[0] == a.top_n and abs(r[1] - a.min_pp) < 1e-12]
        if not pick:
            raise SystemExit(f"top_n={a.top_n} min_pp={a.min_pp} is not in the grid above")
        top_n, min_pp, cpq, kept, f, tau = pick[0]
    else:
        ok = [r for r in rows if r[4] >= base - a.max_loss]
        if not ok:
            print(f"\nno rule within {a.max_loss} of the unpruned F0.5 - taking the best-scoring rule")
            ok = [max(rows, key=lambda r: r[4])]
        top_n, min_pp, cpq, kept, f, tau = min(ok, key=lambda r: r[2])
    json.dump({"top_n": top_n, "min_pp": min_pp, "tau": float(tau), "rho": dec["rho"], "valid_f05": f,
               "cand_per_query": cpq, "features": feats}, open(f"{a.work}/model/prune.json", "w"), indent=1)
    print(f"\nCHOSEN: top_n={top_n} min_pp={min_pp} -> {cpq:.2f} cand/query (was {len(va) / nq:.1f}), "
          f"F0.5 {f:.4f} (unpruned {base:.4f}), tau={tau}  [{(time.time() - t0) / 60:.1f} min]")


def cmd_predict(a):
    import lightgbm as lgb
    t0 = time.time()
    model = lgb.Booster(model_file=f"{a.work}/model/lgbm.txt")
    dec = json.load(open(f"{a.work}/model/decision.json"))
    feats, tau, rho = dec["features"], dec["tau"], dec["rho"]
    pruner = None
    if a.prune:
        pj = json.load(open(f"{a.work}/model/prune.json"))
        pruner = lgb.Booster(model_file=f"{a.work}/model/pruner.txt")
        tau, rho = pj["tau"], pj["rho"]
        print(f"stage-1 pruner: top_n={pj['top_n']} min_pp={pj['min_pp']} "
              f"(valid {pj['cand_per_query']:.2f} cand/query)", flush=True)
    print(f"decision rule: tau={tau} rho={rho}", flush=True)

    pred_dir = f"{a.work}/preds_test"
    import shutil
    shutil.rmtree(pred_dir, ignore_errors=True)
    os.makedirs(pred_dir, exist_ok=True)
    parts = sorted(glob.glob(f"{a.work}/pairs/test/part-*.parquet"))
    for i, pth in enumerate(parts):
        df = pd.read_parquet(pth)
        missing = [f for f in feats if f not in df.columns]
        if missing:
            raise SystemExit(f"test features missing: {missing}")
        if pruner is not None:          # only stage-1 survivors reach the matcher (= candidate_pairs.tsv)
            pp = pruner.predict(df[pj["features"]].astype(np.float32))
            df = df[prune_keep(df.q.to_numpy(), pp, pj["top_n"], pj["min_pp"])]
        out = df[ID_COLS].copy()
        out["p"] = model.predict(df[feats].astype(np.float32), num_iteration=dec.get("best_iteration"))
        out.to_parquet(f"{pred_dir}/part-{i:03d}.parquet", index=False)
        print(f"scored part {i + 1}/{len(parts)} ({len(df):,} pairs)", flush=True)

    con = duckdb.connect()
    con.execute(f"SET threads={a.threads}; SET memory_limit='{a.mem}';")
    W = a.work
    con.execute(f"CREATE TABLE pr AS SELECT * FROM read_parquet('{pred_dir}/part-*.parquet')")
    # original entity-id strings (never rebuild ids from numbers - avoids any leading-zero issue)
    con.execute(f"""CREATE TABLE ids AS
        SELECT 1 AS src, CAST(substr(entity_id,4) AS BIGINT) AS id, entity_id FROM read_parquet('{W}/l0/test_s1/part-*.parquet')
        UNION ALL SELECT 2, CAST(substr(entity_id,4) AS BIGINT), entity_id FROM read_parquet('{W}/l0/test_s2/part-*.parquet')
        UNION ALL SELECT 3, CAST(substr(entity_id,4) AS BIGINT), entity_id FROM read_parquet('{W}/l0/test_s3/part-*.parquet')""")
    dup = con.execute("SELECT count(*) FROM (SELECT src, id FROM ids GROUP BY 1, 2 HAVING count(*) > 1)").fetchone()[0]
    if dup:
        raise SystemExit(f"{dup} numeric id collisions - ids cannot be mapped back safely")
    con.execute(f"""CREATE TABLE kept AS
        WITH a AS (SELECT *, max(p) OVER (PARTITION BY q) AS best FROM pr WHERE p >= {tau}),
             b AS (SELECT * FROM a WHERE p >= {rho} * best),
             c AS (SELECT *, row_number() OVER (PARTITION BY src, c ORDER BY p DESC, q) AS r FROM b)
        SELECT q, src, c, p FROM c WHERE r = 1""")
    n_k, n_qk = con.execute("SELECT count(*), count(DISTINCT q) FROM kept").fetchone()
    n_c = con.execute("SELECT count(*) FROM pr").fetchone()[0]

    s1 = con.execute(f"""SELECT entity_id, CAST(substr(entity_id,4) AS BIGINT) AS q
                         FROM read_parquet('{W}/l0/test_s1/part-*.parquet')""").df()

    def agg(table):
        d = con.execute(f"""SELECT t.q, string_agg(i.entity_id, ',') AS ids
                            FROM {table} t JOIN ids i ON i.src = t.src AND i.id = t.c GROUP BY t.q""").df()
        return dict(zip(d.q, d.ids))

    os.makedirs(a.out_dir, exist_ok=True)
    for fname, header, table in (("matching_results.tsv", "matched_entity_ids", "kept"),
                                 ("candidate_pairs.tsv", "candidate_entity_ids", "pr")):
        m = agg(table)
        with open(os.path.join(a.out_dir, fname), "w", encoding="utf-8", newline="\n") as f:
            f.write(f"source1_entity_id\t{header}\n")
            for eid, q in zip(s1.entity_id, s1.q):
                f.write(f"{eid}\t{m.get(q, '')}\n")
        print(f"wrote {a.out_dir}/{fname}", flush=True)
    print(f"candidate set: {n_c:,} pairs = {n_c / len(s1):.2f} per S1 entity")
    print(f"{len(s1):,} S1 entities | {n_k:,} predicted matches for {n_qk:,} entities "
          f"({1 - n_qk / len(s1):.3f} predicted singletons) | {(time.time() - t0) / 60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train")
    t.add_argument("--truth", required=True)
    t.add_argument("--valid-permille", type=int, default=100, help="share of queries held out (100 = 10%%)")
    t.add_argument("--rounds", type=int, default=1500)
    t.add_argument("--lr", type=float, default=0.08)
    t.add_argument("--leaves", type=int, default=255)
    t.add_argument("--fit-frac", type=float, default=1.0,
                   help="use only this fraction of the training (non-valid) queries, to save RAM")
    r = sp.add_parser("prune-train")
    r.add_argument("--truth", required=True)
    r.add_argument("--valid-permille", type=int, default=100)
    r.add_argument("--max-loss", type=float, default=0.001, help="allowed F0.5 drop vs. no pruning")
    r.add_argument("--top-n", type=int, default=None, help="force this rule instead of choosing automatically")
    r.add_argument("--min-pp", type=float, default=0.0)
    r.add_argument("--rich", action="store_true", help="also give the pruner cheap string-similarity features")
    p = sp.add_parser("predict")
    p.add_argument("--out-dir", default="output")
    p.add_argument("--prune", action="store_true", help="apply the stage-1 pruner (see prune-train)")
    p.add_argument("--mem", default="8GB")
    for x in (t, r, p):
        x.add_argument("--work", default="work")
        x.add_argument("--threads", type=int, default=12)
    a = ap.parse_args()
    {"train": cmd_train, "prune-train": cmd_prune_train, "predict": cmd_predict}[a.cmd](a)


if __name__ == "__main__":
    main()
