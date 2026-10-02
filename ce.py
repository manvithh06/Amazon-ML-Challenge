#!/usr/bin/env python3
"""
Layer 5 - CROSS-ENCODER re-scoring + stacker

The LightGBM matcher (v4) sees ~57 hand-made similarity numbers.  It rejects ~4 of every 100 true pairs
that ARE in the candidate set (valid recall 0.901 vs 0.943 available): transliterations, website-style
names, abbreviations, heavy address noise.  A cross-encoder reads BOTH records as one token sequence
    "<S1 name> | <S1 address> [SEP] <candidate name> | <candidate address>"
and learns these patterns directly.

Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0, 118M params, already
cached from Layer 1) + a fresh 1-logit classification head, fine-tuned on the train-sample candidate
pairs of the FIT queries (the 90% hash split used everywhere).  It only scores the pruned candidate set
(v4 stage-1 survivors), so candidate_pairs.tsv does not grow.

  data    : text for train pairs (pruned v4 survivors, work/exp/train_scored.parquet) and test pairs
  train   : fine-tune on fit-query pairs (1 epoch, fp16), report AUC on valid-query pairs
  score   : cross-encoder probability for valid pairs (--split valid) or all test pairs (--split test)
  stack   : small LightGBM on the VALID queries: v4 p + ce p + "how does this pair rank inside its query"
            features, 5-fold by query -> out-of-fold F0.5 vs v4, tunes the threshold, saves the stacker
  predict : test -> stacker -> F0.5 rule + one-to-one -> output/*.tsv

    python ce.py data
    python ce.py train --max-train 800000
    python ce.py score --split valid
    python ce.py stack --truth student_resource/dataset/train/train_ground_truth.tsv
    python ce.py score --split test
    python ce.py predict --out-dir output
"""
import argparse
import glob
import json
import math
import os
import time

import duckdb
import numpy as np
import pandas as pd

MODEL = os.environ.get("CE_BASE", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
VALID_HASH = 2654435761
ID_COLS = ["q", "src", "c"]


def is_valid(q, permille=100):
    return (np.asarray(q) * VALID_HASH % 1000) < permille


def log(t0, msg):
    print(f"[{time.time() - t0:7.0f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------- data
def text_db(W, split, tmp):
    """On-disk DuckDB with id -> 'name | address' for S1/S2/S3 of a split."""
    f = f"{tmp}/ce_text_{split}.duckdb"
    for x in (f, f + ".wal"):
        if os.path.exists(x):
            os.remove(x)
    con = duckdb.connect(f)
    con.execute(f"SET memory_limit='4GB'; SET temp_directory='{tmp}'; SET preserve_insertion_order=false;")
    con.execute(f"""CREATE TABLE txt AS
        SELECT s AS src, CAST(substr(entity_id, 4) AS BIGINT) AS id,
               trim(coalesce(name_emb, '')) || ' | ' || trim(coalesce(addr_emb, '')) AS t
        FROM (SELECT 1 AS s, * FROM read_parquet('{W}/l0/{split}_s1/part-*.parquet')
              UNION ALL BY NAME SELECT 2 AS s, * FROM read_parquet('{W}/l0/{split}_s2/part-*.parquet')
              UNION ALL BY NAME SELECT 3 AS s, * FROM read_parquet('{W}/l0/{split}_s3/part-*.parquet'))""")
    return con


def attach_text(con, df):
    con.register("p_df", df)
    out = con.execute("""SELECT p.*, a.t AS ta, b.t AS tb FROM p_df p
                         JOIN txt a ON a.src = 1 AND a.id = p.q
                         JOIN txt b ON b.src = p.src AND b.id = p.c""").df()
    con.unregister("p_df")
    return out


def cmd_data(a):
    t0 = time.time()
    W = a.work
    os.makedirs(f"{W}/ce", exist_ok=True)
    tmp = f"{W}/duck_tmp"
    os.makedirs(tmp, exist_ok=True)
    con = text_db(W, "train", tmp)
    sc = pd.read_parquet(f"{W}/exp/train_scored.parquet")          # pruned v4 survivors: q, src, c, label, p
    sc["src"] = sc.src.astype(np.int8)
    d = attach_text(con, sc)
    v = is_valid(d.q)
    d[~v].sample(frac=1.0, random_state=0).to_parquet(f"{W}/ce/fit.parquet", index=False)
    d[v].to_parquet(f"{W}/ce/valid.parquet", index=False)
    log(t0, f"train pairs with text: fit {int((~v).sum()):,} ({d.label[~v].mean():.3f} positive) | "
            f"valid {int(v.sum()):,}")
    ex = d[d.label == 1].sample(5, random_state=1)
    for r in ex.itertuples():
        print(f"   + {r.ta[:70]}  <->  {r.tb[:70]}")
    con.close()
    con = text_db(W, "test", tmp)
    parts = sorted(glob.glob(f"{W}/preds_test/part-*.parquet"))
    os.makedirs(f"{W}/ce/test_text", exist_ok=True)
    n = 0
    for i, p in enumerate(parts):
        df = pd.read_parquet(p)
        df["src"] = df.src.astype(np.int8)
        out = attach_text(con, df)
        out.to_parquet(f"{W}/ce/test_text/part-{i:03d}.parquet", index=False)
        n += len(out)
    log(t0, f"test pairs with text: {n:,} in {len(parts)} parts")
    con.close()
    for x in glob.glob(f"{tmp}/ce_text_*.duckdb*"):
        os.remove(x)


# ----------------------------------------------------------------------------- model helpers
def load_model(path_or_name, device):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path_or_name)
    model = AutoModelForSequenceClassification.from_pretrained(path_or_name, num_labels=1)
    return tok, model.to(device)


def batches_by_length(ta, tb, bs):
    """Sort by length so each batch pads little; returns list of index arrays."""
    ln = np.array([len(x) + len(y) for x, y in zip(ta, tb)])
    order = np.argsort(ln, kind="stable")
    return [order[i:i + bs] for i in range(0, len(order), bs)]


def predict_probs(tok, model, ta, tb, bs, max_len, device):
    import torch
    model.eval()
    out = np.empty(len(ta), np.float32)
    with torch.no_grad():
        for idx in batches_by_length(ta, tb, bs):
            enc = tok([ta[i] for i in idx], [tb[i] for i in idx], truncation="longest_first",
                      max_length=max_len, padding=True, return_tensors="pt").to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                logit = model(**enc).logits.squeeze(-1)
            out[idx] = torch.sigmoid(logit.float()).cpu().numpy()
    return out


# ----------------------------------------------------------------------------- train
def cmd_train(a):
    import torch
    from sklearn.metrics import roc_auc_score
    from transformers import get_linear_schedule_with_warmup
    t0 = time.time()
    W = a.work
    torch.manual_seed(a.seed)
    fit = pd.read_parquet(f"{W}/ce/fit.parquet")
    if a.seed:
        fit = fit.sample(frac=1.0, random_state=a.seed)
    if a.max_train and len(fit) > a.max_train:
        fit = fit.iloc[:a.max_train]
    val = pd.read_parquet(f"{W}/ce/valid.parquet")
    val = val.sample(min(len(val), a.eval_n), random_state=0)
    tok, model = load_model(MODEL, a.device)
    ta, tb = fit.ta.tolist(), fit.tb.tolist()
    y = fit.label.to_numpy(np.float32)
    # random order but length-bucketed: shuffle buckets of similar length
    rng = np.random.default_rng(a.seed)
    batches = batches_by_length(ta, tb, a.batch)
    rng.shuffle(batches)
    steps = len(batches) * a.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.cuda.amp.GradScaler(enabled=a.device == "cuda")
    lossf = torch.nn.BCEWithLogitsLoss()
    log(t0, f"fine-tuning on {len(fit):,} pairs ({y.mean():.3f} positive), {steps:,} steps of {a.batch}")
    step = 0
    for ep in range(a.epochs):
        model.train()
        run = 0.0
        for idx in batches:
            enc = tok([ta[i] for i in idx], [tb[i] for i in idx], truncation="longest_first",
                      max_length=a.max_len, padding=True, return_tensors="pt").to(a.device)
            lab = torch.from_numpy(y[idx]).to(a.device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=a.device == "cuda"):
                logit = model(**enc).logits.squeeze(-1)
            loss = lossf(logit.float(), lab)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
            step += 1
            run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
            if step % 500 == 0 or step == steps:
                eta = (time.time() - t0) / step * (steps - step) / 60
                log(t0, f"step {step:,}/{steps:,}  loss {run:.4f}  (~{eta:.0f} min left)")
            if step % a.eval_every == 0 or step == steps:
                p = predict_probs(tok, model, val.ta.tolist(), val.tb.tolist(), 256, a.max_len, a.device)
                log(t0, f"   valid AUC {roc_auc_score(val.label, p):.5f}  "
                        f"acc@0.5 {np.mean((p >= 0.5) == val.label):.4f}")
                model.train()
    os.makedirs(f"{W}/ce/{a.model_dir}", exist_ok=True)
    model.save_pretrained(f"{W}/ce/{a.model_dir}")
    tok.save_pretrained(f"{W}/ce/{a.model_dir}")
    log(t0, f"saved -> {W}/ce/{a.model_dir}")


# ----------------------------------------------------------------------------- score
def cmd_score(a):
    """--model-dir model --col ce  -> the original files (valid_scored.parquet, test_scored/)
       any other --col (e.g. ce2)  -> side files valid_scored_<col>.parquet, test_scored_<col>/ with q,src,c,<col>"""
    t0 = time.time()
    W = a.work
    tok, model = load_model(f"{W}/ce/{a.model_dir}", a.device)
    main_ = a.col == "ce"
    if a.split == "valid":
        d = pd.read_parquet(f"{W}/ce/valid.parquet")
        d[a.col] = predict_probs(tok, model, d.ta.tolist(), d.tb.tolist(), a.batch, a.max_len, a.device)
        if main_:
            d.drop(columns=["ta", "tb"]).to_parquet(f"{W}/ce/valid_scored.parquet", index=False)
        else:
            d[ID_COLS + [a.col]].to_parquet(f"{W}/ce/valid_scored_{a.col}.parquet", index=False)
            from sklearn.metrics import roc_auc_score
            print(f"   valid AUC {a.col}: {roc_auc_score(d.label, d[a.col]):.5f}")
        log(t0, f"scored {len(d):,} valid pairs")
        return
    od = f"{W}/ce/test_scored" if main_ else f"{W}/ce/test_scored_{a.col}"
    os.makedirs(od, exist_ok=True)
    parts = sorted(glob.glob(f"{W}/ce/test_text/part-*.parquet"))
    done0 = len(glob.glob(f"{od}/part-*.parquet"))
    for i, p in enumerate(parts):
        out = f"{od}/part-{i:03d}.parquet"
        if os.path.exists(out):                       # resumable
            continue
        d = pd.read_parquet(p)
        d[a.col] = predict_probs(tok, model, d.ta.tolist(), d.tb.tolist(), a.batch, a.max_len, a.device)
        (d.drop(columns=["ta", "tb"]) if main_ else d[ID_COLS + [a.col]]).to_parquet(out, index=False)
        el = time.time() - t0
        n_done = i + 1 - done0
        log(t0, f"test part {i + 1}/{len(parts)} ({len(d):,} pairs)  "
                f"~{el / max(n_done, 1) * (len(parts) - i - 1) / 60:.0f} min left")


# ----------------------------------------------------------------------------- stack
def stack_features(d, score_cols=("p", "ce")):
    d = d.copy()
    for col in score_cols:
        v = d[col]
        d[f"{col}_rk"] = v.groupby(d.q).rank(ascending=False, method="min").astype(np.float32)
        d[f"{col}_gap"] = (v.groupby(d.q).transform("max") - v).astype(np.float32)
        d[f"{col}_rk_src"] = v.groupby([d.q, d.src]).rank(ascending=False, method="min").astype(np.float32)
        d[f"{col}_gap_src"] = (v.groupby([d.q, d.src]).transform("max") - v).astype(np.float32)
        d[f"{col}_n05"] = (v >= 0.5).groupby(d.q).transform("sum").astype(np.float32)
    d["n_cand"] = d.groupby("q").q.transform("size").astype(np.float32)
    d["pce"] = (d.p * d.ce).astype(np.float32)
    extra = [c for c in score_cols if c not in ("p", "ce")]
    if extra:
        d["ce_mean"] = d[["ce"] + extra].mean(axis=1).astype(np.float32)
        d["pce_mean"] = (d.p * d.ce_mean).astype(np.float32)
    return d


def stack_feature_names(extra):
    cols = ["p", "ce"] + list(extra)
    names = ["p", "ce", "src", "pce", "n_cand"] + list(extra) + \
        [f"{c}_{s}" for c in cols for s in ("rk", "gap", "rk_src", "gap_src", "n05")]
    if extra:
        names += ["ce_mean", "pce_mean"]
    return names


def v4_features(W):
    return [f for f in json.load(open(f"{W}/model/decision.json"))["features"] if f not in ID_COLS]


STACK_FEATS = ["p", "ce", "src", "pce", "n_cand"] + [f"{c}_{s}" for c in ("p", "ce")
                                                     for s in ("rk", "gap", "rk_src", "gap_src", "n05")]


def cmd_stack(a):
    import lightgbm as lgb
    from match import decide, load_truth, macro_f05
    t0 = time.time()
    W = a.work
    base_dir = f"{W}/ce/{a.scored_dir}" if a.scored_dir else f"{W}/ce"
    extra = [c for c in a.extra_ce.split(",") if c]
    d = pd.read_parquet(f"{base_dir}/valid_scored.parquet")
    d["src"] = d.src.astype(np.int8)
    for c in extra:
        x = pd.read_parquet(f"{W}/ce/valid_scored_{c}.parquet")
        x["src"] = x.src.astype(np.int8)
        n0 = len(d)
        d = d.merge(x, on=ID_COLS, how="left")
        assert len(d) == n0 and d[c].notna().all(), f"valid scores for {c} missing - run score --split valid --col {c}"
    d = stack_features(d, ["p", "ce"] + extra).reset_index(drop=True)
    feats = stack_feature_names(extra)
    if a.rich:                      # + all 57 v4 pair features (prefixed f_), taken from work/pairs/train
        v4f = v4_features(W)
        parts = []
        for pth in sorted(glob.glob(f"{W}/pairs/train/part-*.parquet")):
            x = pd.read_parquet(pth, columns=ID_COLS + v4f)
            parts.append(x[is_valid(x.q)])
        x = pd.concat(parts, ignore_index=True).rename(columns={f: "f_" + f for f in v4f})
        x["src"] = x.src.astype(np.int8)
        d["src"] = d.src.astype(np.int8)
        n0 = len(d)
        d = d.merge(x, on=ID_COLS, how="left").reset_index(drop=True)
        assert len(d) == n0, "duplicate pairs while joining v4 features"
        feats += ["f_" + f for f in v4f]
        print(f"rich stacker: {len(feats)} features ({len(v4f)} v4 pair features added)")
    meta = pd.read_parquet(f"{W}/emb_name/train_s1/meta.parquet", columns=["entity_id"])
    all_q = meta.entity_id.str.slice(3).astype("int64").to_numpy()
    n_true = load_truth(a.truth, all_q[is_valid(all_q)])
    tau4 = json.load(open(f"{W}/model/prune.json"))["tau"]
    base = macro_f05(d, decide(d, tau4, 0.0), n_true)
    ced = d.assign(p=d.ce)
    ce_only = max((macro_f05(ced, decide(ced, t, 0.0), n_true)[0], t) for t in np.arange(0.3, 0.96, 0.05))
    print(f"valid pairs {len(d):,} | v4 alone: F0.5 {base[0]:.4f} (prec {base[1]:.4f} rec {base[2]:.4f}) | "
          f"cross-encoder alone: {ce_only[0]:.4f} @ {ce_only[1]:.2f}")

    params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=a.threads)
    fold = (d.q.to_numpy() * 40503 % 5)
    oof = np.zeros(len(d))
    for k in range(5):
        tr, te = fold != k, fold == k
        m = lgb.train(params, lgb.Dataset(d.loc[tr, feats].astype(np.float32), d.label[tr]), num_boost_round=a.rounds)
        oof[te] = m.predict(d.loc[te, feats].astype(np.float32))
    s = d.assign(p=oof)
    print(f"\n{'tau':>5} | {'F0.5':>7} {'prec':>6} {'rec':>6} {'F_single':>8} {'F_nonsingle':>11}   (stacker, out-of-fold)")
    best = (0, 0.5)
    for t in np.round(np.arange(0.30, 0.96, 0.02), 2):
        r = macro_f05(s, decide(s, t, 0.0), n_true)
        if r[0] > best[0]:
            best = (r[0], float(t))
        if round(t * 100) % 10 == 0 or t == best[1]:
            print(f"{t:5.2f} | {r[0]:7.4f} {r[1]:6.3f} {r[2]:6.3f} {r[3]:8.3f} {r[4]:11.3f}")
    m = lgb.train(params, lgb.Dataset(d[feats].astype(np.float32), d.label), num_boost_round=a.rounds)
    sd = f"{W}/ce/{a.stack_dir}"
    os.makedirs(sd, exist_ok=True)
    m.save_model(f"{sd}/lgbm.txt")
    imp = pd.Series(m.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print("\ntop stacker features (gain):\n" + (imp.head(12) / imp.sum()).round(4).to_string())
    json.dump({"tau": best[1], "valid_f05_oof": best[0], "valid_f05_v4": base[0], "features": feats,
               "rich": bool(a.rich), "extra_ce": extra}, open(f"{sd}/decision.json", "w"), indent=1)
    print(f"saved -> {sd}")
    print(f"\nBEST: tau={best[1]:.2f} -> stacked valid F0.5 {best[0]:.4f}  (v4 {base[0]:.4f}, "
          f"gain {best[0] - base[0]:+.4f})  [{(time.time() - t0) / 60:.1f} min]")


# ----------------------------------------------------------------------------- predict
def cmd_predict(a):
    import lightgbm as lgb
    from match import decide
    from expand import write_submission
    t0 = time.time()
    W = a.work
    sd = f"{W}/ce/{a.stack_dir}"
    dec = json.load(open(f"{sd}/decision.json"))
    m = lgb.Booster(model_file=f"{sd}/lgbm.txt")
    base_dir = f"{W}/ce/{a.scored_dir}" if a.scored_dir else f"{W}/ce"
    ce_parts = sorted(glob.glob(f"{base_dir}/test_scored/part-*.parquet"))
    pr_parts = sorted(glob.glob(f"{W}/preds_test/part-*.parquet"))
    if len(ce_parts) != len(pr_parts):
        raise SystemExit(f"test scoring incomplete: {len(ce_parts)} of {len(pr_parts)} parts")
    v4f = v4_features(W) if dec.get("rich") else []
    pair_parts = sorted(glob.glob(f"{W}/pairs/test/part-*.parquet"))
    out = []
    for i, cp in enumerate(ce_parts):          # parts hold complete queries, so group features are exact
        d = pd.read_parquet(cp)
        n_pr = len(d) if a.scored_dir else pd.read_parquet(pr_parts[i], columns=["q"]).shape[0]
        if len(d) != n_pr:
            raise SystemExit(f"part {i}: {len(d)} scored vs {n_pr} candidates - rerun 'score --split test'")
        d["src"] = d.src.astype(np.int8)
        if v4f:
            x = pd.read_parquet(pair_parts[i], columns=ID_COLS + v4f).rename(columns={f: "f_" + f for f in v4f})
            x["src"] = x.src.astype(np.int8)
            d = d.merge(x, on=ID_COLS, how="left")
            if len(d) != n_pr:
                raise SystemExit(f"part {i}: feature join changed the row count")
        extra = dec.get("extra_ce", [])
        for c in extra:
            xp = f"{W}/ce/test_scored_{c}/part-{i:03d}.parquet"
            if not os.path.exists(xp):
                raise SystemExit(f"missing {xp} - run: python ce.py score --split test --model-dir ... --col {c}")
            x = pd.read_parquet(xp)
            x["src"] = x.src.astype(np.int8)
            n1 = len(d)
            d = d.merge(x, on=ID_COLS, how="left")
            if len(d) != n1 or d[c].isna().any():
                raise SystemExit(f"part {i}: {c} scores do not cover all pairs")
        d = stack_features(d, ["p", "ce"] + extra).reset_index(drop=True)
        d["p"] = m.predict(d[dec["features"]].astype(np.float32))
        out.append(d[ID_COLS + ["p"]])
    d = pd.concat(out, ignore_index=True)
    keep = decide(d, dec["tau"], 0.0)
    log(t0, f"{len(d):,} candidate pairs, {int(keep.sum()):,} predicted matches (tau {dec['tau']})")
    write_submission(W, a.out_dir, d.loc[keep, ID_COLS], d[ID_COLS], a.mem, a.threads)
    log(t0, "done")


# ----------------------------------------------------------------------------- widen (looser stage-1 pruner)
def cmd_widen(a):
    """Re-apply the v4 stage-1 pruner with a LOOSER rule (e.g. top 10, min_pp 0.005) so more true pairs reach
    the cross-encoder.  Only the NEW pairs are text-joined and cross-encoded; already scored pairs are reused.
    Writes work/ce/<tag>/valid_scored.parquet and work/ce/<tag>/test_scored/part-XXX.parquet (resumable)."""
    import lightgbm as lgb
    t0 = time.time()
    W = a.work
    out = f"{W}/ce/{a.tag}"
    os.makedirs(f"{out}/test_scored", exist_ok=True)
    pj = json.load(open(f"{W}/model/prune.json"))
    dec = json.load(open(f"{W}/model/decision.json"))
    pruner = lgb.Booster(model_file=f"{W}/model/pruner.txt")
    matcher = lgb.Booster(model_file=f"{W}/model/lgbm.txt")
    from match import prune_keep
    tok, model = load_model(f"{W}/ce/model", a.device)
    cols = list(dict.fromkeys(ID_COLS + pj["features"] + dec["features"]))

    def widen_frame(df, old):
        """df: all blocked pairs of complete queries (with features); old: already scored pairs (q,src,c,p,ce)."""
        pp = pruner.predict(df[pj["features"]].astype(np.float32))
        df = df[prune_keep(df.q.to_numpy(), pp, a.top_n, a.min_pp)].copy()
        df["src"] = df.src.astype(np.int8)
        key_old = set(zip(old.q.to_numpy(), old.src.to_numpy(), old.c.to_numpy()))
        is_new = np.fromiter((k not in key_old for k in zip(df.q.to_numpy(), df.src.to_numpy(), df.c.to_numpy())),
                             bool, len(df))
        new = df[is_new]
        keep = ID_COLS + (["label"] if "label" in df.columns else [])
        res = new[keep].copy()
        res["p"] = matcher.predict(new[dec["features"]].astype(np.float32), num_iteration=dec.get("best_iteration")) \
            if len(new) else np.zeros(0)
        return res

    def ce_score(con, res):
        if len(res) == 0:
            res["ce"] = np.zeros(0, np.float32)
            return res
        tx = attach_text(con, res)
        tx["ce"] = predict_probs(tok, model, tx.ta.tolist(), tx.tb.tolist(), a.batch, a.max_len, a.device)
        return tx.drop(columns=["ta", "tb"])

    tmp = f"{W}/duck_tmp"
    os.makedirs(tmp, exist_ok=True)
    # ---- valid queries (train sample)
    if not os.path.exists(f"{out}/valid_scored.parquet"):
        con = text_db(W, "train", tmp)
        old = pd.read_parquet(f"{W}/ce/valid_scored.parquet")
        old["src"] = old.src.astype(np.int8)
        news = []
        for pth in sorted(glob.glob(f"{W}/pairs/train/part-*.parquet")):
            df = pd.read_parquet(pth, columns=cols + ["label"])
            df = df[is_valid(df.q)]
            news.append(widen_frame(df, old))
        new = ce_score(con, pd.concat(news, ignore_index=True))
        con.close()
        allv = pd.concat([old, new[old.columns]], ignore_index=True)
        allv.to_parquet(f"{out}/valid_scored.parquet", index=False)
        log(t0, f"valid: {len(old):,} old + {len(new):,} new pairs ({new.label.mean():.3f} positive among new) | "
                f"true pairs kept {old.label.sum():,} -> {allv.label.sum():,}")
    # ---- test
    con = text_db(W, "test", tmp)
    ce_parts = sorted(glob.glob(f"{W}/ce/test_scored/part-*.parquet"))
    pair_parts = sorted(glob.glob(f"{W}/pairs/test/part-*.parquet"))
    n_old = n_new = 0
    for i, (cp, pth) in enumerate(zip(ce_parts, pair_parts)):
        dst = f"{out}/test_scored/part-{i:03d}.parquet"
        if os.path.exists(dst):
            continue
        old = pd.read_parquet(cp)
        old["src"] = old.src.astype(np.int8)
        new = ce_score(con, widen_frame(pd.read_parquet(pth, columns=cols), old))
        pd.concat([old, new[old.columns]], ignore_index=True).to_parquet(dst, index=False)
        n_old += len(old)
        n_new += len(new)
        el = time.time() - t0
        log(t0, f"test part {i + 1}/{len(ce_parts)}: +{len(new):,} new pairs  "
                f"(~{el / (i + 1) * (len(ce_parts) - i - 1) / 60:.0f} min left)")
    con.close()
    for x in glob.glob(f"{tmp}/ce_text_*.duckdb*"):
        os.remove(x)
    log(t0, f"done: test {n_old:,} old + {n_new:,} new pairs this run -> {out}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    pd_ = sp.add_parser("data")
    pt = sp.add_parser("train")
    pt.add_argument("--max-train", type=int, default=800_000)
    pt.add_argument("--epochs", type=int, default=1)
    pt.add_argument("--batch", type=int, default=64)
    pt.add_argument("--lr", type=float, default=3e-5)
    pt.add_argument("--eval-every", type=int, default=4000)
    pt.add_argument("--eval-n", type=int, default=20000)
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--model-dir", default="model", help="saved to work/ce/<dir>")
    ps = sp.add_parser("score")
    ps.add_argument("--split", required=True, choices=["valid", "test"])
    ps.add_argument("--batch", type=int, default=512)
    ps.add_argument("--model-dir", default="model")
    ps.add_argument("--col", default="ce", help="score column; anything but 'ce' goes to side files")
    pk = sp.add_parser("stack")
    pk.add_argument("--truth", required=True)
    pk.add_argument("--rounds", type=int, default=300)
    pk.add_argument("--rich", action="store_true", help="also feed the stacker all v4 pair features")
    pp = sp.add_parser("predict")
    pp.add_argument("--out-dir", default="output")
    pp.add_argument("--mem", default="6GB")
    pw = sp.add_parser("widen")
    pw.add_argument("--tag", default="w10")
    pw.add_argument("--top-n", type=int, default=10)
    pw.add_argument("--min-pp", type=float, default=0.005)
    pw.add_argument("--batch", type=int, default=512)
    for x in (pk, pp):
        x.add_argument("--stack-dir", default="stack", help="work/ce/<dir> for the stacker model")
        x.add_argument("--scored-dir", default="", help="work/ce/<dir> holding widened scores (see widen)")
    pk.add_argument("--extra-ce", default="", help="comma list of extra cross-encoder score columns, e.g. ce2")
    for x in (pd_, pt, ps, pk, pp, pw):
        x.add_argument("--work", default="work")
        x.add_argument("--threads", type=int, default=10)
        x.add_argument("--device", default="cuda")
        x.add_argument("--max-len", type=int, default=128)
    a = ap.parse_args()
    {"data": cmd_data, "train": cmd_train, "score": cmd_score, "stack": cmd_stack, "predict": cmd_predict,
     "widen": cmd_widen}[a.cmd](a)


if __name__ == "__main__":
    main()
