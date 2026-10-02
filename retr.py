#!/usr/bin/env python3
"""
Layer 6 - FINE-TUNED RETRIEVER (second-generation blocking) + final stacker

The off-the-shelf MiniLM blocker never learned what "the same business" means in THIS data, so ~5.7% of
true pairs never become candidates.  Here the same Apache-2.0 model (multilingual MiniLM-L12, 118M) is
fine-tuned as a bi-encoder on train matches (S1 text <-> matched S2/S3 text, in-batch negatives), every
record is re-embedded as one "name | address" vector, and the top-k neighbours per source that the
existing pipeline never scored become NEW candidates.  Both cross-encoders score them, and a final
stacker decides on old + new candidates together (retriever cosine added as a feature).

    python retr.py train  --n 800000                       # fine-tune bi-encoder  -> work/ft/model
    python retr.py embed  --split train                    # sample S1 + full S2/S3 -> work/emb_ft/train_s*
    python retr.py embed  --split test                     # all test records       -> work/emb_ft/test_s*
    python retrieve_dense.py --queries work/emb_ft/train_s1 --db2 work/emb_ft/train_s2 --db3 work/emb_ft/train_s3 \
        --out work/cand/train_ft.parquet --k 3 --tag ft
    python retrieve_dense.py --queries work/emb_ft/test_s1 --db2 work/emb_ft/test_s2 --db3 work/emb_ft/test_s3 \
        --out work/cand/test_ft.parquet --k 3 --tag ft
    python retr.py newpairs --split valid --truth ...      # new candidates + recall report
    python retr.py score    --split valid                  # both cross-encoders on new pairs
    python retr.py stack    --truth ...                    # old-only vs old+new, out-of-fold F0.5
    python retr.py newpairs --split test ; python retr.py score --split test ; python retr.py predict --out-dir output_v9
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd

from ce import (ID_COLS, attach_text, is_valid, load_model, log, predict_probs, stack_features,
                stack_feature_names, text_db, v4_features)

BASE = os.environ.get("CE_BASE", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
N_PARTS = 130


def pool(out, mask):
    m = mask.unsqueeze(-1).to(out.dtype)
    return (out * m).sum(1) / m.sum(1).clamp(min=1e-6)


def encode(tok, model, texts, bs, max_len, device):
    import torch
    model.eval()
    res = np.zeros((len(texts), model.config.hidden_size), np.float16)
    ln = np.fromiter((len(t) for t in texts), np.int32, len(texts))
    order = np.argsort(ln, kind="stable")
    with torch.no_grad():
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            batch = [texts[i] for i in idx]
            enc = tok(batch, truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                e = pool(model(**enc).last_hidden_state, enc["attention_mask"])
            e = torch.nn.functional.normalize(e.float(), dim=-1)
            e[torch.tensor([len(b.strip(" |")) == 0 for b in batch], device=e.device)] = 0   # empty text
            res[idx] = e.cpu().numpy().astype(np.float16)
    return res


def lower_text(t):
    return t.str.lower().fillna("")


# ----------------------------------------------------------------------------- train
def cmd_train(a):
    import torch
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
    t0 = time.time()
    W = a.work
    tmp = f"{W}/duck_tmp"
    os.makedirs(tmp, exist_ok=True)
    t = pd.read_csv(a.truth, sep="\t", dtype=str, keep_default_na=False, quoting=3)
    t["q"] = t.source1_entity_id.str.slice(3).astype("int64")
    t = t[~is_valid(t.q) & (t.matched_entity_ids != "")]
    m = t.set_index("q").matched_entity_ids.str.split(",").explode()
    pr = pd.DataFrame({"q": m.index.to_numpy(), "src": m.str.slice(1, 2).astype("int8").to_numpy(),
                       "c": m.str.slice(3).astype("int64").to_numpy()})
    # one random positive per S1 (no duplicate anchors in a batch -> no false in-batch negatives)
    pr = pr.sample(frac=1.0, random_state=0).drop_duplicates("q").head(a.n).reset_index(drop=True)
    con = text_db(W, "train", tmp)
    d = attach_text(con, pr)
    con.close()
    d = d.sample(frac=1.0, random_state=1).reset_index(drop=True)
    ta, tb = lower_text(d.ta).tolist(), lower_text(d.tb).tolist()
    log(t0, f"{len(d):,} training pairs (S1 <-> matched S2/S3), excluding validation queries")
    for i in range(3):
        print(f"   {ta[i][:70]}  <->  {tb[i][:70]}")

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModel.from_pretrained(BASE).to(a.device)
    steps = len(d) // a.batch
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=a.device == "cuda")
    lossf = torch.nn.CrossEntropyLoss()
    model.train()
    run = None
    for st in range(steps):
        sl = slice(st * a.batch, (st + 1) * a.batch)
        ea = tok(ta[sl], truncation=True, max_length=a.max_len, padding=True, return_tensors="pt").to(a.device)
        eb = tok(tb[sl], truncation=True, max_length=a.max_len, padding=True, return_tensors="pt").to(a.device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=a.device == "cuda"):
            va = pool(model(**ea).last_hidden_state, ea["attention_mask"])
            vb = pool(model(**eb).last_hidden_state, eb["attention_mask"])
        va = torch.nn.functional.normalize(va.float(), dim=-1)
        vb = torch.nn.functional.normalize(vb.float(), dim=-1)
        sc = va @ vb.T * a.scale
        lab = torch.arange(len(va), device=a.device)
        loss = (lossf(sc, lab) + lossf(sc.T, lab)) / 2
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sch.step()
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if (st + 1) % 500 == 0 or st + 1 == steps:
            eta = (time.time() - t0) / (st + 1) * (steps - st - 1) / 60
            log(t0, f"step {st + 1:,}/{steps:,}  loss {run:.4f}  in-batch acc "
                    f"{(sc.argmax(1) == lab).float().mean().item():.3f}  (~{eta:.0f} min left)")
    os.makedirs(f"{W}/ft/model", exist_ok=True)
    model.save_pretrained(f"{W}/ft/model")
    tok.save_pretrained(f"{W}/ft/model")
    log(t0, f"saved -> {W}/ft/model")


# ----------------------------------------------------------------------------- embed
def cmd_embed(a):
    import duckdb
    import torch
    from transformers import AutoModel, AutoTokenizer
    t0 = time.time()
    W, S = a.work, a.split
    tok = AutoTokenizer.from_pretrained(f"{W}/ft/model")
    model = AutoModel.from_pretrained(f"{W}/ft/model").to(a.device)
    if a.device == "cuda":
        model = model.half()
    con = duckdb.connect()
    for s in (1, 2, 3):
        out = f"{W}/emb_ft/{S}_s{s}"
        if os.path.exists(f"{out}/emb.npy"):
            log(t0, f"{out} exists - skipped")
            continue
        flt = ""
        if S == "train" and s == 1:     # only the query sample (its meta lists the ids)
            flt = (f" WHERE entity_id IN (SELECT entity_id FROM read_parquet('{W}/emb_name/train_s1/meta.parquet'))")
        df = con.execute(f"""SELECT entity_id, country,
                 lower(trim(coalesce(name_emb, '')) || ' | ' || trim(coalesce(addr_emb, ''))) AS t
                 FROM read_parquet('{W}/l0/{S}_s{s}/part-*.parquet'){flt}""").df()
        emb = encode(tok, model, df.t.tolist(), a.batch, a.max_len, a.device)
        os.makedirs(out, exist_ok=True)
        df[["entity_id", "country"]].to_parquet(f"{out}/meta.parquet", index=False)
        np.save(f"{out}/emb.npy", emb)
        log(t0, f"{S} S{s}: embedded {len(df):,} records -> {out}  ({len(df) / (time.time() - t0):.0f} rows/s so far)")


# ----------------------------------------------------------------------------- new candidate pairs
def cmd_newpairs(a):
    t0 = time.time()
    W = a.work
    tmp = f"{W}/duck_tmp"
    os.makedirs(f"{W}/ce/ft_new", exist_ok=True)
    split = "train" if a.split == "valid" else "test"
    cand = pd.read_parquet(f"{W}/cand/{split}_ft.parquet")
    cand = cand[cand.ft_rank <= a.k]
    cand["src"] = cand.src.astype(np.int8)
    if a.split == "valid":
        cand = cand[is_valid(cand.q)]
        old = pd.read_parquet(f"{W}/ce/valid_scored.parquet", columns=ID_COLS + ["label"])
    else:
        old = pd.concat([pd.read_parquet(p, columns=ID_COLS) for p in sorted(glob.glob(f"{W}/ce/test_scored/part-*.parquet"))],
                        ignore_index=True)
    old["src"] = old.src.astype(np.int8)
    m = cand.merge(old[ID_COLS].assign(_o=1), on=ID_COLS, how="left")
    new = m[m._o.isna()].drop(columns="_o").reset_index(drop=True)
    con = text_db(W, split, tmp)
    new = attach_text(con, new)
    con.close()
    nq = old.q.nunique()
    log(t0, f"{a.split}: {len(new):,} NEW pairs from the fine-tuned retriever (top-{a.k}/source), "
            f"+{len(new) / nq:.2f} per S1 entity (old candidates {len(old) / nq:.2f})")
    if a.split == "valid":
        from eval_blocking import load_truth as lt
        meta = pd.read_parquet(f"{W}/emb_name/train_s1/meta.parquet", columns=["entity_id"])
        vq = meta.entity_id.str.slice(3).astype("int64")
        vq = set(vq[is_valid(vq)].tolist())
        tr = lt(a.truth, vq)
        tr["src"] = tr.src.astype(np.int8)
        new = new.merge(tr.assign(label=1), on=ID_COLS, how="left")
        new["label"] = new.label.fillna(0).astype(np.int8)
        n_true = len(tr)
        ft_all = cand.merge(tr, on=ID_COLS).shape[0]
        print(f"   valid true pairs {n_true:,} | fine-tuned retriever top-{a.k} alone finds {ft_all / n_true:.4f}")
        print(f"   candidate recall: old {old.label.sum() / n_true:.4f} -> old+new {(old.label.sum() + new.label.sum()) / n_true:.4f}"
              f"   (new pairs precision {new.label.mean():.3f})")
        new.to_parquet(f"{W}/ce/ft_new/valid.parquet", index=False)
    else:
        os.makedirs(f"{W}/ce/ft_new/test_text", exist_ok=True)
        part = (new.q.to_numpy() % N_PARTS)
        for i in range(N_PARTS):
            new[part == i].to_parquet(f"{W}/ce/ft_new/test_text/part-{i:03d}.parquet", index=False)
        log(t0, f"wrote {N_PARTS} parts -> {W}/ce/ft_new/test_text")


# ----------------------------------------------------------------------------- score new pairs with both cross-encoders
def cmd_score(a):
    t0 = time.time()
    W = a.work
    models = [("ce2", f"{W}/ce/model2")] if a.only_ce2 else [("ce", f"{W}/ce/model"), ("ce2", f"{W}/ce/model2")]
    loaded = [(c, *load_model(p, a.device)) for c, p in models]
    def run(d):
        for c, tok, mdl in loaded:
            d[c] = predict_probs(tok, mdl, d.ta.tolist(), d.tb.tolist(), a.batch, 128, a.device) if len(d) \
                else np.zeros(0, np.float32)
        if a.only_ce2:            # half the GPU time: new pairs get model-2 score in both columns
            d["ce"] = d["ce2"]
        return d.drop(columns=["ta", "tb"])
    if a.split == "valid":
        d = run(pd.read_parquet(f"{W}/ce/ft_new/valid.parquet"))
        d.to_parquet(f"{W}/ce/ft_new/valid_scored.parquet", index=False)
        log(t0, f"scored {len(d):,} new valid pairs")
        return
    od = f"{W}/ce/ft_new/test_scored"
    os.makedirs(od, exist_ok=True)
    parts = sorted(glob.glob(f"{W}/ce/ft_new/test_text/part-*.parquet"))
    done0 = len(glob.glob(f"{od}/part-*.parquet"))
    for i, p in enumerate(parts):
        out = f"{od}/part-{i:03d}.parquet"
        if os.path.exists(out):
            continue
        d = run(pd.read_parquet(p))
        d.to_parquet(out, index=False)
        n = i + 1 - done0
        log(t0, f"test part {i + 1}/{len(parts)} ({len(d):,} new pairs)  "
                f"~{(time.time() - t0) / max(n, 1) * (len(parts) - i - 1) / 60:.0f} min left")


# ----------------------------------------------------------------------------- stacking on old + new
def ft_cosine(W, split, d):
    from build_pairs import EmbLookup, cosines
    lk = [EmbLookup(f"{W}/emb_ft/{split}_s{s}") for s in (1, 2, 3)]
    return cosines(*lk, d.q.to_numpy(np.int64), d.src.to_numpy(np.int8), d.c.to_numpy(np.int64))


def features(d):
    d = stack_features(d, ["p", "ce", "ce2"])
    v = d.ft.fillna(-1)
    d["ft_rk"] = v.groupby(d.q).rank(ascending=False, method="min").astype(np.float32)
    d["ft_gap"] = (v.groupby(d.q).transform("max") - v).astype(np.float32)
    d["ft_rk_src"] = v.groupby([d.q, d.src]).rank(ascending=False, method="min").astype(np.float32)
    d["n_new"] = d.is_new.groupby(d.q).transform("sum").astype(np.float32)
    return d


def feat_names(v4f):
    return stack_feature_names(["ce2"]) + ["ft", "ft_rk", "ft_gap", "ft_rk_src", "is_new", "n_new"] + \
        ["f_" + f for f in v4f]


def with_v4(d, pair_files, v4f):
    x = pd.concat([pd.read_parquet(p, columns=ID_COLS + v4f) for p in pair_files], ignore_index=True)
    x = x.rename(columns={f: "f_" + f for f in v4f})
    x["src"] = x.src.astype(np.int8)
    return d.merge(x.drop_duplicates(ID_COLS), on=ID_COLS, how="left")


def cmd_stack(a):
    import lightgbm as lgb
    from match import decide, load_truth, macro_f05
    t0 = time.time()
    W = a.work
    old = pd.read_parquet(f"{W}/ce/valid_scored.parquet")
    old["src"] = old.src.astype(np.int8)
    c2 = pd.read_parquet(f"{W}/ce/valid_scored_ce2.parquet")
    c2["src"] = c2.src.astype(np.int8)
    old = old.merge(c2, on=ID_COLS, how="left")
    old["is_new"] = np.int8(0)
    new = pd.read_parquet(f"{W}/ce/ft_new/valid_scored.parquet")
    new["src"] = new.src.astype(np.int8)
    new["p"] = np.nan
    new["is_new"] = np.int8(1)
    d = pd.concat([old, new[old.columns.drop("ft", errors="ignore")]], ignore_index=True)
    d["ft"] = ft_cosine(W, "train", d)
    v4f = v4_features(W)
    parts = [p for p in sorted(glob.glob(f"{W}/pairs/train/part-*.parquet"))]
    x = []
    for p in parts:
        y = pd.read_parquet(p, columns=ID_COLS + v4f)
        x.append(y[is_valid(y.q)])
    x = pd.concat(x, ignore_index=True).rename(columns={f: "f_" + f for f in v4f})
    x["src"] = x.src.astype(np.int8)
    d = d.merge(x.drop_duplicates(ID_COLS), on=ID_COLS, how="left")
    feats = feat_names(v4f)
    meta = pd.read_parquet(f"{W}/emb_name/train_s1/meta.parquet", columns=["entity_id"])
    all_q = meta.entity_id.str.slice(3).astype("int64").to_numpy()
    n_true = load_truth(a.truth, all_q[is_valid(all_q)])
    params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=a.threads)

    def oof_f05(df, label):
        df = features(df).reset_index(drop=True)
        fold = df.q.to_numpy() * 40503 % 5
        oof = np.zeros(len(df))
        for k in range(5):
            tr, te = fold != k, fold == k
            m = lgb.train(params, lgb.Dataset(df.loc[tr, feats].astype(np.float32), df.label[tr]), a.rounds)
            oof[te] = m.predict(df.loc[te, feats].astype(np.float32))
        s = df.assign(p=oof)
        best = (0, 0.5, None)
        for t in np.round(np.arange(0.40, 0.95, 0.02), 2):
            r = macro_f05(s, decide(s, t, 0.0), n_true)
            if r[0] > best[0]:
                best = (r[0], float(t), r)
        r = best[2]
        print(f"{label:<28s} F0.5 {best[0]:.4f} @ tau {best[1]:.2f}  (prec {r[1]:.4f} rec {r[2]:.4f})  "
              f"candidate recall {df.label.sum() / n_true.sum():.4f}  cand/q {len(df) / len(n_true):.2f}")
        return best, df

    (f_old, _, _), _ = oof_f05(d[d.is_new == 0].copy(), "old candidates only")
    best, df = oof_f05(d.copy(), "old + fine-tuned retriever")
    m = lgb.train(params, lgb.Dataset(df[feats].astype(np.float32), df.label), a.rounds)
    os.makedirs(f"{W}/ce/stack_ft", exist_ok=True)
    m.save_model(f"{W}/ce/stack_ft/lgbm.txt")
    json.dump({"tau": best[1], "valid_f05_oof": best[0], "valid_f05_old": f_old, "features": feats},
              open(f"{W}/ce/stack_ft/decision.json", "w"), indent=1)
    print(f"\nBEST: tau={best[1]:.2f} -> valid F0.5 {best[0]:.4f}  (same stacker on old candidates {f_old:.4f}, "
          f"gain {best[0] - f_old:+.4f})  [{(time.time() - t0) / 60:.1f} min]")


def cmd_predict(a):
    import lightgbm as lgb
    from expand import write_submission
    from match import decide
    t0 = time.time()
    W = a.work
    dec = json.load(open(f"{W}/ce/stack_ft/decision.json"))
    m = lgb.Booster(model_file=f"{W}/ce/stack_ft/lgbm.txt")
    v4f = v4_features(W)
    old_parts = sorted(glob.glob(f"{W}/ce/test_scored/part-*.parquet"))
    out = []
    for i, op in enumerate(old_parts):
        old = pd.read_parquet(op)
        old["src"] = old.src.astype(np.int8)
        c2 = pd.read_parquet(f"{W}/ce/test_scored_ce2/part-{i:03d}.parquet")
        c2["src"] = c2.src.astype(np.int8)
        old = old.merge(c2, on=ID_COLS, how="left")
        old["is_new"] = np.int8(0)
        np_ = f"{W}/ce/ft_new/test_scored/part-{i:03d}.parquet"
        if not os.path.exists(np_):
            raise SystemExit(f"missing {np_} - run: python retr.py score --split test")
        new = pd.read_parquet(np_)
        new["src"] = new.src.astype(np.int8)
        new["p"] = np.nan
        new["is_new"] = np.int8(1)
        d = pd.concat([old, new[[c for c in old.columns if c in new.columns]]], ignore_index=True)
        d["ft"] = ft_cosine(W, "test", d)
        d = with_v4(d, [f"{W}/pairs/test/part-{i:03d}.parquet"], v4f)
        d = features(d).reset_index(drop=True)
        d["p"] = m.predict(d[dec["features"]].astype(np.float32))
        out.append(d[ID_COLS + ["p"]])
        if (i + 1) % 10 == 0:
            log(t0, f"part {i + 1}/{len(old_parts)}")
    d = pd.concat(out, ignore_index=True)
    keep = decide(d, dec["tau"], 0.0)
    log(t0, f"{len(d):,} candidate pairs, {int(keep.sum()):,} predicted matches (tau {dec['tau']})")
    write_submission(W, a.out_dir, d.loc[keep, ID_COLS], d[ID_COLS], a.mem, a.threads)
    log(t0, "done")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p1 = sp.add_parser("train")
    p1.add_argument("--truth", default="student_resource/dataset/train/train_ground_truth.tsv")
    p1.add_argument("--n", type=int, default=800_000)
    p1.add_argument("--batch", type=int, default=128)
    p1.add_argument("--lr", type=float, default=3e-5)
    p1.add_argument("--scale", type=float, default=20.0)
    p2 = sp.add_parser("embed")
    p2.add_argument("--split", required=True, choices=["train", "test"])
    p2.add_argument("--batch", type=int, default=1024)
    p3 = sp.add_parser("newpairs")
    p3.add_argument("--split", required=True, choices=["valid", "test"])
    p3.add_argument("--truth", default="student_resource/dataset/train/train_ground_truth.tsv")
    p3.add_argument("--k", type=int, default=3)
    p4 = sp.add_parser("score")
    p4.add_argument("--split", required=True, choices=["valid", "test"])
    p4.add_argument("--batch", type=int, default=512)
    p4.add_argument("--only-ce2", action="store_true", help="score new pairs with cross-encoder #2 only (2x faster)")
    p5 = sp.add_parser("stack")
    p5.add_argument("--truth", default="student_resource/dataset/train/train_ground_truth.tsv")
    p5.add_argument("--rounds", type=int, default=500)
    p6 = sp.add_parser("predict")
    p6.add_argument("--out-dir", default="output_v9")
    p6.add_argument("--mem", default="6GB")
    for x in (p1, p2, p3, p4, p5, p6):
        x.add_argument("--work", default="work")
        x.add_argument("--device", default="cuda")
        x.add_argument("--threads", type=int, default=10)
        x.add_argument("--max-len", type=int, default=64)
    a = ap.parse_args()
    {"train": cmd_train, "embed": cmd_embed, "newpairs": cmd_newpairs, "score": cmd_score,
     "stack": cmd_stack, "predict": cmd_predict}[a.cmd](a)


if __name__ == "__main__":
    main()
