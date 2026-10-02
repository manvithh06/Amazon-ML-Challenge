import pandas as pd, csv
D = "student_resource/dataset/train/"
rd = lambda f, **k: pd.read_csv(D+f, sep="\t", dtype=str, keep_default_na=False,
                                quoting=csv.QUOTE_NONE, **k)
cty = {}
for s in ("1", "2", "3"):
    df = rd(f"train_source{s}.tsv", usecols=["entity_id", "country"])
    df["country"] = df.country.str.strip().str.upper()
    print(f"S{s}: {len(df):,} rows;", df.country.value_counts().to_dict())
    cty.update(zip(df.entity_id, df.country))

gt = rd("train_ground_truth.tsv")
gt["ids"] = gt.matched_entity_ids.map(lambda x: [i for i in x.split(",") if i])
print("\nS1 entities:", len(gt), "| singletons:", (gt.ids.str.len() == 0).mean().round(3))
print("matches per entity:", gt.ids.str.len().describe().round(2).to_dict())

rows = [(a, b, cty.get(a), cty.get(b)) for a, l in zip(gt.source1_entity_id, gt.ids) for b in l]
m = pd.DataFrame(rows, columns=["s1", "match", "c1", "c2"])
print("\nmatched pairs:", len(m), "| missing country lookup:", m[["c1","c2"]].isna().any(axis=1).sum())
bad = m[m.c1 != m.c2]
print("CROSS-COUNTRY true matches:", len(bad))
print(bad.groupby(["c1", "c2"]).size() if len(bad) else "-> none: no cross-country fallback needed")
print("\nS2 vs S3 share:", m["match"].str[:2].value_counts(normalize=True).round(3).to_dict())
print("matched IDs claimed by >1 S1 entity:", m["match"].duplicated().sum())