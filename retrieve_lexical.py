#!/usr/bin/env python3
"""
Layer 1c - lexical blocker v2: multi-key inverted-index join in DuckDB (CPU only)

v1 (name words only) found just 20% of true pairs on the real data: many matches have unrelated,
transliterated or website-style names, but an almost identical ADDRESS.  v2 therefore matches on
several independent key families, each scored by IDF (rare keys count more):

  NAME   n  single name word (>=3 chars), plus the whole name with spaces removed
            ("Kappes Creative Flex" -> "kappescreativeflex" == the website-style "kappescreativeflex.com")
         p  unordered PAIR of name words (survives word-order shuffles, dropped words, extra words)
  ADDR   a  (street/unit number, address word)  e.g. ("266", "raipur")  <- strongest key
         b  adjacent address-word pair          e.g. ("fieldcrest", "lane") (survives number typos)
         z  postal code, y = (postal code, number)

Keys are hashed together with the country, so records only ever meet inside their own country
(country stays an open set).  Keys that occur in more than --maxdf-* database records are ignored.
For every (query, source) we keep the union of three top-k pools - best total score, best NAME
score, best ADDRESS score - so neither signal can crowd the other out.

Output parquet: q int64 | src int8 | c int64 | lex_name float | lex_addr float | lex_rank int
(lex_rank = best rank of the pair in any pool; use  lex_rank <= k  like the other candidate files)

    python retrieve_lexical.py --q work/l0/train_s1 --db2 work/l0/train_s2 --db3 work/l0/train_s3 \
        --ids-from work/emb_name/train_s1 --out work/cand/train_lex.parquet
    python retrieve_lexical.py --q work/l0/test_s1 --db2 work/l0/test_s2 --db3 work/l0/test_s3 \
        --out work/cand/test_lex.parquet --chunks 8
"""
import argparse
import glob
import os
import shutil
import time

import duckdb

REC_COLS = "country, name_core, dba_core, addr_words, streetnums, postal"


def rec_sql(path, src):
    return (f"SELECT {src} AS src, CAST(substr(entity_id, 4) AS BIGINT) AS id, {REC_COLS} "
            f"FROM read_parquet('{path}/part-*.parquet')")


def key_views(con, rec):
    """Create view <rec>_keys(src, id, key UBIGINT, fam VARCHAR) from relation <rec> (NOT de-duplicated:
    callers apply DISTINCT after filtering, which keeps memory small on the 10M-row database side)."""
    con.execute(f"""
    CREATE OR REPLACE VIEW {rec}_nt AS
      SELECT src, id, country, tok, pos FROM (
        SELECT src, id, country, unnest(toks) AS tok, generate_subscripts(toks, 1) AS pos
        FROM (SELECT src, id, country, string_split(trim(name_core || ' ' || dba_core), ' ') AS toks FROM {rec})
      ) WHERE length(tok) >= 3 AND pos <= 8""")
    con.execute(f"""
    CREATE OR REPLACE VIEW {rec}_aw AS
      SELECT src, id, country, w, pos FROM (
        SELECT src, id, country, unnest(ws) AS w, generate_subscripts(ws, 1) AS pos
        FROM (SELECT src, id, country, string_split(addr_words, ' ') AS ws FROM {rec})
      ) WHERE length(w) >= 3 AND pos <= 12""")
    con.execute(f"""
    CREATE OR REPLACE VIEW {rec}_num AS
      SELECT src, id, country, unnest(list_slice(string_split(streetnums, '|'), 1, 3)) AS num
      FROM {rec} WHERE streetnums <> ''""")
    con.execute(f"""
    CREATE OR REPLACE VIEW {rec}_pc AS
      SELECT src, id, country, unnest(string_split(postal, '|')) AS pc FROM {rec} WHERE postal <> ''""")
    con.execute(f"""
    CREATE OR REPLACE VIEW {rec}_keys AS
    SELECT src, id, hash(country || '|' || ks) AS key, substr(ks, 1, 1) AS fam FROM (
        SELECT src, id, country, ks FROM (
            SELECT src, id, country, 'n|' || tok AS ks FROM {rec}_nt
            UNION ALL SELECT src, id, country, 'n|' || replace(name_core, ' ', '') FROM {rec}
                      WHERE length(replace(name_core, ' ', '')) >= 6 AND name_core LIKE '% %'
            UNION ALL SELECT a.src, a.id, a.country, 'p|' || least(a.tok, b.tok) || '|' || greatest(a.tok, b.tok)
                      FROM {rec}_nt a JOIN {rec}_nt b ON a.src = b.src AND a.id = b.id AND a.pos < b.pos
                      WHERE b.pos <= 6
            UNION ALL SELECT n.src, n.id, n.country, 'a|' || n.num || '|' || w.w
                      FROM {rec}_num n JOIN {rec}_aw w ON n.src = w.src AND n.id = w.id
            UNION ALL SELECT a.src, a.id, a.country, 'b|' || a.w || '|' || b.w
                      FROM {rec}_aw a JOIN {rec}_aw b ON a.src = b.src AND a.id = b.id AND b.pos = a.pos + 1
            UNION ALL SELECT src, id, country, 'z|' || pc FROM {rec}_pc
            UNION ALL SELECT p.src, p.id, p.country, 'y|' || p.pc || '|' || n.num
                      FROM {rec}_pc p JOIN {rec}_num n ON p.src = n.src AND p.id = n.id
        )
    )""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True, help="Layer-0 dir of the query side (S1)")
    ap.add_argument("--db2", required=True)
    ap.add_argument("--db3", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids-from", help="embedding dir whose meta.parquet lists the query sample")
    ap.add_argument("--maxdf-name", type=int, default=300, help="ignore single name words more common than this")
    ap.add_argument("--maxdf-key", type=int, default=60, help="ignore pair/address keys more common than this")
    ap.add_argument("--k", type=int, default=20, help="per (query, source): size of each of the 3 pools")
    ap.add_argument("--chunks", type=int, default=1, help="process queries in N slices to bound memory")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--mem", default="8GB")
    ap.add_argument("--tmp", default="work/duck_tmp")
    a = ap.parse_args()

    os.makedirs(a.tmp, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET threads={a.threads}; SET memory_limit='{a.mem}'; SET temp_directory='{a.tmp}'; "
                f"SET preserve_insertion_order=false;")
    t0 = time.time()

    def step(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    flt = ""
    if a.ids_from:
        flt = (f" WHERE CAST(substr(entity_id, 4) AS BIGINT) IN (SELECT CAST(substr(entity_id, 4) AS BIGINT) "
               f"FROM read_parquet('{a.ids_from}/meta.parquet'))")
    con.execute(f"CREATE OR REPLACE VIEW qrec AS {rec_sql(a.q, 1)}{flt}")
    con.execute(f"CREATE OR REPLACE VIEW drec AS {rec_sql(a.db2, 2)} UNION ALL {rec_sql(a.db3, 3)}")
    key_views(con, "qrec")
    key_views(con, "drec")

    con.execute("CREATE TABLE q_keys AS SELECT DISTINCT id AS q, key, fam FROM qrec_keys")
    nq = con.execute("SELECT count(DISTINCT q) FROM q_keys").fetchone()[0]
    step(f"query keys: {con.execute('SELECT count(*) FROM q_keys').fetchone()[0]:,} for {nq:,} queries")

    # only database keys that some query also has are needed (df is still exact for those keys)
    con.execute("CREATE TABLE qkeyset AS SELECT DISTINCT key FROM q_keys")
    con.execute("""CREATE TABLE d_keys AS SELECT DISTINCT src, id, key, fam FROM drec_keys
                   WHERE key IN (SELECT key FROM qkeyset)""")
    step(f"database key hits: {con.execute('SELECT count(*) FROM d_keys').fetchone()[0]:,}")
    con.execute(f"""CREATE TABLE d_keep AS
        SELECT d.src, d.id, d.key, ln(3000000.0 / k.df) AS idf
        FROM d_keys d JOIN (SELECT key, count(*) AS df FROM d_keys GROUP BY key) k USING (key)
        WHERE k.df <= CASE WHEN d.fam = 'n' THEN {a.maxdf_name} ELSE {a.maxdf_key} END""")
    con.execute("DROP TABLE d_keys")
    step(f"kept after df cap: {con.execute('SELECT count(*) FROM d_keep').fetchone()[0]:,}")

    parts = os.path.join(a.tmp, "parts")
    shutil.rmtree(parts, ignore_errors=True)
    os.makedirs(parts)
    for i in range(a.chunks):
        con.execute(f"""COPY (
          WITH pairs AS (
            SELECT q.q AS q, d.src AS src, d.id AS c,
                   sum(CASE WHEN q.fam IN ('n', 'p') THEN d.idf ELSE 0 END) AS lex_name,
                   sum(CASE WHEN q.fam IN ('n', 'p') THEN 0 ELSE d.idf END) AS lex_addr
            FROM q_keys q JOIN d_keep d ON q.key = d.key
            WHERE q.q % {a.chunks} = {i}
            GROUP BY q.q, d.src, d.id),
          ranked AS (
            SELECT *,
              row_number() OVER (PARTITION BY q, src ORDER BY lex_name + lex_addr DESC, c) AS r_tot,
              row_number() OVER (PARTITION BY q, src ORDER BY lex_name DESC, lex_addr DESC, c) AS r_name,
              row_number() OVER (PARTITION BY q, src ORDER BY lex_addr DESC, lex_name DESC, c) AS r_addr
            FROM pairs)
          SELECT q, CAST(src AS TINYINT) AS src, c, CAST(lex_name AS FLOAT) AS lex_name,
                 CAST(lex_addr AS FLOAT) AS lex_addr,
                 CAST(least(r_tot, CASE WHEN lex_name > 0 THEN r_name ELSE 1000000 END,
                                   CASE WHEN lex_addr > 0 THEN r_addr ELSE 1000000 END) AS INTEGER) AS lex_rank
          FROM ranked
          WHERE r_tot <= {a.k} OR (lex_name > 0 AND r_name <= {a.k}) OR (lex_addr > 0 AND r_addr <= {a.k})
        ) TO '{parts}/part-{i:03d}.parquet' (FORMAT parquet)""")
        step(f"chunk {i + 1}/{a.chunks} done")
    con.execute(f"COPY (SELECT * FROM read_parquet('{parts}/part-*.parquet')) TO '{a.out}' (FORMAT parquet)")
    n = con.execute(f"SELECT count(*), count(DISTINCT q) FROM read_parquet('{a.out}')").fetchone()
    shutil.rmtree(parts, ignore_errors=True)
    step(f"wrote {n[0]:,} candidate rows for {n[1]:,} of {nq:,} queries -> {a.out}")


if __name__ == "__main__":
    main()
