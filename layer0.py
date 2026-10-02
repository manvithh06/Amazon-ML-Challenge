#!/usr/bin/env python3
"""
Layer 0 - Normalization & cleaning  (Amazon ML Challenge 2026 - Business Entity Resolution)

Streams a big TSV (entity_id, business_name, business_address, country) in byte-range
chunks across worker processes and writes Parquet parts.  Nothing is loaded whole.

    pip install pandas pyarrow anyascii
    python layer0.py --in student_resource/dataset/train/train_source1.tsv --out work/l0/train_s1
    python layer0.py --selftest            # sanity output on built-in sample rows
    python layer0.py --in FILE --out DIR --show 15   # also print 15 processed rows

Read the result with:  pd.read_parquet("work/l0/train_s1")   (directory of parts)

Design notes
  * Two views of every string are kept:
      - romanized/normalized (anyascii) -> for lexical + string-similarity features
      - original script, only lightly cleaned (name_emb / addr_emb) -> for the multilingual
        embedding model and the cross-encoder, which can READ Devanagari/Tamil/... directly.
        (anyascii romanization of Indic scripts is phonetic, e.g. "finance" -> "phainens",
        so it is a weak match for the English spelling; the embedding view covers that gap.)
  * Country is an open set. Country-specific tables (state abbreviations, city synonyms) are
    looked up by the normalized country label and simply skipped for unknown labels (France
    etc.). Nothing else in the pipeline depends on the country value.
  * No external lookups: every dictionary below is a static table written by hand.
"""
import argparse
import os
import re
import sys
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor

from anyascii import anyascii

COLUMNS = ["entity_id", "business_name", "business_address", "country"]

# --------------------------------------------------------------------------- tables
NULL_SEGMENTS = {"null", "none", "nan", "n/a", "nil", "-", "--"}

NAME_ABBR = {
    "pvt": "private", "ltd": "limited", "corp": "corporation", "co": "company",
    "inc": "incorporated", "incorp": "incorporated", "intl": "international",
    "assoc": "associates", "svcs": "services", "mfg": "manufacturing", "bros": "brothers",
}
LEGAL = {
    "private", "limited", "incorporated", "corporation", "company", "llc", "llp", "lp",
    "pllc", "plc", "pc", "pa", "opc", "sarl", "sarlu", "sas", "sasu", "sa", "eurl", "sci",
    "snc", "selarl", "scop", "gmbh", "ag", "bv", "nv",
}
STOP = {"and", "the", "of", "de", "du", "la", "le", "les", "des", "et", "d", "l"}

ADDR_ABBR = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "dr": "drive", "ln": "lane", "hwy": "highway",
    "ct": "court", "pl": "place", "cir": "circle", "pkwy": "parkway", "ste": "suite",
    "apt": "apartment", "flr": "floor", "fl": "floor", "bldg": "building", "blk": "block",
    "ind": "industrial", "est": "estate", "sec": "sector", "sq": "square", "rte": "route",
    "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast",
    "nw": "northwest", "se": "southeast", "sw": "southwest",
    "r": "rue", "imp": "impasse", "chem": "chemin", "all": "allee", "nos": "no",
    "number": "no",
}
STREET_TYPES = {
    "road", "street", "avenue", "boulevard", "drive", "lane", "highway", "court", "place",
    "circle", "parkway", "way", "rue", "impasse", "chemin", "allee", "nagar", "marg",
}


def _tbl(s):
    return {a.lower(): n for a, n in (p.split(" ", 1) for p in s.split(";"))}


US_STATES = _tbl(
    "AL alabama;AK alaska;AZ arizona;AR arkansas;CA california;CO colorado;CT connecticut;"
    "DE delaware;DC district of columbia;FL florida;GA georgia;HI hawaii;ID idaho;"
    "IL illinois;IN indiana;IA iowa;KS kansas;KY kentucky;LA louisiana;ME maine;"
    "MD maryland;MA massachusetts;MI michigan;MN minnesota;MS mississippi;MO missouri;"
    "MT montana;NE nebraska;NV nevada;NH new hampshire;NJ new jersey;NM new mexico;"
    "NY new york;NC north carolina;ND north dakota;OH ohio;OK oklahoma;OR oregon;"
    "PA pennsylvania;RI rhode island;SC south carolina;SD south dakota;TN tennessee;"
    "TX texas;UT utah;VT vermont;VA virginia;WA washington;WV west virginia;"
    "WI wisconsin;WY wyoming;PR puerto rico")
IN_STATES = _tbl(
    "AP andhra pradesh;AR arunachal pradesh;AS assam;BR bihar;CG chhattisgarh;"
    "CH chandigarh;DL delhi;GA goa;GJ gujarat;HP himachal pradesh;HR haryana;"
    "JH jharkhand;JK jammu and kashmir;KA karnataka;KL kerala;LA ladakh;MH maharashtra;"
    "ML meghalaya;MN manipur;MP madhya pradesh;MZ mizoram;NL nagaland;OD odisha;"
    "OR odisha;PB punjab;PY puducherry;RJ rajasthan;SK sikkim;TG telangana;"
    "TN tamil nadu;TR tripura;UK uttarakhand;UP uttar pradesh;WB west bengal")
IN_SYNONYMS = {
    "orissa": "odisha", "bangalore": "bengaluru", "bombay": "mumbai",
    "calcutta": "kolkata", "madras": "chennai", "gurgaon": "gurugram",
    "pondicherry": "puducherry", "trivandrum": "thiruvananthapuram", "cochin": "kochi",
    "baroda": "vadodara", "poona": "pune", "mysore": "mysuru", "benares": "varanasi",
}
# country label (normalized) -> (state table, token synonyms). Unknown labels: no-op.
COUNTRY_CFG = {
    "us": (US_STATES, {}), "usa": (US_STATES, {}), "united states": (US_STATES, {}),
    "india": (IN_STATES, IN_SYNONYMS),
}

# --------------------------------------------------------------------------- regexes
NULL_WORD_RE = re.compile(r"\bnull\b", re.I)
LEAD_SYM_RE = re.compile(r"^[^\w]+")
WS_RE = re.compile(r"\s+")
DBA_RE = re.compile(
    r"(?<![a-z0-9])(?:d/b/a|d\.b\.a\.?|dba|doing business as|trading as|t/a|f/k/a|fka|"
    r"a\.k\.a\.?|aka)(?![a-z0-9])")
DOTTED_RE = re.compile(r"(?<![a-z])(?:[a-z]\.)+[a-z]?(?![a-z])")
DOMAIN_RE = re.compile(
    r"\b(?:www\.)?([a-z0-9][a-z0-9-]*)\.(?:com|net|org|biz|info|co\.in|in|fr|us)\b")
APOS_RE = re.compile(r"['’`]")
NONALNUM_RE = re.compile(r"[^a-z0-9]+")
LLC_RE = re.compile(r"\blimited liability (company|partnership)\b")
SPACED_LEGAL_RE = re.compile(r"\b(p l l c|l l c|l l p|s a r l|s a s)\b")
PUBLIC_RE = re.compile(r"\bpublic (?=limited|company)")
# a landmark runs from the keyword to the next comma / closing bracket
LANDMARK_RE = re.compile(
    r"\b(?:near|nr|opp|opposite|behind|beside|next\s+to|adjacent\s+to)\b\.?[^,)]*\)?", re.I)
NONLATIN_RE = re.compile(r"[^\x00-\u024F\u1E00-\u1EFF\u2000-\u206F\u20A0-\u20CF]")
ORDINAL_RE = re.compile(r"^\d+(?:st|nd|rd|th)$")
DIGITS_RE = re.compile(r"\d+")


# --------------------------------------------------------------------------- helpers
def translit(s):
    """Romanize non-ASCII text (offline, algorithmic). ASCII passes through untouched."""
    if s.isascii():
        return s
    return anyascii(unicodedata.normalize("NFKC", s))


def clean_nulls(s):
    """Drop literal null-ish comma segments / words that some sources use as filler."""
    if not s:
        return ""
    s = s.strip()
    if s.lower() in NULL_SEGMENTS:
        return ""
    parts = [p.strip() for p in s.split(",")]
    parts = [p for p in parts if p and p.lower() not in NULL_SEGMENTS]
    return WS_RE.sub(" ", NULL_WORD_RE.sub(" ", ", ".join(parts))).strip()


def _dedup(tokens):
    out = []
    for t in tokens:
        if not out or out[-1] != t:
            out.append(t)
    return out


def name_tokens(s):
    """lower-cased ASCII string -> canonical token list (abbrev-expanded, deduped)."""
    s = s.replace("&", " and ").replace("+", " ")
    s = DOMAIN_RE.sub(r"\1", s)
    s = DOTTED_RE.sub(lambda m: m.group(0).replace(".", ""), s)
    s = APOS_RE.sub("", s)
    s = NONALNUM_RE.sub(" ", s).strip()
    s = SPACED_LEGAL_RE.sub(lambda m: m.group(0).replace(" ", ""), s)
    s = LLC_RE.sub(lambda m: "llc" if m.group(1) == "company" else "llp", s)
    s = PUBLIC_RE.sub("", s)
    return _dedup([NAME_ABBR.get(t, t) for t in s.split()])


def name_views(part):
    toks = name_tokens(part)
    legal = sorted({t for t in toks if t in LEGAL})
    core = [t for t in toks if t not in LEGAL and t not in STOP] or toks
    return toks, core, legal


def process_name(raw):
    raw = clean_nulls(raw)
    emb = LEAD_SYM_RE.sub("", raw).strip()          # original script kept
    low = translit(raw).lower()
    m = DBA_RE.search(low)
    if m:
        main, alt = low[:m.start()], low[m.end():]
    else:
        main, alt = low, ""
    toks, core, legal = name_views(main)
    a_toks, a_core, a_legal = name_views(alt) if alt.strip() else ([], [], [])
    if not core and a_core:                          # name was "dba X"
        toks, core, legal, a_toks, a_core = a_toks, a_core, a_legal, [], []
    return {
        "name_emb": emb,
        "name_norm": " ".join(toks),
        "name_core": " ".join(core),
        "name_sorted": " ".join(sorted(core)),
        "dba_core": " ".join(a_core),
        "legal": " ".join(sorted(set(legal) | set(a_legal))),
        "has_dba": bool(m),
    }


def tidy_commas(s):
    """Remove dangling brackets and empty comma segments left behind by landmark removal."""
    if s.count("(") != s.count(")"):
        s = s.replace("(", " ").replace(")", " ")
    parts = [WS_RE.sub(" ", p).strip() for p in s.split(",")]
    return ", ".join(p for p in parts if p)


def process_addr(raw, country):
    a = clean_nulls(raw)
    landmarks = [m.group(0) for m in LANDMARK_RE.finditer(a)]
    a_nolm = tidy_commas(LANDMARK_RE.sub(" ", a))
    emb = a_nolm                                      # original script kept
    states, syn = COUNTRY_CFG.get(country, ({}, {}))

    low = translit(a_nolm).lower()
    low = APOS_RE.sub("", low)
    segs = [sg for sg in (s.strip() for s in low.split(",")) if sg]
    if states:                                       # expand "tx" / "kA" only in the tail
        for i in range(max(0, len(segs) - 3), len(segs)):
            words = re.findall(r"[a-z]+", segs[i])
            if len(words) == 1 and words[0] in states:
                segs[i] = re.sub(r"[a-z]+", states[words[0]], segs[i], count=1)
    toks = NONALNUM_RE.sub(" ", " ".join(segs)).split()
    toks = _dedup([syn.get(t, ADDR_ABBR.get(t, t)) for t in toks])

    postal, nums = [], []
    for i, t in enumerate(toks):
        if not any(ch.isdigit() for ch in t) or ORDINAL_RE.match(t):
            continue
        if t.isdigit() and (len(t) == 6 or (len(t) == 5 and i > 0 and
                            (i + 1 >= len(toks) or toks[i + 1] not in STREET_TYPES))):
            postal.append(t)
        else:
            nums.extend(d.lstrip("0") or "0" for d in DIGITS_RE.findall(t))
    lm_norm = " ".join(NONALNUM_RE.sub(" ", translit(" ; ".join(landmarks)).lower()).split())
    return {
        "addr_emb": emb,
        "addr_norm": " ".join(toks),
        "addr_words": " ".join(t for t in toks if not any(c.isdigit() for c in t)),
        "landmark": lm_norm,
        "postal": "|".join(dict.fromkeys(postal)),
        "streetnums": "|".join(dict.fromkeys(nums)),
    }


OUT_COLS = ["entity_id", "country", "name_emb", "name_norm", "name_core", "name_sorted",
            "dba_core", "legal", "addr_emb", "addr_norm", "addr_words", "landmark",
            "postal", "streetnums", "has_dba", "has_nonlatin", "name_missing",
            "addr_missing"]


def process_record(eid, name, addr, country):
    c = WS_RE.sub(" ", country.strip().casefold()) or "unknown"
    n = process_name(name)
    a = process_addr(addr, c)
    row = {"entity_id": eid.strip(), "country": c, **n, **a}
    row["has_nonlatin"] = bool(NONLATIN_RE.search(name) or NONLATIN_RE.search(addr))
    row["name_missing"] = not n["name_core"]
    row["addr_missing"] = not a["addr_norm"]
    return [row[k] for k in OUT_COLS]


# --------------------------------------------------------------------------- IO / parallel
def parse_line(line):
    """Tolerant TSV parse: never drops a row (every test S1 entity must be answered)."""
    f = line.rstrip("\r\n").split("\t")
    if len(f) == 4:
        return f, False
    if len(f) > 4:                                    # stray tabs inside the address
        return [f[0], f[1], " ".join(f[2:-1]), f[-1]], True
    return f + [""] * (4 - len(f)), True


def split_ranges(path, part_bytes):
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        header = fh.readline().decode("utf-8-sig", "replace").rstrip("\r\n").split("\t")
        start = fh.tell()
        n_parts = max(1, (size - start) // part_bytes)
        step = (size - start) // n_parts
        bounds = [start]
        for i in range(1, n_parts):
            fh.seek(max(start + i * step, bounds[-1]))
            fh.readline()                             # align to the next line start
            pos = fh.tell()
            if bounds[-1] < pos < size:
                bounds.append(pos)
    bounds.append(size)
    return header, list(zip(bounds[:-1], bounds[1:]))


def _worker(args):
    import pandas as pd
    path, a, b, out_dir, idx = args
    with open(path, "rb") as fh:
        fh.seek(a)
        text = fh.read(b - a).decode("utf-8", "replace")
    rows, bad = [], 0
    for line in text.split("\n"):
        if not line.strip():
            continue
        f, was_bad = parse_line(line)
        bad += was_bad
        rows.append(process_record(*f))
    df = pd.DataFrame(rows, columns=OUT_COLS)
    df.to_parquet(os.path.join(out_dir, f"part-{idx:05d}.parquet"), index=False)
    return len(df), bad, int(df.has_nonlatin.sum()), int(df.addr_missing.sum())


def run(path, out_dir, workers, part_mb, show):
    import pandas as pd
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    header, ranges = split_ranges(path, part_mb * 1024 * 1024)
    if [h.strip() for h in header] != COLUMNS:
        print(f"WARNING: unexpected header {header}", file=sys.stderr)
    tasks = [(path, a, b, out_dir, i) for i, (a, b) in enumerate(ranges)]
    tot = bad = nonlat = noaddr = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for n, bd, nl, na in ex.map(_worker, tasks):
            tot += n; bad += bd; nonlat += nl; noaddr += na
            print(f"\r{tot:,} rows", end="", file=sys.stderr, flush=True)
    dt = time.time() - t0
    print(f"\n{path}: {tot:,} rows in {dt:.0f}s ({tot / max(dt, 1e-9):,.0f} rows/s) | "
          f"malformed-but-kept {bad:,} | non-Latin {nonlat:,} ({nonlat / max(tot, 1):.1%}) | "
          f"empty address {noaddr:,} | parts {len(tasks)} -> {out_dir}")
    if show:
        pd.set_option("display.width", 250, "display.max_colwidth", 60,
                      "display.max_columns", 30)
        print(pd.read_parquet(os.path.join(out_dir, "part-00000.parquet")).head(show).T)


# --------------------------------------------------------------------------- self test
SAMPLE = [
    ("S1-1", "Zephyr Labs Inc", "2621 Cotten Road, Tyler, TX", "US"),
    ("S1-2", "<< Team Ecole", "175 Boulevard du Président Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine", "France"),
    ("S1-3", "Roongta Sangh", "Bhubaneswar, Sub Plot No.-L6/29, Mahodadhi Bhawan (Next To Iter College) Panchasakha Nagar, Dumuduma, Khordha, Plot No.-780, Orissa", "India"),
    ("S1-4", "Anupama Estates Pvt Ltd", "26, Ram Nagar, Behind M.C Quarters Near Jagadhri Gate, Ambala City, Haryana", "India"),
    ("S2-1", "राम मार्केटिंग प्राइवेट लिमिटेड", "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi", "India"),
    ("S2-2", "*** Yamuica Venus Pvt Ltd", "1312C/11, GALI NO. 05 OPP. MUKHI HOSPITAL, SUDAMA NAGAR, SONEPAT, HARYANA, हरियाणा", "India"),
    ("S2-3", "VIOVERA TOLNS", "MA, 25 NIAGARA STREET, NULL, SPRINGFIELD", "US"),
    ("S2-4", "Crestline Crestline Clean LP", "HIGHWAY 49, ASHLAND CITY, TN", "US"),
    ("S2-5", "Indian Investment L.L.P.", "G-202 ACE ASPIRE, ATUL MARG, BISHRAKH, Uttar Pradesh", "India"),
    ("S2-6", "Marina Ecole France Sarl", "63 R. DE DIEPPE, LILLE, Hauts-de-France", "France"),
    ("S3-1", "Ectolumdrex dba X+ Madison Inc", "S03575 Cty Tk M, Town Of Buffalo, WI", "US"),
    ("S3-2", "Prime Realty Ventures Public Limited", "G.t. Karnal Road, Industrial Area, New Delhi, null, A-68, दिल्ली", "India"),
    ("S3-3", "wilfordhancock.com", "Mack Rd, Haltom City, Texas", "US"),
    ("S3-4", "Europ & Frères Distribution S.A.", "(41) Rue Des Thuyas, Lège-cap-ferret, Gironde", "France"),
    ("S3-5", "Sai Infratech Pvt Pvt Ltd Service", "Hbr 2Nd Stage Opp Inam Masjid, Shampura Main Road, Bangalore, Bengaluru, KA", "India"),
    ("S3-6", "Empty Address Co", "NULL", "Germany"),
    ("S3-7", "Big Zip LLC", "18900 Harl Weiller Road, Caldwell, OH 43724", "US"),
]


def selftest():
    import pandas as pd
    pd.set_option("display.width", 250, "display.max_colwidth", 55, "display.max_columns", 30)
    df = pd.DataFrame([process_record(*r) for r in SAMPLE], columns=OUT_COLS)
    for _, r in df.iterrows():
        print("-" * 100)
        for k in OUT_COLS[1:]:
            print(f"  {k:13s}: {r[k]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp")
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=max(1, min(10, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--part-mb", type=int, default=32, help="bytes of input per task")
    ap.add_argument("--show", type=int, default=0, help="print N processed rows afterwards")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.inp and a.out):
        ap.error("--in and --out are required (or use --selftest)")
    run(a.inp, a.out, a.workers, a.part_mb, a.show)


if __name__ == "__main__":
    main()
