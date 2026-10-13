#!/usr/bin/env python3
"""Synthetic data generator for the Amazon ML Challenge 2026 entity-resolution smoke test.

Generates tab-separated files in the exact challenge format:
    <split>_source1.tsv / <split>_source2.tsv / <split>_source3.tsv
        columns: entity_id, business_name, business_address, country
    <split>_ground_truth.tsv   (train only)
        columns: source1_entity_id, matched_entity_ids  (comma-separated, "" if none)

Realistic noise: abbreviations (Corp/Corporation, Pvt/Private, Ltd/Limited,
Rd/Road, St/Street ...), typos, word-order swaps, dropped components,
address abbreviation variants, landmark phrases, transliteration-ish variants.
Train countries: US, India. Test adds France (open label set).
~20% of S1 entities are singletons (no matches); the rest get 1-4 matches
spread across Source 2 and Source 3, plus unmatched "distractor" S2/S3 records.

Usage:
    python3 make_synthetic.py --seed 42 --out synthetic
"""
import argparse
import csv
import random
import re
from collections import defaultdict

# ---------------------------------------------------------------------------
# Per-country lexicons
# ---------------------------------------------------------------------------
COUNTRY = {
    "US": {
        "cities": ["New York", "Chicago", "Houston", "Los Angeles", "San Francisco",
                   "Seattle", "Austin", "Denver", "Boston", "Miami", "Atlanta", "Dallas"],
        "states": ["NY", "IL", "TX", "CA", "WA", "CO", "MA", "FL", "GA"],
        "street_names": ["Maple", "Oak", "Cedar", "Pine", "Elm", "River", "Lake",
                         "Hill", "Park", "Sunset", "Broadway", "Main", "Washington",
                         "Franklin", "Lincoln", "Highland", "Valley", "Meadow"],
        "street_types": ["Street", "Avenue", "Road", "Boulevard", "Lane", "Drive"],
        "suffixes": ["Inc", "Corp", "LLC", "Co"],
        "zip_fmt": lambda r: f"{r.randint(10000, 99999)}",
    },
    "India": {
        "cities": ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Hyderabad",
                   "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow"],
        "states": ["Maharashtra", "Karnataka", "Tamil Nadu", "Telangana", "Gujarat"],
        "street_names": ["Gandhi", "Nehru", "Sharma", "MG", "Station", "Market",
                         "Temple", "Lake", "Garden", "Church", "Fort", "Palace",
                         "Station", "Linking", "Brigade", "Residency"],
        "street_types": ["Road", "Street", "Marg", "Nagar", "Lane", "Chowk"],
        "suffixes": ["Pvt Ltd", "Ltd", "LLP", "Private Limited"],
        "zip_fmt": lambda r: f"{r.randint(100000, 999999)}",
    },
    "France": {
        "cities": ["Paris", "Lyon", "Marseille", "Toulouse", "Bordeaux",
                   "Nantes", "Lille", "Nice", "Strasbourg", "Rennes"],
        "states": ["", "", ""],
        "street_names": ["Victor Hugo", "de la Paix", "des Fleurs", "du Moulin",
                         "de la Gare", "Saint Martin", "de Rivoli", "Pasteur",
                         "de la Republique", "Jeanne d'Arc", "du Chateau", "Carnot"],
        "street_types": ["Rue", "Avenue", "Boulevard", "Impasse", "Place"],
        "suffixes": ["SARL", "SAS", "SA", "EURL"],
        "zip_fmt": lambda r: f"{r.randint(1000, 95999):05d}",
    },
}

# Abbreviation <-> full-form swaps (both directions used as noise)
ABBR_PAIRS = [
    ("Corporation", "Corp"), ("Incorporated", "Inc"), ("Company", "Co"),
    ("Limited", "Ltd"), ("Private", "Pvt"),
    ("Technologies", "Tech"), ("Technology", "Tech"),
    ("Services", "Svcs"), ("Industries", "Ind"),
    ("Street", "St"), ("Avenue", "Ave"), ("Road", "Rd"),
    ("Boulevard", "Blvd"), ("Lane", "Ln"), ("Drive", "Dr"),
    ("Marg", "Mg"), ("Nagar", "Ngr"),
]
SUFFIX_ABBR = {
    "US": [("Inc", "Incorporated"), ("Corp", "Corporation"), ("Co", "Company")],
    "India": [("Pvt Ltd", "Private Limited"), ("Ltd", "Limited")],
    "France": [],
}

BRAND_PREFIX = ["Blue", "Green", "Red", "Nova", "Apex", "Prime", "Swift",
                "Bright", "True", "Ever", "Grand", "Royal", "Silver", "Golden",
                "Star", "Moon", "Sun", "Crystal", "Iron", "Copper"]
BRAND_CORE = ["Peak", "Star", "Line", "Wave", "Forge", "Nest", "Path",
              "Craft", "Labs", "Works", "Point", "Bridge", "Stone", "Leaf", "Bay"]
SECTORS = ["Technologies", "Foods", "Textiles", "Logistics", "Retail",
           "Pharma", "Energy", "Media", "Chemicals", "Furniture",
           "Electronics", "Beverages", "Apparel", "Metals", "Paper"]

LANDMARKS = {
    "US": ["Central Park", "the station", "City Hall", "the mall", "the river"],
    "India": ["the railway station", "City Mall", "the temple", "the bus stand", "MG Circle"],
    "France": ["la gare", "la mairie", "le parc", "l'eglise", "le marche"],
}

# Transliteration-ish single-token variants (India-flavoured)
TRANSLIT = [("ph", "f"), ("ee", "i"), ("oo", "u"), ("sh", "s"),
            ("w", "v"), ("z", "j"), ("kh", "k"), ("aa", "a")]


# ---------------------------------------------------------------------------
# Noise operators
# ---------------------------------------------------------------------------
def typo(word, r):
    if len(word) < 4:
        return word
    op = r.choice(["sub", "del", "ins", "swap"])
    i = r.randrange(len(word))
    letters = "abcdefghijklmnopqrstuvwxyz"
    if op == "sub":
        return word[:i] + r.choice(letters) + word[i + 1:]
    if op == "del":
        return word[:i] + word[i + 1:]
    if op == "ins":
        return word[:i] + r.choice(letters) + word[i:]
    j = min(i + 1, len(word) - 1)
    lst = list(word)
    lst[i], lst[j] = lst[j], lst[i]
    return "".join(lst)


def swap_abbr(text, pairs, r):
    """Randomly swap one abbreviation<->full-form occurrence in text."""
    cands = []
    for a, b in pairs:
        for pat in (a, b):
            for m in re.finditer(r"\b" + re.escape(pat) + r"\b", text):
                cands.append((m.start(), pat, b if pat == a else a))
    if not cands:
        return text
    pos, pat, rep = r.choice(cands)
    return text[:pos] + rep + text[pos + len(pat):]


def drop_token(text, r):
    toks = text.split()
    if len(toks) <= 3:
        return text
    i = r.randrange(1, len(toks) - 1)  # keep first/last token
    return " ".join(toks[:i] + toks[i + 1:])


def swap_word_order(text, r):
    toks = text.split()
    if len(toks) < 3:
        return text
    i, j = r.sample(range(len(toks)), 2)
    toks[i], toks[j] = toks[j], toks[i]
    return " ".join(toks)


def add_landmark(address, country, r):
    lm = r.choice(LANDMARKS[country])
    joiner = r.choice([", Near ", ", Opp. ", ", Behind "])
    if country == "France":
        joiner = r.choice([", pres de ", ", face a "])
    return address + joiner + lm


def transliterate(text, r):
    toks = text.split()
    long_toks = [t for t in toks if len(t) > 5 and t.isalpha()]
    if not long_toks:
        return text
    tok = r.choice(long_toks)
    src, dst = r.choice(TRANSLIT)
    if src in tok.lower():
        new = re.sub(src, dst, tok, count=1, flags=re.IGNORECASE)
        return text.replace(tok, new, 1)
    return text


def noisify(text, country, r, level, is_address):
    """Apply `level` random noise ops to a name or address string."""
    ops = [lambda t: swap_abbr(t, ABBR_PAIRS, r),
           lambda t: swap_abbr(t, SUFFIX_ABBR.get(country, []), r)]
    if is_address:
        ops += [lambda t: drop_token(t, r),
                lambda t: add_landmark(t, country, r)]
    else:
        ops += [lambda t: drop_token(t, r),
                lambda t: swap_word_order(t, r),
                lambda t: transliterate(t, r)]
    # typo op works on a random long token
    def typo_op(t):
        toks = t.split()
        idx = [i for i, w in enumerate(toks) if len(w) > 5]
        if not idx:
            return t
        i = r.choice(idx)
        toks[i] = typo(toks[i], r)
        return " ".join(toks)
    ops.append(typo_op)
    t = text
    for _ in range(level):
        t = r.choice(ops)(t)
    return t


def make_address(country, r):
    lex = COUNTRY[country]
    city = r.choice(lex["cities"])
    stype = r.choice(lex["street_types"])
    sname = r.choice(lex["street_names"])
    num = r.randint(1, 499)
    if country == "US":
        state = r.choice(lex["states"])
        return f"{num} {sname} {stype}, {city}, {state} {lex['zip_fmt'](r)}"
    if country == "India":
        area = r.choice(["Sector " + str(r.randint(1, 60)),
                         r.choice(lex["street_names"]) + " Layout"])
        return f"{num}, {sname} {stype}, {area}, {city} - {lex['zip_fmt'](r)}"
    return f"{num} {stype} {sname}, {lex['zip_fmt'](r)} {city}"


# ---------------------------------------------------------------------------
# Base entity generation
# ---------------------------------------------------------------------------
def make_base_entities(n, countries, weights, r, used_names):
    entities = []
    combos = [(p, c) for p in BRAND_PREFIX for c in BRAND_CORE]
    r.shuffle(combos)
    ci = 0
    while len(entities) < n:
        country = r.choices(countries, weights=weights)[0]
        lex = COUNTRY[country]
        pre, core = combos[ci % len(combos)]
        ci += 1
        brand = f"{pre}{core}"
        sector = r.choice(SECTORS)
        suffix = r.choice(lex["suffixes"])
        name = f"{brand} {sector} {suffix}"
        if name in used_names:
            name = f"{brand} {sector} {r.randint(2, 99)} {suffix}"
        used_names.add(name)
        entities.append({"country": country, "name": name,
                         "address": make_address(country, r),
                         "stem": f"{brand} {sector}"})
    return entities


def make_twins(s1_entities, frac, r):
    """Hard negatives: same brand+sector stem (often same normalized name),
    different address/city -> a genuinely different entity."""
    twins = []
    for ent in r.sample(s1_entities, max(1, int(frac * len(s1_entities)))):
        lex = COUNTRY[ent["country"]]
        suffix = r.choice(lex["suffixes"])
        addr = make_address(ent["country"], r)
        tries = 0
        while addr.split(",")[-1] == ent["address"].split(",")[-1] and tries < 10:
            addr = make_address(ent["country"], r)
            tries += 1
        twins.append({"country": ent["country"],
                      "name": f"{ent['stem']} {suffix}",
                      "address": addr, "stem": ent["stem"], "twin_of": True})
    return twins


# ---------------------------------------------------------------------------
# Split assembly
# ---------------------------------------------------------------------------
def build_split(n_s1, n_s23_target, countries, weights, seed, singleton_rate=0.2):
    r = random.Random(seed)
    used_names = set()
    s1_entities = make_base_entities(n_s1, countries, weights, r, used_names)
    # name-twin hard negatives (same stem, different entity/address)
    s1_entities += make_twins(s1_entities, frac=0.08, r=r)

    s1_rows, s2_rows, s3_rows = [], [], []
    truth = {}
    s2_i = s3_i = 0

    def add_s23(ent, noisy_name, noisy_addr):
        nonlocal s2_i, s3_i
        if r.random() < 0.5:
            s2_i += 1
            eid = f"S2-{s2_i:05d}"
            s2_rows.append((eid, noisy_name, noisy_addr, ent["country"]))
        else:
            s3_i += 1
            eid = f"S3-{s3_i:05d}"
            s3_rows.append((eid, noisy_name, noisy_addr, ent["country"]))
        return eid

    match_dist = [(1, 0.45), (2, 0.30), (3, 0.15), (4, 0.10)]

    def sample_n_matches():
        x = r.random()
        cum = 0.0
        for k, p in match_dist:
            cum += p
            if x < cum:
                return k
        return 1

    for i, ent in enumerate(s1_entities, start=1):
        s1_id = f"S1-{i:05d}"
        # S1 is the "reference": light noise only
        s1_name = noisify(ent["name"], ent["country"], r,
                          level=r.choice([0, 0, 1]), is_address=False)
        s1_addr = noisify(ent["address"], ent["country"], r,
                          level=r.choice([0, 0, 1]), is_address=True)
        s1_rows.append((s1_id, s1_name, s1_addr, ent["country"]))

        matched = []
        if r.random() >= singleton_rate:
            for _ in range(sample_n_matches()):
                nn = noisify(ent["name"], ent["country"], r,
                             level=r.randint(1, 3), is_address=False)
                na = noisify(ent["address"], ent["country"], r,
                             level=r.randint(1, 3), is_address=True)
                matched.append(add_s23(ent, nn, na))
        truth[s1_id] = matched

    # Distractor S2/S3 records that match nothing
    n_have = len(s2_rows) + len(s3_rows)
    n_distractors = max(0, n_s23_target - n_have)
    distractors = make_base_entities(n_distractors, countries, weights, r, used_names)
    for ent in distractors:
        nn = noisify(ent["name"], ent["country"], r, level=r.randint(1, 3), is_address=False)
        na = noisify(ent["address"], ent["country"], r, level=r.randint(1, 3), is_address=True)
        add_s23(ent, nn, na)

    return s1_rows, s2_rows, s3_rows, truth


def write_tsv(path, rows, header):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL,
                       lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def write_split(out_dir, split, s1_rows, s2_rows, s3_rows, truth=None):
    import os
    os.makedirs(out_dir, exist_ok=True)
    src_header = ["entity_id", "business_name", "business_address", "country"]
    write_tsv(f"{out_dir}/{split}_source1.tsv", s1_rows, src_header)
    write_tsv(f"{out_dir}/{split}_source2.tsv", s2_rows, src_header)
    write_tsv(f"{out_dir}/{split}_source3.tsv", s3_rows, src_header)
    if truth is not None:
        gt_rows = [(sid, ",".join(truth[sid])) for sid, _, _, _ in s1_rows]
        write_tsv(f"{out_dir}/{split}_ground_truth.tsv", gt_rows,
                  ["source1_entity_id", "matched_entity_ids"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="synthetic")
    args = ap.parse_args()

    train = build_split(n_s1=300, n_s23_target=500,
                        countries=["US", "India"], weights=[0.5, 0.5],
                        seed=args.seed)
    write_split(args.out, "train", *train)

    test = build_split(n_s1=150, n_s23_target=250,
                       countries=["US", "India", "France"], weights=[0.4, 0.4, 0.2],
                       seed=args.seed + 1)
    s1, s2, s3, _truth = test
    write_split(args.out, "test", s1, s2, s3, truth=None)

    n_single = sum(1 for v in train[3].values() if not v)
    multi = sum(1 for v in train[3].values() if len(v) > 1)
    print(f"train: {len(train[0])} S1, {len(train[1])} S2, {len(train[2])} S3, "
          f"singletons={n_single}, multi-match={multi}")
    tc = defaultdict(int)
    for _, _, _, c in s1:
        tc[c] += 1
    print(f"test:  {len(s1)} S1 ({dict(tc)}), {len(s2)} S2, {len(s3)} S3")
    print(f"wrote {args.out}/")


if __name__ == "__main__":
    main()
