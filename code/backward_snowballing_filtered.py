#!/usr/bin/env python3
"""
refs_2020plus.py  -  robust & tolerant

• Scans *all* TEIs in tei_cache/.
• Drops any TEI whose PDF now sits in */Exclude/*  (if the PDF still exists).
• Keeps refs with DOI & year >= 2020.
• Removes refs already in the primary corpus (by DOI or title normalised).
• Writes refs_2020plus.csv.
"""

import re, csv, unicodedata, pathlib
from lxml import etree
from collections import OrderedDict

BASE = pathlib.Path(__file__).parent
TEI_CACHE = BASE / "tei_cache"
INCLUDE = BASE / "Include"
MAYBE = BASE / "Maybe"
OUT_CSV = "refs_2020plus.csv"
YEAR_RE = re.compile(r"\b(20[0-9]{2})\b")


def norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", t.lower())


# ---------- which stems must we skip? (any PDF in */Exclude/*) ----------
SKIP_STEMS = {
    p.stem.lower() for root in (INCLUDE, MAYBE) for p in root.glob("*/*/Exclude/*.pdf")
}

# ---------- primary DOI & title sets (from all *remaining* TEIs) ----------
PRIMARY_DOIS, PRIMARY_TITLES = set(), set()
ns = {"tei": "http://www.tei-c.org/ns/1.0"}

for tei in TEI_CACHE.glob("*.tei.xml"):
    if tei.stem.lower() in SKIP_STEMS:  # skip excluded paper
        continue
    root = etree.parse(str(tei))
    doi = root.xpath("//tei:idno[@type='DOI']/text()", namespaces=ns)
    if doi:
        PRIMARY_DOIS.add(doi[0].lower().strip())
    title = root.xpath("//tei:titleStmt/tei:title[@level='a']//text()", namespaces=ns)
    if title:
        PRIMARY_TITLES.add(norm(" ".join(title)))

# ---------- collect references -----------------------------------------
unique = OrderedDict()

for tei in TEI_CACHE.glob("*.tei.xml"):
    if tei.stem.lower() in SKIP_STEMS:
        continue
    root = etree.parse(str(tei))
    for bibl in root.xpath(".//tei:biblStruct", namespaces=ns):
        title = " ".join(bibl.xpath(".//tei:title//text()", namespaces=ns)).strip()
        if not title or norm(title) in PRIMARY_TITLES:
            continue
        doi = bibl.xpath(".//tei:idno[@type='DOI']/text()", namespaces=ns)
        if not doi:
            continue
        doi = doi[0].lower().strip()
        if doi in PRIMARY_DOIS:
            continue
        # year filter
        when = bibl.xpath(".//tei:date/@when", namespaces=ns)
        year = int(when[0][:4]) if when else None
        if year is None:
            txt = " ".join(bibl.xpath(".//tei:date//text()", namespaces=ns))
            m = YEAR_RE.search(txt)
            year = int(m.group(1)) if m else None
        if year is None or year < 2020:
            continue
        key = norm(title)
        unique.setdefault(key, {"year": year, "title": title, "doi": doi})

# ---------- write -------------------------------------------------------
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["year", "title", "doi"])
    w.writeheader()
    w.writerows(unique.values())

print(f"Wrote {len(unique)} unique refs ≥2020 → {OUT_CSV}")
