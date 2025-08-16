#!/usr/bin/env python3
"""
forward_snowball.py
-------------------
1. Detect the 55 core PDFs     (Include/*/*.pdf + Maybe/*/*.pdf, minus */Exclude/*)
2. Get their DOIs (runs GROBID if needed)
3. Build EXCLUDED sets:
     • DOIs & titles of All Papers
     • DOIs already in refs_2020plus.csv  (backward snowball)
4. Query OpenAlex 'cited-by' for each core DOI
5. Keep citing works  up to 2024, drop duplicates, write forward_refs_raw.csv
"""

import re, csv, time, unicodedata, requests, pathlib
from collections import OrderedDict
from lxml import etree
from tqdm import tqdm

# ---------- paths -----------------------------------------------------
BASE = pathlib.Path(__file__).parent
INCLUDE = BASE / "Include"
MAYBE = BASE / "Maybe"
ALL_PAPERS = BASE / "All Papers"
TEI_CACHE = BASE / "tei_cache"
BACKWARD_CSV = BASE / "refs_2020plus.csv"
OUT_CSV = "forward_refs_raw.csv"
GROBID_URL = "http://localhost:8070/api/processFulltextDocument"
OPENALEX_W = "https://api.openalex.org/works/https://doi.org/"
YEAR_MAX = 2099

ns = {"tei": "http://www.tei-c.org/ns/1.0"}
YEAR_RE = re.compile(r"\b(20[0-9]{2})\b")


# ---------------- helpers --------------------------------------------
def norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", text.lower())


def grobid_fetch(pdf: pathlib.Path) -> etree._Element:
    import requests

    with open(pdf, "rb") as fh:
        r = requests.post(GROBID_URL, files={"input": fh}, timeout=180)
    r.raise_for_status()
    return etree.fromstring(r.content)


def tei_path_for(pdf: pathlib.Path) -> pathlib.Path:
    return TEI_CACHE / (pdf.stem + ".tei.xml")


def get_or_create_tei(pdf: pathlib.Path) -> etree._Element:
    tei_p = tei_path_for(pdf)
    if tei_p.exists():
        return etree.parse(str(tei_p))
    # call GROBID, save cache
    root = grobid_fetch(pdf)
    tei_p.write_bytes(etree.tostring(root, encoding="utf-8"))
    return etree.ElementTree(root)


def doi_from_tei(root) -> str:
    doi = root.xpath("//tei:idno[@type='DOI']/text()", namespaces=ns)
    return doi[0].lower().strip() if doi else ""


def title_from_tei(root) -> str:
    ttl = root.xpath("//tei:titleStmt/tei:title[@level='a']//text()", namespaces=ns)
    return " ".join(ttl).strip()


# ---------------- 1. locate 55 core PDFs ------------------------------
CORE_PDFS = [
    p
    for root in (INCLUDE, MAYBE)
    for p in root.glob("*/*.pdf")
    if "/Exclude/" not in str(p)
]
print("Core PDFs detected :", len(CORE_PDFS))  # should be 55

# ---------------- 2. extract their DOIs & titles ----------------------
CORE_DOIS, CORE_TITLES = set(), set()
for pdf in CORE_PDFS:
    root = get_or_create_tei(pdf)
    doi = doi_from_tei(root)
    ttl = title_from_tei(root)
    if doi:
        CORE_DOIS.add(doi)
    if ttl:
        CORE_TITLES.add(norm(ttl))
print("Core DOIs gathered :", len(CORE_DOIS))

# ---------------- 3. build exclusion sets ----------------------------
# A) DOIs & titles from All Papers
EXCL_DOIS, EXCL_TITLES = set(), set()
for pdf in ALL_PAPERS.glob("**/*.pdf"):
    stem = norm(pdf.stem.replace("_", " "))
    EXCL_TITLES.add(stem)

# B) DOIs from backward snowball list
if BACKWARD_CSV.exists():
    EXCL_DOIS.update(
        d.lower().strip()
        for d in pathlib.Path(BACKWARD_CSV).read_text().splitlines()[1:]  # skip header
        if d
    )

EXCL_DOIS.update(CORE_DOIS)
EXCL_TITLES.update(CORE_TITLES)

print("Excluded DOIs       :", len(EXCL_DOIS))
print("Excluded titles     :", len(EXCL_TITLES))


# ---------------- 4. OpenAlex cited‑by lookup ------------------------
def citing_works(doi):
    url = OPENALEX_W + doi
    try:
        w = requests.get(url, timeout=20).json()
    except Exception:
        return []
    api = w.get("cited_by_api_url")
    if not api:
        return []
    results = []
    while api:
        pg = requests.get(api, timeout=20).json()
        for work in pg["results"]:
            yr = work.get("publication_year")
            if yr and yr <= YEAR_MAX:
                w_doi = work.get("doi", "").lower().strip()
                ttl = work.get("title", "")
                if not w_doi or w_doi in EXCL_DOIS:
                    continue
                if norm(ttl) in EXCL_TITLES:
                    continue
                results.append({"year": yr, "title": ttl, "doi": w_doi})
        api = pg["meta"].get("next_cursor_url")
        time.sleep(0.25)
    return results


unique, rows = OrderedDict(), []
candidates_seen = 0

for doi in tqdm(CORE_DOIS, desc="Forward snowball"):
    for ref in citing_works(doi):
        candidates_seen += 1
        if ref["doi"] not in unique:
            unique[ref["doi"]] = ref

print(f"\nCiting works fetched  : {candidates_seen}")
print(f"After all exclusions  : {len(unique)} kept")
# print(next(iter(CORE_DOIS)))


# ---------------- 5. write -------------------------------------------
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["year", "title", "doi"])
    w.writeheader()
    w.writerows(unique.values())

print(f"\nSaved {len(unique)} forward-snowballed refs → {OUT_CSV}")
