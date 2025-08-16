#!/usr/bin/env python3
"""
Batch-screen 'Include/' and 'Maybe/' PDFs with GPT-4-Turbo.

Output: screen_fulltext_results.csv  (decision, confidence, rationale)
"""

import os, json, asyncio, pathlib, re, textwrap, time, hashlib
from typing import Tuple
from collections import defaultdict
import tiktoken  # pip install tiktoken
from lxml import etree
import pandas as pd
from tqdm.asyncio import tqdm as atqdm

from openai import AsyncOpenAI  # pip install openai==1.14.0
import aiohttp, async_timeout, asyncio, backoff
from pathlib import Path

# ---------------------------------------------------------------------
# ---------- CONFIG ----------------------------------------------------
BASE_DIR = pathlib.Path(__file__).parent
# PDF_SETS = ["Include", "Maybe"]
PDF_SETS = ["Final Papers"]

GROBID_URL = "http://localhost:8070/api/processFulltextDocument"
TEI_CACHE_DIR = BASE_DIR / "tei_cache_final"  # e.g. "~/SLR/tei_cache"
TEI_CACHE_DIR.mkdir(exist_ok=True)

MAX_PARALLEL_GROBID = 4  # <= GROBID’s default worker pool
MAX_GROBID_RETRIES = 3

grobid_sem = asyncio.Semaphore(MAX_PARALLEL_GROBID)

MODEL = "gpt-4-1106-preview"  # any non‑o GPT‑4 model – adjust if needed
TEMPERATURE = 0.0
MAX_TOKENS = 8192  # let GPT decide output ≤ 8 k (cheap)

# Inclusion/exclusion criteria
CRITERIA = textwrap.dedent(
    """
    Inclusion
      i1  Robotics domain focus.
      i2  Energy efficiency focus.
      i3  Addresses primarily SOFTWARE aspects, specifically implementation of runnable software (algorithm, control layer, OS/power lib) is the PRIMARY lever for saving energy.
      i4  Presents at least some EVALUATION, such as an empirical assessment or application to a concrete system (sim, real, or both), measuring the energy impact of the code implemented.

    Exclusion
      e1  Energy focus but NO runnable software component evaluated (simulation-only / hardware-only).
      e2  Mentions energy only as an example or secondary topic.
"""
).strip()

# Section labels already tuned in the previous step
METHOD_LABELS = {
    "methods",
    "methodology",
    "materials_and_methods",
    "materials and methods",
    "experimental_setup",
    "system_design",
    "implementation",
    "architecture",
    "proposed_method",
    "approach",
}

RESULT_LABELS = {
    "results",
    "evaluation",
    "findings",
    "performance",
    "experiment",
    "experimental_results",
}

# Fallback slice when no Methods found: first 3 000 tokens after Introduction
FALLBACK_LIMIT = 3000

# ---------------------------------------------------------------------
enc = tiktoken.get_encoding("cl100k_base")

MAX_PROMPT_TOKENS = 24000  # well under 128k; adjust if you move back to 8k

KEYWORDS = re.compile(
    r"\b(energy|power|battery|consum|efficien|efficient|"
    r"software|code|algorithm|implementation|method|evaluation|experiment)\b",
    re.I,
)


def keyword_trim(text: str, limit: int = MAX_PROMPT_TOKENS) -> str:
    """Return a keyword-filtered, token-capped version of `text`."""
    # 1. Short enough already?
    toks = enc.encode(text)
    if len(toks) <= limit:
        return text

    # 2. Keep only keyword paragraphs
    paras = [p for p in text.splitlines() if KEYWORDS.search(p)]
    if not paras:  # keyword filter too aggressive → fallback
        paras = text.splitlines()

    trimmed = " ".join(paras)
    toks = enc.encode(trimmed)
    if len(toks) > limit:
        trimmed = enc.decode(toks[:limit])

    return trimmed


client = AsyncOpenAI()  # uses OPENAI_API_KEY env var


def num_tokens(text: str) -> int:
    return len(enc.encode(text))


def _write_cache(cache_path: Path, tei_xml: str):
    sha = hashlib.sha256(tei_xml.encode()).hexdigest()
    cache_path.write_text(tei_xml, encoding="utf-8")
    meta = {"hash": sha, "created": time.time()}
    (cache_path.parent / (cache_path.stem + ".meta.json")).write_text(
        json.dumps(meta), encoding="utf-8"
    )
    return sha


@backoff.on_exception(
    backoff.expo,
    (aiohttp.ClientError, asyncio.TimeoutError),
    max_tries=MAX_GROBID_RETRIES,
)
async def grobid_async(pdf_path: Path) -> tuple[str, str]:
    """
    Return (tei_xml, sha256) for `pdf_path`, using on-disk cache.
    If the TEI is already cached, read it, verify its hash, and return it.
    If the cache is missing or a re-fetch is forced, call GROBID,
    cache the result, and return the new hash.
    """
    cache_path = TEI_CACHE_DIR / (pdf_path.stem + ".tei.xml")
    meta_path = cache_path.with_suffix(".meta.json")

    # ---------- 1. Cached TEI exists ----------
    if cache_path.exists() and meta_path.exists():
        tei_xml = cache_path.read_text(encoding="utf-8")
        old_meta = json.loads(meta_path.read_text())
        old_sha = old_meta.get("hash", "")
        new_sha = hashlib.sha256(tei_xml.encode()).hexdigest()

        # hash mismatch should be extremely rare; warn & update meta
        if old_sha != new_sha:
            print(
                f"[DIFF] Hash mismatch for {pdf_path.name} "
                f"(cached={old_sha[:8]} new={new_sha[:8]}) — keeping cached text."
            )
            _write_cache(cache_path, tei_xml)  # refresh meta file
        return tei_xml, new_sha

    # ---------- 2. Need to fetch from GROBID ----------
    async with grobid_sem:
        async with aiohttp.ClientSession() as sess:
            with open(pdf_path, "rb") as fh:
                data = aiohttp.FormData()
                data.add_field(
                    "input", fh, filename=pdf_path.name, content_type="application/pdf"
                )
                async with async_timeout.timeout(180):
                    async with sess.post(GROBID_URL, data=data) as resp:
                        resp.raise_for_status()
                        tei_xml = await resp.text()

    sha = _write_cache(cache_path, tei_xml)
    return tei_xml, sha


def extract_sections(tei_xml: str) -> Tuple[str, str]:
    """
    Return (methods_text, results_text).  Either can be '' if missing.
    """
    root = etree.fromstring(tei_xml.encode())
    ns = {"tei": "http://www.tei-c.org/ns/1.0"}
    methods, results = "", ""

    for div in root.xpath(".//tei:div", namespaces=ns):
        header = div.get("type", "").lower()

        # 1) by @type
        if header in METHOD_LABELS:
            methods = " ".join(div.xpath(".//text()"))
        elif header in RESULT_LABELS:
            results = " ".join(div.xpath(".//text()"))

        # 2) by <head> text fallback
        if not header:
            head_node = div.find(".//tei:head", namespaces=ns)
            if head_node is not None:
                head_text = "".join(head_node.itertext()).lower()
                if any(lbl in head_text for lbl in METHOD_LABELS):
                    methods = " ".join(div.xpath(".//text()"))
                elif any(lbl in head_text for lbl in RESULT_LABELS):
                    results = " ".join(div.xpath(".//text()"))

    # 3) fallback slice if no explicit methods
    if not methods:
        body = root.find(".//tei:text/tei:body", namespaces=ns)
        if body is not None:
            full_body_text = " ".join(body.xpath(".//text()"))
            # remove references tail
            full_body_text = full_body_text.split("references", 1)[0]
            tokens = enc.encode(full_body_text)
            slice_tokens = tokens[:FALLBACK_LIMIT]
            methods = keyword_trim(methods)
            results = keyword_trim(results)

    return methods, results


def build_prompt(methods: str, results: str) -> list[dict]:
    body = textwrap.dedent(
        f"""
      You are assisting a systematic literature review.

      Criteria:
      {CRITERIA}

      TASK:
      1. Decide include / maybe / exclude.
      2. Provide 1-2 sentence QUOTE that justifies the decision (from text below).
      3. Rate Rate confidence as:
        • high   → ≥90 % sure the decision is correct
        • medium → 60-89 %
        • low    → <60 %.

      Return JSON EXACTLY:
      {{
        "decision": "...",
        "confidence": "...",
        "quote": "..."      # one sentence proving runnable software is evaluated
      }}

      --- BEGIN STUDY TEXT ---
      METHODS
      {methods[:20000]}
      RESULTS
      {results[:20000]}
      --- END STUDY TEXT ---
    """
    ).strip()
    return [
        {
            "role": "system",
            "content": (
                "You are assisting a systematic literature review. "
                "Answer ONLY with a JSON object that matches the schema "
                '{"decision": "...", "confidence": "...", "quote": "..."} — '
                "no extra keys, no markdown."
                "If rules are ambiguous, choose decision = maybe and set P between 0.40-0.60."
            ),
        },
        {"role": "user", "content": body},
    ]


async def screen_one(pdf_path: Path, set_name: str) -> dict:
    try:
        tei, tei_hash = await grobid_async(pdf_path)
        meth, res = extract_sections(tei)
        prompt = build_prompt(meth, res)

        resp = await client.chat.completions.create(
            model=MODEL,
            temperature=TEMPERATURE,
            max_tokens=800,
            response_format={"type": "json_object"},
            messages=prompt,
        )
        data = json.loads(resp.choices[0].message.content)

        return {
            "set": set_name,
            "file": f"{pdf_path.parent.name}/{pdf_path.name}",
            "tei_hash": tei_hash,
            **data,
        }

    except Exception as e:
        return {
            "set": set_name,
            "file": f"{pdf_path.parent.name}/{pdf_path.name}",
            "tei_hash": "error",
            "decision": "error",
            "confidence": "low",
            "quote": str(e),
        }


async def screen_missing(missing_list):
    rows = []
    for full in missing_list:
        set_name, year, fname = full.split("/", 2)
        pdf_path = BASE_DIR / set_name / year / fname
        rows.append(await screen_one(pdf_path, set_name))
    return rows


async def main():
    tasks = []
    for set_name in PDF_SETS:
        for pdf in sorted((BASE_DIR / set_name).glob("*/*.pdf")):
            tasks.append(screen_one(pdf, set_name))

    print(f"Submitting {len(tasks)} PDFs to GPT-4 …")
    rows = []
    for coro in atqdm.as_completed(tasks):
        rows.append(await coro)

    df = pd.DataFrame(rows)
    out_path = BASE_DIR / "screen_final.csv"
    # out_path = BASE_DIR / "consensus_snow_run3.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")

    # Quick console roll‑up
    print("\n=== Counts by decision ===")
    print(df.groupby(["set", "decision"]).size())

    # -------------------------------------------------------------
    # Accurate reconciliation of expected vs processed PDFs
    # -------------------------------------------------------------
    df = pd.read_csv(out_path)

    # What we *actually* processed (prefix set/ to the saved 'file' column)
    processed = {f"{row.set}/{row.file}" for row in df.itertuples(index=False)}

    # Everything that exists on disk (recursive, in case you add subdirs later)
    expected = {
        f"{s}/{p.parent.name}/{p.name}"
        for s in PDF_SETS
        for p in (BASE_DIR / s).glob("*/*.pdf")
    }

    missing = sorted(expected - processed)
    extra = sorted(processed - expected)

    print(f"\nProcessed: {len(processed)}  |  Expected: {len(expected)}")
    if missing:
        print("\n[WARN] Missing PDFs (not screened):")
        for m in missing:
            print("  ", m)
    if extra:
        print("\n[WARN] Rows in CSV that don't exist on disk anymore:")
        for e in extra:
            print("  ", e)
    if not missing and not extra:
        print("\nAll PDFs truly accounted for.")


if __name__ == "__main__":
    asyncio.run(main())
