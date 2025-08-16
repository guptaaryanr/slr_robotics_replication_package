#!/usr/bin/env python3
"""
extract_fields.py  --  Part 1 of data-extraction pipeline (updated)
---------------------------------------------------------
• For every PDF in 'Final Papers/', run GROBID ➜ TEI cache
• Isolate METHODS + RESULTS + DISCUSSION sections
• Call GPT-4-1106-preview THREE times with prompt variants A/B/C (temperature=0)
• Field-level consensus: 2-of-3 majority; else confidence-weighted; else 'other'
• Save per-paper JSONs + aggregated consensus + aggregate CSV

Requirements: pip install openai lxml pandas tqdm requests
"""

import json, re, pathlib, asyncio
from collections import Counter, defaultdict
from lxml import etree
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI

# ---------------- CONFIG ----------------
BASE = pathlib.Path(__file__).parent
PDF_DIR = BASE / "Final Papers"
TEI_CACHE = BASE / "tei_cache"
JSON_DIR = BASE / "extracted_json"
JSON_DIR.mkdir(exist_ok=True)
CSV_OUT = "extraction_table_data.csv"

MODEL = "gpt-4-1106-preview"
TEMPERATURE = 0
N_RUNS = 3  # three passes (A/B/C) for consensus
SEED = 42
RATE_DELAY = 0.2  # polite pause to OpenAI

GROBID_URL = "http://localhost:8070/api/processFulltextDocument"

client = AsyncOpenAI()

# ------------ GPT schema ---------------
FUNCTION_DEF = {
    "name": "extract_study",
    "parameters": {
        "type": "object",
        "properties": {
            "metric": {
                "type": "string",
                "description": "The primary metric of energy evaluation, e.g., 'average power (W)'",
                "enum": [
                    "aggregate energy/power",
                    "task-normalized energy",
                    "performance per energy",
                    "relative change versus baseline",
                    "composite objective including energy",
                    "physics-based integrals",
                    "validation statistics",
                    "operational proxies",
                ],
            },
            "domain": {
                "type": "string",
                "description": "The application domain of the robotic system, e.g., 'robot exploration'",
                "enum": [
                    "robot exploration",
                    "industrial",
                    "service or domestic",
                    "aerial",
                    "aquatic",
                    "modular",
                    "additive manufacturing",
                    "iot power",
                    "swarm or multi-robot",
                    "mixed",
                ],
            },
            "major_consumers": {
                "type": "string",
                "description": "The major energy consumers in the robotic system, e.g., 'motors and actuators'",
                "enum": [
                    "motors and actuators",
                    "sensors",
                    "computing and controllers",
                    "communication subsystem",
                    "battery and power electronics",
                    "mechanical motion pattern",
                    "idle and stand-by overhead",
                ],
            },
            "evaluation_type": {
                "type": "string",
                "description": "The type of evaluation performed, e.g., 'simulation'",
                "enum": [
                    "simulation",
                    "physical",
                    "hybrid",
                ],
            },
            "energy_model": {
                "type": "string",
                "description": "The energy model used in the evaluation, e.g., 'representational'",
                "enum": [
                    "abstract",
                    "representational",
                ],
            },
            "techniques": {
                "type": "array",
                "description": "Techniques used to optimize energy efficiency, e.g., 'motion planning'",
                "items": {
                    "type": "string",
                    "enum": [
                        "power management and idle control",
                        "motion and trajectory optimization",
                        "computation allocation and scheduling",
                        "learning or predictive optimization",
                        "communication and data efficiency",
                        "hardware or morphology and harvesting",  # not used anywhere in the corpus apparently
                    ],
                },
            },
            "qa_tradeoff": {
                "type": "string",
                "description": "The quality-assurance trade-off considered, e.g., 'performance vs energy'",
                "enum": [
                    "performance vs energy",
                    "reliability vs energy",
                    "accuracy vs energy",
                    "coverage vs energy",
                    "mission quality vs energy",
                    "safety vs energy",
                    "stability vs energy",
                    "maintainability vs energy",
                    "code complexity vs energy",
                    "cost vs energy",
                ],
            },
            "confidence": {"type": "number"},
            "quote_per_field": {"type": "object"},
        },
        "required": ["metric", "domain", "evaluation_type", "confidence"],
    },
}

PROMPT_SYS = (
    "You are a meticulous research assistant in charge of extracting structured data for an SLR on energy-efficient robotics software. "
    "Use ONLY the allowed enumeration labels. If uncertain, choose 'other'. Return a JSON function call that matches the schema exactly."
)

# ---- Prompt variants (A/B/C) ----
PROMPT_USER_A = (
    "Extract from METHODS, RESULTS, DISCUSSION, and any GRAPHS/FIGURES/TABLES describing the evaluation. "
    "Populate: metric, domain, major_consumers, evaluation_type, energy_model, techniques, qa_tradeoff, confidence, quote_per_field. "
    "Rules: (1) Use EXACT enum labels; (2) choose EXACTLY one label for single-valued fields; (3) 'techniques' is a list of enum labels; "
    "(4) if uncertain, prefer 'other'; (5) JSON only."
)

PROMPT_USER_B = (
    "Task: Produce JSON for metric, domain, major_consumers, evaluation_type, energy_model, techniques, qa_tradeoff, confidence, quote_per_field. "
    "Guidelines: One label per categorical field (except 'techniques' which is a list). "
    "Do NOT invent labels; use only allowed enumerations. If the paper does not use an explicit energy model, set energy_model='other'. "
    "Confidence is in [0,1]. JSON only."
)

PROMPT_USER_C = (
    "You will output JSON per schema with enum-only labels. If uncertain, use 'other'. JSON only.\n\n"
    "Example 1 (condensed):\n"
    "Text: 'evaluated in Gazebo; measured average power (W); technique: motion planning; trade-off: performance vs energy'\n"
    'Output: {"metric":"average power (W)","domain":"robot exploration","major_consumers":"motors and actuators",'
    '"evaluation_type":"simulation","energy_model":"representational","techniques":["motion and trajectory optimization"],'
    '"qa_tradeoff":"performance vs energy","confidence":0.78,"quote_per_field":{}}\n\n'
    "Example 2 (condensed):\n"
    "Text: 'hybrid evaluation; DVFS and duty-cycling; no explicit model; trade-off: reliability vs energy'\n"
    'Output: {"metric":"energy consumption (Wh)","domain":"service or domestic","major_consumers":"computing and controllers",'
    '"evaluation_type":"hybrid","energy_model":"other","techniques":["power management and idle control"],'
    '"qa_tradeoff":"reliability or accuracy vs energy","confidence":0.70,"quote_per_field":{}}\n\n'
    "Now extract for this paper:"
)

PROMPTS = [PROMPT_USER_A, PROMPT_USER_B, PROMPT_USER_C]


# ------------ helpers ------------------
def norm(text):
    return re.sub(r"\s+", " ", text.strip())


def tei_path(pdf: pathlib.Path) -> pathlib.Path:
    return TEI_CACHE / (pdf.stem + ".tei.xml")


def get_or_create_tei(pdf: pathlib.Path) -> etree._Element:
    tei_p = tei_path(pdf)
    if tei_p.exists():
        return etree.parse(str(tei_p)).getroot()
    # call GROBID
    with open(pdf, "rb") as fh:
        import requests

        r = requests.post(GROBID_URL, files={"input": fh}, timeout=180)
    r.raise_for_status()
    TEI_CACHE.mkdir(exist_ok=True)
    tei_p.write_bytes(r.content)
    return etree.fromstring(r.content)


METHOD_HEADS = r"(method|approach|materials?|evaluation design)"
RESULT_HEADS = r"(result|experiment|findings|performance)"
DISCUSS_HEADS = r"(discussion|analysis|conclusion|threats)"
SEC_RE = re.compile(f"{METHOD_HEADS}|{RESULT_HEADS}|{DISCUSS_HEADS}", re.I)


def extract_sections(root):
    """
    Return concatenated text of any <div> whose <head> matches
    Methods / Results / Discussion patterns. Fallback: first 10k tokens.
    """
    ns = {"tei": "http://www.tei-c.org/ns/1.0"}
    chunks = []
    for div in root.xpath(".//tei:div", namespaces=ns):
        head_txt = " ".join(div.xpath("./tei:head//text()", namespaces=ns)).strip()
        if SEC_RE.search(head_txt):
            body = " ".join(div.xpath(".//text()", namespaces=ns))
            chunks.append(body)
    text = " ".join(chunks).strip()
    if not text:
        full = " ".join(root.xpath(".//tei:body//text()", namespaces=ns))
        text = " ".join(full.split()[:10000])
    return re.sub(r"\s+", " ", text)


# ------------- GPT call ----------------
async def gpt_extract(chunk: str, user_prompt: str) -> dict:
    msgs = [
        {"role": "system", "content": PROMPT_SYS},
        {"role": "user", "content": user_prompt + "\n\n" + chunk},
    ]
    rsp = await client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        seed=SEED,
        response_format={"type": "json_object"},
        tools=[{"type": "function", "function": FUNCTION_DEF}],
        tool_choice={"type": "function", "function": {"name": "extract_study"}},
        messages=msgs,
    )
    # Prefer function call arguments if present; else fall back to content
    msg = rsp.choices[0].message
    if (
        msg.tool_calls
        and msg.tool_calls[0].function
        and msg.tool_calls[0].function.arguments
    ):
        raw_json = msg.tool_calls[0].function.arguments
    else:
        raw_json = msg.content
    try:
        return json.loads(raw_json) if raw_json else {}
    except Exception:
        return {}


# -------- aggregation helpers ----------
SINGLE = ["domain", "major_consumers", "evaluation_type", "energy_model", "qa_tradeoff"]
LISTF = ["techniques"]
FREE = ["metric"]


def aggregate_runs(outs: list) -> dict:
    chosen = {}
    votes_meta = {}
    needs_review = False

    # single-label fields with 2-of-3 majority, else confidence-weighted
    for f in SINGLE:
        votes = [o.get(f) for o in outs if o.get(f)]
        cnt = Counter(votes)
        votes_meta[f] = dict(cnt)
        label, count = cnt.most_common(1)[0] if cnt else ("other", 0)
        if count >= 2:
            chosen[f] = label
        else:
            w = defaultdict(float)
            for o in outs:
                if o.get(f):
                    w[o[f]] += float(o.get("confidence", 0.0) or 0.0)
            if w:
                chosen[f] = max(w.items(), key=lambda kv: kv[1])[0]
            else:
                chosen[f] = "other"
            needs_review = True

    # list field: keep items that appear in ≥2 runs
    if "techniques" in LISTF:
        sets = [set(o.get("techniques", []) or []) for o in outs]
        union = set().union(*sets)
        keep = [t for t in union if sum(t in s for s in sets) >= 2]
        chosen["techniques"] = keep or sorted(list(union))[:3]  # conservative fallback

    # free text: take from highest-confidence run
    best_idx = max(
        range(len(outs)), key=lambda i: float(outs[i].get("confidence", 0.0) or 0.0)
    )
    chosen["metric"] = outs[best_idx].get("metric")

    chosen["confidence"] = max(float(o.get("confidence", 0.0) or 0.0) for o in outs)
    chosen["needs_review"] = needs_review
    chosen["_votes"] = votes_meta
    return chosen


# ------------ main async loop ----------
async def process_pdf(pdf):
    tei_root = get_or_create_tei(pdf)
    chunk = extract_sections(tei_root)
    if not chunk:
        print(f"[WARN] No extractable sections in {pdf.name}")
        return None
    print(f"{pdf.name} — extracted {len(chunk.split())} words")

    outs = []
    for i in range(N_RUNS):
        outs.append(await gpt_extract(chunk, PROMPTS[i]))
        await asyncio.sleep(RATE_DELAY)

    # save per-run JSONs
    stem = pdf.stem
    for i, o in enumerate(outs, start=1):
        (JSON_DIR / f"{stem}_run{i}.json").write_text(json.dumps(o, indent=2))

    # aggregate to consensus and save
    aggregated = aggregate_runs(outs)
    cons_payload = {
        "runs": outs,
        "chosen": aggregated,
        "votes": aggregated.get("_votes", {}),
        "needs_review": aggregated.get("needs_review", False),
    }
    (JSON_DIR / f"{stem}_consensus.json").write_text(json.dumps(cons_payload, indent=2))

    aggregated["file"] = pdf.name
    # remove internal meta before CSV if you prefer:
    aggregated.pop("_votes", None)
    return aggregated


async def main():
    pdfs = list(PDF_DIR.glob("*.pdf"))
    rows = await tqdm_asyncio.gather(*(process_pdf(p) for p in pdfs))
    rows = [r for r in rows if r]  # drop None
    pd.DataFrame(rows).to_csv(CSV_OUT, index=False)
    print(f"Saved extraction table → {CSV_OUT}")


if __name__ == "__main__":
    asyncio.run(main())
