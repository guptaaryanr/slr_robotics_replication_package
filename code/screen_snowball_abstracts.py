#!/usr/bin/env python3
"""
screen_snowball_abstracts.py
----------------------------
• Merge backward + forward snowball lists
• Fetch abstracts via OpenAlex (skip rows with no abstract)
• Screen abstracts with the *same prompt* used in process_papers.py
• Keep "include/high" hits
• Output: snowball_abstracts_screened.csv
"""

import csv, json, time, requests, pandas as pd
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

# ------------------ CONFIG --------------------------------------------------
BACKWARD_CSV = "backward_refs_raw.csv"
FORWARD_CSV = "forward_refs_raw.csv"
OUT_CSV = "snowball_abstracts_screened.csv"
OPENALEX = "https://api.openalex.org/works/doi:"
RATE_DELAY = 0.6  # ≤ 100 req/min keeps us safe
MODEL = "gpt-4-1106-preview"  # cheap, 128k; match earlier flow
TEMPERATURE = 0
SEED = 42
# ----------------------------------------------------------------------------

# 0) bring in the old prompt EXACTLY as in process_papers.py
PROMPT_TEMPLATE = """
### ROLE
You are a meticulous research assistant screening abstracts
for a systematic literature review on energy-efficient robotics
software.

### CRITERIA
Inclusion:
- i1: Focuses on robotics.
- i2: Focuses on energy efficiency.
- i3: Focuses on software aspects of the robotic system (executable code, algorithm, control layer, middleware, OS/power lib, etc.).
- i4: Provides a certain level of evaluation on the software implementation (e.g., empirical assessment, case study, practical implementation).
- i5: Is a peer-reviewed study.
- i6: Is written in English.

Exclusion:
- e1: Does not explicitly deal with any software aspect, despite energy focus.
- e2: Energy efficiency is only used as an example and not as the primary source of study.
- e3: Is a Secondary or Tertiary study (e.g., a review, survey).
- e4: Is not a Journal Article, Conference Paper, or Book Chapter.
- e5: Is not available as full-text.

### ABSTRACT
{abstract}

### RESPOND WITH JSON ONLY
{{"decision":"include|exclude|maybe","confidence":"high|med|low"}}
"""

client = OpenAI()

# 1) merge and deduplicate ----------------------------------------------------
df_back = pd.read_csv(BACKWARD_CSV)
df_fwd = pd.read_csv(FORWARD_CSV)
df_all = pd.concat([df_back, df_fwd], ignore_index=True)
df_all.drop_duplicates(subset="doi", inplace=True)
df_all["doi"] = df_all["doi"].str.lower().str.strip()
print("Merged refs (unique DOI):", len(df_all))

# 2) fetch abstracts ----------------------------------------------------------
records = []
for rec in tqdm(df_all.to_dict(orient="records"), desc="Fetch abstracts"):
    doi = rec["doi"]
    try:
        r = requests.get(OPENALEX + doi, timeout=20)
        if r.status_code != 200:
            continue
        data = r.json()
        abs_inv = data.get("abstract_inverted_index")
        if not abs_inv:
            continue  # skip rows w/o abstract
        # reconstruct abstract
        abstract = " ".join(
            word
            for word, idx in sorted(
                ((w, i) for w, idxs in abs_inv.items() for i in idxs),
                key=lambda t: t[1],
            )
        )
        rec["abstract"] = abstract
        records.append(rec)
    except Exception:
        continue
    time.sleep(RATE_DELAY)

print("Abstracts fetched :", len(records))

# 3) GPT-4 abstract screen ----------------------------------------------------
rows_keep = []
for rec in tqdm(records, desc="GPT-4 screen"):
    prompt = PROMPT_TEMPLATE.format(abstract=rec["abstract"])
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        seed=SEED,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    data = json.loads(resp.choices[0].message.content)
    rec.update(data)
    if data["decision"] == "include" and data["confidence"] == "high":
        rows_keep.append(rec)

print("Include/high kept :", len(rows_keep))

# 4) write output -------------------------------------------------------------
pd.DataFrame(rows_keep).to_csv(OUT_CSV, index=False, quoting=csv.QUOTE_MINIMAL)
print(f"Saved {len(rows_keep)} rows → {OUT_CSV}")
