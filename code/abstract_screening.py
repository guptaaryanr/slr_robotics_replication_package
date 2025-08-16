import os
import json
import pandas as pd
import pdfplumber
from openai import OpenAI
import time

# --- Configuration ---
TEST_MODE = False  # Set to True for testing with a limited sample size
TEST_SAMPLE_SIZE = 1

FINAL_OUTPUT_FILE = (
    "final_analysis_test.csv" if TEST_MODE else "final_analysis_full.csv"
)
BASE_PAPERS_DIR = "All Papers"
API_KEY_FILE = "API.txt"


# --- Functions ---
def load_api_key(filepath):
    try:
        with open(filepath, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: API key file not found at '{filepath}'")
        return None


def extract_first_page_text(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if not pdf.pages:
                return "Error: PDF has no pages."
            return pdf.pages[0].extract_text(x_tolerance=1, y_tolerance=1)
    except Exception as e:
        return f"Error opening or processing PDF file: {e}"


def llm_extract_abstract(full_text, client):
    if "Error:" in full_text:
        return full_text
    extraction_prompt = f"""
    You are a text extraction expert. From the following text from a research paper, find and return ONLY the abstract or summary section from the research paper's first page. The section may not have a title, but it will always be present and it will be near the start of the research paper. Do not include the word 'Abstract' or whatever the section is titled. Do not add any commentary. If no abstract is present, return only the string 'Error: No abstract found by LLM.'.
    ### TEXT FROM PAGE 1 ###
    {full_text}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error during LLM extraction: {e}"


def analyze_abstract(abstract, client):
    if "Error:" in abstract:
        return {"error": abstract}
    master_prompt_template = """
    ### ROLE ###
    You are a meticulous research assistant performing a systematic literature review. Your analysis must be objective and based solely on the provided abstract.

    ### CONTEXT & CRITERIA ###
    Your goal is to identify primary studies on energy efficiency in robotics software. A study must meet all inclusion criteria to be considered.

    **Inclusion Criteria:**
    - i1: Focusses on robotics.
    - i2: Focusses on energy efficiency as a primary goal.
    - i3: Focusses on software aspects of the robotic system.
    - i4: Provides a certain level of evaluation (e.g., empirical assessment, case study).

    **Exclusion Criteria:**
    - e1: Does not explicitly deal with any software aspect (e.g., hardware-only).
    - e2: Energy efficiency is only a minor, secondary mention.
    - e3: Is a review, survey, or other secondary/tertiary study.

    ### TASK ###
    1.  Read the abstract below carefully.
    2.  Analyze it against the criteria.
    3.  Fill out the JSON template with your findings.

    ### INSTRUCTIONS ###
    - For each field in the template, provide a concise summary.
    - The "supporting_quote" field MUST be an exact quote from the abstract.
    - If information for a field is not mentioned in the abstract, you MUST write "Not Mentioned".
    - Provide your overall recommendation based on the criteria.

    ### ABSTRACT ###
    {abstract}

    ### OUTPUT ###
    Please provide your response in a single, valid JSON format:
    ```json
    {{
        "energy_focus": {{
            "stated_goal": "",
            "supporting_quote": ""
        }},
        "software_focus": {{
            "core_contribution": "",
            "supporting_quote": ""
        }},
        "evaluation": {{
            "evaluation_method": "",
            "supporting_quote": ""
        }},
        "recommendation": {{
            "decision": "Include | Exclude | Maybe",
            "justification": ""
        }}
    }}
    ```
    """
    try:
        prompt = master_prompt_template.format(abstract=abstract)
        response = client.chat.completions.create(
            model="gpt-4", messages=[{"role": "user", "content": prompt}], temperature=0
        )
        # if not response.choices[0].message.content.strip().startswith("{"):
        #     # print("\n--- LLM Response ---")
        #     print("yoyohoneysingh")
        #     # print("\n--- End LLM Response ---\n")
        response_text = response.choices[0].message.content
        # if not response_text or not response_text.strip().startswith("{"):
        #     return {"error": "Analyzer LLM returned empty or non-JSON response"}
        json_str = response_text[response_text.find("{") : response_text.rfind("}") + 1]
        return json.loads(json_str)
    except Exception as e:
        return {"error": f"Error parsing analyzer response: {e}"}


# --- Main Execution ---
all_pdf_paths = set()
for root, _, files in os.walk(BASE_PAPERS_DIR):
    for name in files:
        if name.lower().endswith(".pdf"):
            all_pdf_paths.add(os.path.join(root, name))
try:
    df = pd.read_csv(ORIGINAL_RESULTS_FILE)
    processed_paths = set(
        df.apply(
            lambda row: os.path.join(
                BASE_PAPERS_DIR, str(row["year"]), row["filename"]
            ),
            axis=1,
        )
    )
    failed_filter = (df["abstract_extraction_status"] != "Success") | (
        df["error"].notnull()
    )
    failed_paths = set(
        df[failed_filter].apply(
            lambda row: os.path.join(
                BASE_PAPERS_DIR, str(row["year"]), row["filename"]
            ),
            axis=1,
        )
    )
except FileNotFoundError:
    processed_paths, failed_paths = set(), set()

missed_paths = all_pdf_paths - processed_paths
paths_to_process = sorted(list(failed_paths.union(missed_paths)))

# --- NEW: Slice list for Test Mode ---
if TEST_MODE:
    print(f"--- RUNNING IN TEST MODE ON UP TO {TEST_SAMPLE_SIZE} FILES ---")
    paths_to_process = paths_to_process[:TEST_SAMPLE_SIZE]

if not paths_to_process:
    print("No failed or missing files found to process.")
    exit()

print(f"Total files to process: {len(paths_to_process)}\n")
API_KEY = load_api_key(API_KEY_FILE)
if not API_KEY:
    exit()
client = OpenAI(api_key=API_KEY)

final_results = []
for paper_path in paths_to_process:
    path_parts = paper_path.split(os.sep)
    year, filename = path_parts[-2], path_parts[-1]

    print(f"  -> Final processing for {filename} from year {year}...")

    first_page_text = extract_first_page_text(paper_path)
    clean_abstract = llm_extract_abstract(first_page_text, client)
    # print(clean_abstract)
    time.sleep(1)

    llm_output = analyze_abstract(clean_abstract, client)

    new_result = {
        "year": year,
        "filename": filename,
        "abstract_extraction_status": (
            "Success" if "Error:" not in clean_abstract else clean_abstract
        ),
        "energy_goal": llm_output.get("energy_focus", {}).get("stated_goal"),
        "energy_quote": llm_output.get("energy_focus", {}).get("supporting_quote"),
        "software_contribution": llm_output.get("software_focus", {}).get(
            "core_contribution"
        ),
        "software_quote": llm_output.get("software_focus", {}).get("supporting_quote"),
        "evaluation_method": llm_output.get("evaluation", {}).get("evaluation_method"),
        "evaluation_quote": llm_output.get("evaluation", {}).get("supporting_quote"),
        "recommendation": llm_output.get("recommendation", {}).get("decision"),
        "justification": llm_output.get("recommendation", {}).get("justification"),
        "error": llm_output.get("error"),
    }
    final_results.append(new_result)
    time.sleep(1)

rerun_df = pd.DataFrame(final_results)
rerun_df.to_csv(FINAL_OUTPUT_FILE, index=False)

print(f"\nFinal processing complete. ✅")
print(f"Results for {len(rerun_df)} files saved to '{FINAL_OUTPUT_FILE}'")
