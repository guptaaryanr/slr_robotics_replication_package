# Energy Efficiency in Robotics Software: SLR (2020-2024) - Replication Package

This repository contains the **minimal, frozen artifacts** and **scripts** to reproduce the analysis, figures, and tables for our systematic literature review (SLR) “Energy Efficiency in Robotics Software: A Systematic Literature Review From 2020 Onward.”

- **Paper:** [add arXiv link after upload]
- **Release tag:** v1.0.0
- **Contact:** Aryan Gupta, guptaaryanr@gmail.com
- **ORCID:** 0009-0001-8179-5145
- **License:** Code MIT; data CC BY 4.0 (see `LICENSE`)

## What’s inside (short)
- `data/` - frozen CSV/JSONL used in the paper (search, screening, snowballing, extraction, final corpus).
- `code/` - small scripts to regenerate results or re-run parts of the pipeline.
- `results/` - final figures and tables exactly as used in the paper.
- `config/` - inclusion/exclusion rules and the extraction schema (`scheme.json`).
- `env/` - `requirements.txt` (pinned) and `versions.txt` (tool & model versions).

> We **do not** distribute paper PDFs. Use the DOIs in `data/corpus/final_79_metadata.csv` to obtain texts.

## Quickstart (no API keys needed)
Recreate all paper figures & tables from the frozen data:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r env/requirements.txt

# Rebuild figures/tables from frozen tidy data, change directory paths as needed
python code/explore_raw.py && python code/plot_frequency.py
