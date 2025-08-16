#!/usr/bin/env python3
# consensus_reduce.py
#
# Combine three screening runs into a majority-vote result and flag
# inconsistent papers for manual review.

import pandas as pd
from collections import Counter

RUN_FILES = [
    "consensus_snow_run1.csv",
    "consensus_snow_run2.csv",
    "consensus_snow_run3.csv",
]

# ----------------------------------------------------------------------
dfs = [
    pd.read_csv(f).rename(columns={"decision": f"dec_{i}"})
    for i, f in enumerate(RUN_FILES, 1)
]

merged = dfs[0]
for df in dfs[1:]:
    merged = merged.merge(
        df[["file", "dec_{}".format(len(merged.columns) - 4)]], on="file"
    )


def majority(row):
    votes = [row[c] for c in row.index if c.startswith("dec_")]
    count = Counter(votes)
    winner, freq = count.most_common(1)[0]
    return winner if freq == len(votes) else None  # None = disagreement


merged["consensus"] = merged.apply(majority, axis=1)

# ----------------------------------------------------------------------
stable = merged.dropna(subset=["consensus"])
unstable = merged[merged["consensus"].isna()]

stable[["file", "consensus"]].to_csv("consensus_snow_results.csv", index=False)
unstable["file"].to_csv("manual_snow_queue.csv", index=False)

print("Stable (unanimous) decisions :", len(stable))
print("Need manual adjudication     :", len(unstable))
print("Files listed in manual_queue.csv")
