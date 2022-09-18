#!/usr/bin/env python3
""" Extract, clean, parse, and analyze a simulated New York Times page view dataset.

Date extracted from:
    http://stat.columbia.edu/~rachel/datasets/
"""

# STL
import glob

# Data Science Tools
import pandas

# Visualization
import tqdm

# global hardcoded constants
DATAPATH = "data/"
LABELS = set(("Age", "Gender", "Impressions", "Clicks", "Signed_In"))

if __name__ == "__main__":
    ### data loading
    print(f"Loading CSVs from {DATAPATH}...")
    data = []
    for filename in tqdm.tqdm(glob.glob(DATAPATH + "*.csv")):
        try:
            day = pandas.read_csv(filename)
            assert (LABELS.issubset(set(day.columns))), f"missing one or more required labels ({LABELS})"
        except Exception as e:
            # @TODO should this be an error?
            # @TODO this changes our indexing count; ramifications?
            print(f"  failed to load {filename}:\n\t{e}")
        else:
            data.append(day)
    print(f"Loaded {len(data)} files.")

    ### data cleaning
    print("Pruning data...")
    dropped = 0
    total = 0
    for frame in tqdm.tqdm(data):
        # drop all NANs
        new = frame.dropna()
        # drop invalid ages
        new = new[new["Age"] > 0]
        # drop unsupported genders
        new = new[new["Gender"].between(0,1,inclusive="both")]
        # drop invalid impressions
        new = new[new["Impressions"] >= 0]
        # drop invalid clicks
        new = new[new["Clicks"] >= 0]
        # drop invalid state
        new = new[new["Signed_In"].between(0,1,inclusive="both")]

        # cast as expected types
        new = new.astype({"Age": "int8", "Gender": "int8", "Impressions": "int32", "Clicks": "int32", "Signed_In": "bool"})

        # record number of drops and update in our list
        total += len(frame)
        dropped += (len(frame) - len(new))
        frame = new
    print(f"Removed {dropped} frames {dropped / total * 100.0}%")



