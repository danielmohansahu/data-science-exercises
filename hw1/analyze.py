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
import seaborn as sns
from matplotlib import pyplot as plt

# global hardcoded constants
DATAPATH = "data/"
LABELS = set(("Age", "Gender", "Impressions", "Clicks", "Signed_In"))

def visualize_ctr_by_age(df):
    """ Plot the distributions of number impressions and click-through-rate

    (a) Create a new variable, age_group, that categorizes users as “<18”, ”18-24”, ”25-34”, ”35-44”, ”45-54”, “55-64” and “65+”.
    (b)
        (i) Plot (CTR=# clicks/# impressions), for these 6 age categories.
        (ii) Define a new variable to segment or categorize users based on their click behavior.
    """
    # convenience binning function
    labels = ["<18","18-24","25-34","35-44","45-54","55-64","65+","???"]
    def bin_by_age(age):
        if age == 0:
            return "???"
        elif age < 18:
            return "<18"
        elif age < 25:
            return "18-24"
        elif age < 35:
            return "25-34"
        elif age < 45:
            return "35-44"
        elif age < 55:
            return "45-54"
        elif age < 65:
            return "55-64"
        else:
            return "65+"

    # extract information as separate column features
    df["Age_Group"] = df.apply(lambda r: bin_by_age(r["Age"]), axis=1)
    df["CTR"] = df.apply(lambda r: 0 if r["Impressions"] == 0 else r["Clicks"] / r["Impressions"], axis=1)

    # plot desired information as subplots
    fig,axs = plt.subplots(2,1)
    fig.suptitle("Impressions and CTR by Age")
    sns.barplot(data=df, ax=axs[0], x="Age_Group", y="CTR", hue="Gender", order=labels)
    sns.barplot(data=df, ax=axs[1], x="Age_Group", y="Impressions", hue="Gender", order=labels)

    # update display options
    axs[0].get_xaxis().set_visible(False)
    for ax in axs:
        for text in ax.get_legend().texts:
            text.set_text("Female" if text.get_text() == "0" else "Male")
    plt.show()

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
    new_data = []
    for frame in tqdm.tqdm(data):
        # drop all NANs
        new = frame.dropna()
        # drop invalid ages (we allow 0 - it's equivalent to not signed in)
        new = new[new["Age"] >= 0]
        # drop unsupported genders and cast as a string
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
        new_data.append(new)
    data = new_data
    print(f"Removed {dropped} rows ({dropped / total * 100.0}%)")

    ### Part A:
    visualize_ctr_by_age(data[11])

