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
DATAPATH = "data/nyt/"
LABELS = set(("Age", "Gender", "Impressions", "Clicks", "Signed_In"))
GROUPS = ["<18","18-24","25-34","35-44","45-54","55-64","65+","???"]

def bin_by_age(df):
    """ Plot the distributions of number impressions and click-through-rate

    Create a new variable, age_group, that categorizes users as “<18”, ”18-24”, ”25-34”, ”35-44”, ”45-54”, “55-64” and “65+”.
    """
    # convenience binning function
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

def visualize_by_age(df):
    """ Plot the distributions of number impressions and click-through-rate

    (i) Plot (CTR=# clicks/# impressions), for these 6 age categories.
    (ii) Define a new variable to segment or categorize users based on their click behavior.
    """
    # plot desired information as subplots
    fig,axs = plt.subplots(2,1)
    fig.suptitle("Impressions and CTR by Age")
    sns.barplot(data=df, ax=axs[0], x="Age_Group", y="CTR", hue="Gender", order=GROUPS)
    sns.barplot(data=df, ax=axs[1], x="Age_Group", y="Impressions", hue="Gender", order=GROUPS)

    # update display options
    axs[0].get_xaxis().set_visible(False)
    for ax in axs:
        for text in ax.get_legend().texts:
            if text.get_text() == "-1":
                text.set_text("Unknown")
            elif text.get_text() == "0":
                text.set_text("Female")
            else:
                text.set_text("Male")

def collect_metrics(day, df, results):
    """ Collect our suite of metrics for a single day.

    Each new set of metrics (per demographic) is added to
    the results dataframe.
    """
    # first time initialization
    metrics = ("Day", "Age_Group", "Count", "CTR_Mean", "CTR_Stddev", "Clicks_Mean", "Clicks_Stddev", "Impressions_Mean", "Impressions_Stddev")
    if results is None:
        results = pandas.DataFrame(columns=metrics)

    # add metrics for each demographic
    for group in GROUPS:
        # filter based on age group
        subset = df[df["Age_Group"] == group]
        # create a new frame with our metric data
        frame = pandas.DataFrame([{
            "Day": day,
            "Age_Group": group,
            "Count": subset.shape[0],
            "CTR_Mean": subset["CTR"].mean(),
            "CTR_Stddev": subset["CTR"].std(),
            "Clicks_Mean": subset["Clicks"].mean(),
            "Clicks_Stddev": subset["Clicks"].std(),
            "Impressions_Mean": subset["Impressions"].mean(),
            "Impressions_Stddev": subset["Impressions"].std()}])
        # concatenate with other results
        results = pandas.concat([results, frame])

    # return collected results
    return results

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
    if len(data) == 0:
        sys.exit(0)

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
        # introduce a third category of gender (unknown) based on Age heuristic
        new = new[new["Gender"].between(0,1,inclusive="both")]
        new["Gender"] = new.apply(lambda r: -1 if r["Age"] == 0 else r["Gender"], axis=1)
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

    ### Part A / B: Visualize a single day split into age groups.
    print("Binning data by age group...")
    for frame in tqdm.tqdm(data):
        bin_by_age(frame)

    # only visualize one (hopefully representative!) datum
    visualize_by_age(data[1])

    ### Part C: Collect Metrics
    print("Collecting metrics...")
    metrics = None
    for i,frame in tqdm.tqdm(enumerate(data)):
        metrics = collect_metrics(i+1, frame, metrics)

    # display metrics
    sns.pairplot(data=metrics, hue="Age_Group", hue_order=GROUPS)

    ### Miscellaneous Visualization
    # sns.pairplot(data=data[11], hue="Gender")
    # sns.pairplot(data=data[11], hue="Age_Group", hue_order=GROUPS)
    plt.show()
