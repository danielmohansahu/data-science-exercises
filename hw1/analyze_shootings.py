#!/usr/bin/env python3
""" Extract, clean, parse, and visualize a Kaggle dataset on Police Shootings.

Date extracted from:
    https://www.kaggle.com/datasets/ramjasmaurya/us-police-shootings-from-20152022
"""

# STL
import glob
from collections import Counter

# Data Science Tools
import pandas

# Visualization
import tqdm
import seaborn as sns
from matplotlib import pyplot as plt

# global hardcoded constants
FILENAME = "data/shootings/shootings_2015_2022.csv"

# map from state name to geographic region
STATE_MAP = {
    "AK": "WEST", "HI": "WEST", "CA": "WEST", "WA": "WEST", "OR": "WEST", "NV": "WEST", "MT": "WEST", "ID": "WEST", "WY": "WEST", "UT": "WEST", "CO": "WEST",
"AZ": "SOUTHWEST", "NM": "SOUTHWEST", "OK": "SOUTHWEST", "TX": "SOUTHWEST",
"KS": "MIDWEST", "ND": "MIDWEST", "SD": "MIDWEST", "NE": "MIDWEST", "MN": "MIDWEST", "IA": "MIDWEST", "MO": "MIDWEST", "IL": "MIDWEST", "IN": "MIDWEST", "WI": "MIDWEST", "MI": "MIDWEST", "OH": "MIDWEST",
"AR": "SOUTHEAST", "LA": "SOUTHEAST", "MS": "SOUTHEAST", "KY": "SOUTHEAST", "TN": "SOUTHEAST", "AL": "SOUTHEAST", "GA": "SOUTHEAST", "FL": "SOUTHEAST", "SC": "SOUTHEAST", "NC": "SOUTHEAST", "VA": "SOUTHEAST", "WV": "SOUTHEAST", "MD": "SOUTHEAST", "DE": "SOUTHEAST", "DC": "SOUTHEAST",
"PA": "NORTHEAST", "NY": "NORTHEAST", "NJ": "NORTHEAST", "CT": "NORTHEAST", "RI": "NORTHEAST", "MA": "NORTHEAST", "NH": "NORTHEAST", "VT": "NORTHEAST", "ME": "NORTHEAST"
}

# map used to bin various weapon types (all others are 'other')
WEAPON_MAP = {
        "gun": "GUN", "guns and explosives": "GUN", "gun and machete": "GUN", "gun and knife": "GUN", "hatchet and gun": "GUN", "machete and gun": "GUN", "gun and sword": "GUN", "gun and car": "GUN", "gun and vehicle": "GUN", "vehicle and gun": "GUN",
    "unarmed": "NONE", "undetermined": "NONE"
}

# map used to convert "fleeing" categories into a boolean
FLEEING_MAP = {
    "Not fleeing": False,
    "Foot": True, "Car": True, "Other": True
}

def plot_year_heatmap(df, count_by_year, column):
    """ Visualize the change over time ('year') for the given column.
    """
    # figure 1: plotting the change over time for various groupings of data
    subframe = dataframe[["year", column]]
    subframe["pct"] = 0
    subframe = subframe.groupby(by=["year", column]).count().reset_index()
    subframe["pct"] = subframe.apply(lambda r: 100.0 * r["pct"] / count_by_year[r["year"]], axis=1)
    sns.heatmap(subframe.pivot(column, "year", "pct"), annot=True)

if __name__ == "__main__":
    ### data loading
    print(f"Loading {FILENAME}...")
    raw_dataframe = pandas.read_csv(FILENAME)

    ### data munging - bin raw data columns and handle NANs in raw data
    dataframe = pandas.DataFrame()

    # copy the following rows unmodified
    dataframe["date"] = raw_dataframe["date"]
    dataframe["age"] = raw_dataframe["age"]
    dataframe["gender"] = raw_dataframe["gender"]
    dataframe["race"] = raw_dataframe["race"].fillna("O")
    dataframe["threat_level"] = raw_dataframe["threat_level"]
    dataframe["signs_of_mental_illness"] = raw_dataframe["signs_of_mental_illness"]
    dataframe["body_camera"] = raw_dataframe["body_camera"]

    # bin the following rows into more manageable categories
    dataframe["year"] = dataframe["date"].apply(lambda r: int(r[:4]))
    dataframe["region"] = raw_dataframe["state"].apply(lambda r: STATE_MAP[r])
    dataframe["weapon"] = raw_dataframe["armed"].fillna("unarmed").apply(lambda r: WEAPON_MAP[r] if r in WEAPON_MAP else "OTHER")
    dataframe["armed"] = dataframe["weapon"].apply(lambda r: r != "NONE")
    dataframe["escaping"] = raw_dataframe["flee"].fillna("Not fleeing").apply(lambda r: FLEEING_MAP[r])

    # drop all unhandled NaNs
    dataframe = dataframe.dropna()

    # collect mapping from YEAR -> COUNT used throughout visualizations
    count_by_year = dataframe.groupby(by=["year"]).count().reset_index().to_dict('list')
    count_by_year = { k:v for k,v in zip(count_by_year["year"], count_by_year["date"]) }

    ### Visualizations

    # figure 0: pairplot, for general understanding
    sns.pairplot(data=dataframe, hue="race")

    # figure 1: plotting the change over time for various groupings of data
    plt.figure("Count by Race over Time")
    plot_year_heatmap(dataframe, count_by_year, "race")

    # figure 2: plotting the change over time based on geographic region
    plt.figure("Count by Region over Time")
    plot_year_heatmap(dataframe, count_by_year, "region")

    # figure 3: plotting the change over time based on weapon
    plt.figure("Count by Weapon over Time")
    plot_year_heatmap(dataframe, count_by_year, "weapon")

    # figure 4: plot age distribution vs. armed state
    g = sns.catplot(data=dataframe, x="weapon", y="age", kind="violin", inner=None)
    sns.swarmplot(data=dataframe, x="weapon", y="age", size=1, color='k', ax=g.ax)

    plt.show()

    # import code
    # code.interact(local=locals())
