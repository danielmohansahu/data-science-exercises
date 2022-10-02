#!/usr/bin/env python3
""" Extract, clean, parse, and visualize a Kaggle dataset on Police Shootings.

Date extracted from:
    https://www.kaggle.com/datasets/ramjasmaurya/us-police-shootings-from-20152022
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
    dataframe["race"] = raw_dataframe["race"]
    dataframe["threat_level"] = raw_dataframe["threat_level"]
    dataframe["signs_of_mental_illness"] = raw_dataframe["signs_of_mental_illness"]
    dataframe["body_camera"] = raw_dataframe["body_camera"]

    # bin the following rows into more manageable categories
    dataframe["region"] = raw_dataframe["state"].apply(lambda r: STATE_MAP[r])
    dataframe["weapon"] = raw_dataframe["armed"].fillna("unarmed").apply(lambda r: WEAPON_MAP[r] if r in WEAPON_MAP else "OTHER")
    dataframe["escaping"] = raw_dataframe["flee"].fillna("Not fleeing").apply(lambda r: FLEEING_MAP[r])

    ### Miscellaneous Visualization
    sns.pairplot(data=dataframe, hue="gender")

    plt.show()

    print(f"Columns: {dataframe.columns}")
    import code
    code.interact(local=locals())
