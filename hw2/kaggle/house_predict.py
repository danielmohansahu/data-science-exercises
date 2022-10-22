#!/usr/bin/env python3
""" House price prediction via Regression.

Submission for Kaggle Competition
  https://www.kaggle.com/competitions/enpm808w-2022-hw2/overview
"""

# STL

# Data Science
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

# Visualization
from matplotlib import pyplot as plt

# Geographic Information
import pgeocode

# hardcoded info
TRAINFILE="house_train.csv"
TESTFILE="house_test.csv"

def calc_weighted_dist(df, dist):
    """ Return the cost weighted by (log of) distance from each zip code.
    """
    # first get all distances
    distances = []
    all_zips = np.array(df["zip"].astype(str))
    all_costs = np.array(df["price2007"])

    # iteratively find distances (O(n^2)!)
    for zip_ in tqdm(all_zips):
        distances.append(dist.query_postal_code([zip_] * len(all_zips), all_zips))

    # convert to numpy arrays
    distances = np.array(distances)
    
    # normalize distances to maximum
    distances = distances / np.nanmax(distances)

    # calculate the log to weight closer distances much higher
    #  note that we replace inf with 0.0 to ignore local zip code
    distances = -np.log10(distances)
    distances[distances == np.inf] = 0.0

    # calculate the cost weighted by distance for each value
    weights = []
    for vals in distances:
        weights.append(np.nansum(vals * all_costs))
    return weights

if __name__ == "__main__":
    # load data
    df_train = pd.read_csv(TRAINFILE)
    df_test = pd.read_csv(TESTFILE)

    # feature construction - adding a weighted distance metric
    #  this feature is an attempt to account for the relative
    #  price of cities that are geographically near the current
    #  property
    dist = pgeocode.GeoDistance("US")
    print("Feature Construction (Train)...")
    df_train["cost_weighted_dist"] = calc_weighted_dist(df_train, dist)
    print("Feature Construction (Test)...")
    df_test["cost_weighted_dist"] = calc_weighted_dist(df_test, dist)

    # data munging - drop NA / unimportant keys
    df_train = df_train.dropna().drop(["id", "zip", "state", "county"], axis=1)

    # convert Y values to log form, to prevent overweighting of expensive homes
    Y_train = df_train["price2013"]
    X_train = df_train[["cost_weighted_dist", "poverty", "price2007"]]

    # fit the model
    lr = LinearRegression()
    lr.fit(X_train, Y_train)

    # plot results (from training set)
    Y_train_pred = lr.predict(X_train)

    plt.scatter(Y_train_pred, Y_train, c="blue", marker="s", label="Training data")
    plt.title("Linear regression")
    plt.xlabel("Predicted Prices ($)")
    plt.ylabel("Actual Prices ($)")
    plt.show()
    
    # format output prediction for test data
    Y_test_pred = lr.predict(df_test[["cost_weighted_dist", "poverty", "price2007"]])
    
    import code
    code.interact(local=locals())
