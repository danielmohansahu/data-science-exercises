#!/usr/bin/env python3

""" Construct a linear regression model for provided Housing Data (2007, 2013)

Resources:
    https://www.kaggle.com/code/juliencs/a-study-on-regression-applied-to-the-ames-dataset
"""

# STL
import os
import sys

# Data Science
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Visualization
from matplotlib import pyplot as plt
import seaborn as sns

# Hardcoded Information
TRAINFILE = "data/house_train.csv"
TESTFILE = "data/house_test.csv"
VERBOSE = True

def fit_linear_model(X, Y):
    """ Construct a Linear Regression model to the given dataframe.
    """
    # split into test / train sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .20)

    # convert Y values to log form, to prevent overweighting of expensive homes
    Y_train = np.log(Y_train)
    Y_test = np.log(Y_test)

    # fit the model
    lr = LinearRegression()
    lr.fit(X_train, Y_train)

    # visualize
    if VERBOSE:
        # predict on our training set
        Y_train_pred = lr.predict(X_train)
        Y_test_pred = lr.predict(X_test)

        # Plot residuals
        plt.scatter(Y_train_pred, Y_train_pred - Y_train, c="blue", marker="s", label="Training data")
        plt.scatter(Y_test_pred, Y_test_pred - Y_test, c="blue", marker="s", label="Test data")
        plt.title("State Linear Regression Residuals")
        plt.xlabel("Predicted values")
        plt.ylabel("Residuals")
        plt.show()
        
        # Plot predictions
        plt.scatter(Y_train_pred, Y_train, c="blue", marker="s", label="Training data")
        plt.scatter(Y_test_pred, Y_test, c="blue", marker="s", label="Testing data")
        plt.title("Linear regression")
        plt.xlabel("Predicted Prices (log)")
        plt.ylabel("Actual Prices (log)")
        plt.show()

    # return the model for evaluation / visualization
    return lr

if __name__ == "__main__":
    # load data
    df_train = pd.read_csv(TRAINFILE)
    df_test = pd.read_csv(TESTFILE)

    # data munging
    df_train = df_train.dropna()
    df_test = df_test.dropna()

    # visualization
    if VERBOSE:
        print(df_train.describe())
        plt.plot(np.log(df_train["price2007"]), np.log(df_train["price2013"]), '*')
        plt.xlabel("2007 Price (log($))")
        plt.ylabel("2013 Price (log($))")
        plt.grid()
        plt.show()

    # build a simple linear regression model for price based on state
    X_state = pd.get_dummies(data=df_train[["state"]], drop_first=True)
    Y_state = df_train["price2013"]
    lr_state = fit_linear_model(X_state, Y_state)

    # determine which state corresponds to what coefficient
    coeffs = [(coef, state.replace("state_","")) for state,coef in zip(X_state.columns, lr_state.coef_)]
    print("State-based regression:")
    print(f"\tIntercept: {lr_state.intercept_}")
    print(f"\tPriciest state: {max(coeffs)[-1]}")
    print(f"\tCheapest state: {min(coeffs)[-1]}")

    # print model summary
    import code
    code.interact(local=locals())

