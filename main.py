# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import data
import numpy as n
import pandas as p
import plotly as px
import matplotlib as plt

"extracts data from csv"
def data_processing(filename):

    dataframe = p.read_csv(filename)
    print(dataframe)

    """Main STATISTICS"""
    print(dataframe.head())
    print(dataframe.size)
    print(dataframe.var)
    print(dataframe.cumsum)
    print(dataframe.cumprod)
    print(dataframe.describe())
    """5 VISUALIZATIONS PLOTLY"""




    # sum of missing entries in dataset
    missing = dataframe.isnull().sum(axis=0)
    print(dataframe.isnull().sum(axis=0))
    # dropna if applicable
    if not missing.empty:
        dataframe = dataframe.dropna()
    #print(dataframe.isnull().sum(axis=0))
    # convert to Categorical if applicable

    return (dataframe)


def linear_regression(filename):
    # data preprocessing from above
    d = data_processing(filename)
    """5 VISUALIZATIONS PLOTLY"""


    #data split
    """
    CHOOSE WHICH FEATURES TO BE INCLUDED AS INPUT FEATURES, AND WHICH ONE
    SERVES AS THE OUTPUT SCALE
    # input:
    # output: quality
    """
    """ divide dataset into 80% training vs 20% testing """
    train = d.sample(frac=0.8, random_state=200)
    test = d.drop(train.index)

    """ Split into X(inputs) and Y(outputs)"""
    y_train = train['quality'].reset_index()
    y_test = test['quality'].reset_index()
    x_train = train.drop(columns='quality').reset_index()
    x_test = test.drop(columns='quality').reset_index()


    """PRINT THE SHAPE OF YOUR X_TRAIN Y_TRAIN X_TEST Y_TEST"""
    print("y_train")
    print(y_train)

    print("y_test")
    print(y_test)

    print("x_train")
    print(x_train)

    print("x_test")
    print(x_test)

    # solve linear regression problem

    """ SOLVE THE LINEAR REGRESSION PROBLEM FOR WEIGHTS"""

    w = n.linalg.inv(x_train.transpose().dot(x_train)).dot(x_train.transpose()).dot(y_train)

    #MSE = ((prediction - test['actual']) ** 2).sum()
    """CALCULATE MSE"""
    MSE = (1/len(y_test.index)) * sum(y_test.index.map(lambda i: (y_test.loc[i] - (w.transpose().dot(x_test.loc[i])))**2))
    print("MSE IS EQUAL TO")
    print(MSE)

    """plot result here"""
    fig = px.scatter(test, x='', y='quality', trendline="ols")
    fig.show


    return MSE


def logistic_regression(filename):
    d = data_processing(filename)
    """normailize no-categorical feature"""
    #normalize =(d-d.min())/(d.max()-d.min())
    """target Y: Gender"""


    return




# Press the green button in the gutter to run the script.
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

#linear_regression("data\winequality-red.csv")
logistic_regression(("data\penguins.csv"))