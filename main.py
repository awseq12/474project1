# This is a sample Python script.
import numpy.random

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import data
import numpy as n
import pandas as p
import plotly.express as px
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import logitRegression as l

"extracts data from csv"
def data_processing(filename):

    dataframe = p.read_csv(filename)
    #print(dataframe)

    """Main STATISTICS"""
    print(dataframe.head())
    print(dataframe.size)
    print(dataframe.var)
    print(dataframe.cumsum)
    print(dataframe.cumprod)
    print(dataframe.describe())
    """5 VISUALIZATIONS MATLIB"""

    # sum of missing entries in dataset
    missing = dataframe.isnull().sum(axis=0)
    # dropna if applicable
    if not missing.empty:
        dataframe = dataframe.dropna()

    return (dataframe)


def linear_regression(filename):
    # data preprocessing from above
    d = data_processing(filename)
    #data split
    """5 VISUALIZATIONS MATLIB"""

    fig = px.scatter(d, x='fixed acidity', y='volatile acidity', trendline="ols", trendline_color_override="red", title="volatile acidity based on fixed cidity")
    fig.show()
    fig = px.scatter(d, x='citric acid', y='volatile acidity', trendline="ols", trendline_color_override="red",title="volatile acidity based on citric acid")
    fig.show()
    fig = px.scatter(d, x='free sulfur dioxide', y='volatile acidity', trendline="ols", trendline_color_override="red",title="volatile acidity based on free sulfur dioxide")
    fig.show()
    fig = px.scatter(d, x='total sulfur dioxide', y='volatile acidity', trendline="ols", trendline_color_override="red",title="volatile acidity based on total sulfur dioxide")
    fig.show()
    fig = px.scatter(d, x='sulphates', y='volatile acidity', trendline="ols", trendline_color_override="red", title="volatile acidity based on sulphates")
    fig.show()


    """
    CHOOSE WHICH FEATURES TO BE INCLUDED AS INPUT FEATURES, AND WHICH ONE
    SERVES AS THE OUTPUT SCALE
    # input:
    # output: volatile acidity
    """
    """ divide dataset into 80% training vs 20% testing """
    train = d.sample(frac=0.8, random_state=200)
    test = d.drop(train.index)

    """ Split into X(inputs) and Y(outputs)"""
    y_train = train['volatile acidity']
    y_test = test['volatile acidity']
    x_train = train.drop(columns='volatile acidity')
    x_test = test.drop(columns='volatile acidity')


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
    w = x_train.transpose()
    w = w.dot(x_train)
    w = n.linalg.inv(w)
    w = w.dot(x_train.transpose())
    w = w.dot(y_train)

    #MSE = ((prediction - test['actual']) ** 2).sum()

    """ Y predict"""
    wt = w.transpose()
    y = []
    for index, row in x_test.iterrows():
        calc = wt.dot(row)
        y.append(calc)
    """CALCULATE MSE"""
    sum = 0
    for i in range(0,len(y)):
        sum += (y_test.iloc[i] - y[i])**2

    #MSE = (1/len(y_test.index)) * sum(y_test.index.map(lambda i: (y_test.loc[i] - (w.transpose().dot(x_test.loc[i])))**2))
    MSE = sum/len(y_test.index)
    print("MSE IS EQUAL TO")
    print(MSE)
    print("Weight is the following:")
    print(w)

    # predicted dataframe

    """plot result here"""
    #actual
    xdata = range(0, len(x_test))
    ydata= y_test
    plt.scatter(xdata, ydata)
    plt.title("actual vs calculated")
    ydata = y
    plt.scatter(xdata, ydata)
    plt.ylabel = ""
    plt.xlabel = ""
    plt.legend(['actual', ' calculated'])
    plt.show()
    return MSE


def logistic_regression(filename):
    d = data_processing(filename)
    """normailize + change into categorical feature"""
    for key, value in d.items():
        if is_numeric_dtype(d[key])==False:
            d[key] = d[key].astype('category')
            d[key] = p.factorize(d[key])[0]

            #X_normalized = -1 + (X - self.feature_min) * (2 / (self.feature_max - self.feature_min))
            #d[key] = -1 +(d[key] - d[key].min()) *2/ (d[key].max() - d[key].min())

       # if d.dtypes[columnName]

    d = (d - d.min()) / (d.max() - d.min())
    #print(d)
    """5 VISUALIZATIONS MATLIB"""

    fig = px.scatter(d, x='bill_length_mm', y='flipper_length_mm', trendline="ols", trendline_color_override="red",
                     title="bill_length_mm' vs flipper_length_mm")
    fig.show()
    fig = px.scatter(d, x='flipper_length_mm', y='body_mass_g', trendline="ols", trendline_color_override="red",
                     title="flipper_length_mm vs body_mass_g")
    fig.show()
    fig = px.scatter(d, x='bill_depth_mm', y='body_mass_g', trendline="ols", trendline_color_override="red",
                     title="bill_length_mm vs  body_mass_g ")
    fig.show()
    fig = px.scatter(d, x='flipper_length_mm', y='bill_depth_mm', trendline="ols", trendline_color_override="red",
                     title="flipper_length_mm vs bill_depth_mm")
    fig.show()
    fig = px.scatter(d, x='body_mass_g', y='bill_depth_mm', trendline="ols", trendline_color_override="red",
                     title="body_mass_g vs bill_depth_mm")
    fig.show()

    """target Y: Gender"""
    """divide into training and test"""
    train = d.sample(frac=0.8, random_state=200)
    test = d.drop(train.index)

    """ Split into X(inputs) and Y(outputs)"""
    y_train = train['sex']
    y_test = test['sex']
    x_train = train.drop(columns='sex')
    x_test = test.drop(columns='sex')
    print("y_train")
    print(y_train)

    print("y_test")
    print(y_test)

    print("x_train")
    print(x_train)

    print("x_test")
    print(x_test)

    learning_rate = 0.01
    iterations = 1000
    # .01 1000 65%-73%
    # 1 10000 64 -65
    # 1e-4 100000 .73
    model = l.LogitRegression(learning_rate, iterations)
    model.fit(x_train, y_train)
    predict = model.predict(x_test, y_test)
    print ('loss values over time',list(reversed(model.cost)))
    plt.title("loss function: iterations vs loss")
    plt.plot(list(reversed(model.cost)))
    print("logitRegression accuracy:",predict)
    print("final weight", model.theta)
    plt.show()
    return predict




# Press the green button in the gutter to run the script.
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

"""comment here"""
#linear_regression("data\winequality-red.csv")
logistic_regression(("data\penguins.csv"))