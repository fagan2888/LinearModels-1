import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

data = pd.DataFrame()
origin_data = pd.DataFrame()


def set_data(df):
    """
    Sets global data variable to be used in all functions

    Parameters
    ----------
    df: Complete airline dataframe to be used
    """

    global data
    data = df.dropna()


def change_origin(origin):
    global origin_data
    origin_data = data.loc[data['Origin'] == origin]


def get_original_data():
    return data


def set_hourly():
    global origin_data
    origin_data = origin_data.groupby(pd.cut(origin_data['CRSDepTime'], np.arange(0, 2400, 100))).mean().dropna()


def plot_seaborn_reg(x,y):
    """
    Uses Seaborn to create a scatter plot of "AirTime" vs "Distance" columns in "df".
    Also fits a linear regression model in the same plot.

    Parameters
    ----------
    x: a string
    y: a string

    Returns
    -------
    A matplotlib Axes object
    """

    ax = sns.regplot(x, y, data=origin_data, scatter=True)  # Create regression line
    ax.set_title(x + ' vs ' + y + ' regression')  # Set title
    ax.set_xlabel(x)  # Set x label
    ax.set_ylabel(y)  # Set y label

    return ax


def get_train_test(train_columns, test_columns, test_size, random_state):
    """
    Uses sklearn.train_test_split to split "df" into a testing set and a test set.
    The "train_columns" parameter lists the attributes that will be used for training.
    The "test_columns" lists the column that we are trying to predict.
    The "test_size" should be between 0.0 and 1.0 and represents the proportion of the
    dataset to include in the test split.
    The "random_state" parameter is used in sklearn.train_test.split.

    Parameters
    ----------
    train_columns: A list of strings
    test_columns: A list of strings
    test_size: A float
    random_state: A numpy.random.RandomState instance

    Returns
    -------
    A 4-tuple of pandas.DataFrames
    """

    # train_test_split should automatically return a 4-tuple
    return train_test_split(origin_data[train_columns], origin_data[test_columns], test_size=test_size, random_state=random_state)


def get_statsmodels_reg(X_train, X_test, y_train):
    """
    Trains OLS on the columns in "df_fit" and makes a prediction for "X_predict".
    Returns the predicted `DepDelay` values.

    Parameters
    ----------
    X_train: A pandas.DataFrame. Should have "AirTime" and "ArrDelay" columns.
    X_test: A pandas.DataFrame. Should have "AirTime" and "ArrDelay" columns.
    y_train: A pandas.DataFrame. Should have "Distance" column.

    Returns
    -------
    x_predict: A pandas.DataFrame
    y_predict: A numpy array
    """

    df_fit = pd.DataFrame({
        "AirTime": X_train["AirTime"].values,
        "ArrDelay": X_train["ArrDelay"].values,
        "Distance": y_train["Distance"].values
    })

    x_predict = pd.DataFrame({
        'AirTime': np.linspace(X_test['AirTime'].min(), X_test['AirTime'].max(), 10),
        'ArrDelay': np.linspace(X_test['ArrDelay'].min(), X_test['ArrDelay'].max(), 10)
    })

    mod = smf.ols(formula='Distance ~ AirTime + ArrDelay', data=df_fit)  # fit a regression and fit
    res = mod.fit()
    y_predict = res.predict(x_predict)  # predict on new X

    return x_predict, y_predict


def plot_statsmodels_reg(X_test, X_pred, y_test, y_pred):
    """
    Plots the following:
    1. A scatter plot of the "AirTime" column of "df_test" on the x-axis
       and the "Distance" column of "df_test" on the y-axis,
    2. A straight line (multivariate linear regression model) of the
       "AirTime" column of "df_pred" on the x-axis and the "Distance"
       column of "df_pred" on the y-axis.

    Parameters
    ----------
    X_test: A pandas.DataFrame
    X_pred: A pandas.DataFrame
    y_test: A pandas.DataFrame
    y_pred: A numpy array

    Returns
    -------
    A matplotlib.Axes object
    """

    df_test = pd.DataFrame({
        'AirTime': X_test['AirTime'].values,
        'Distance': y_test['Distance'].values
    })

    df_pred = pd.DataFrame({
        'AirTime': X_pred['AirTime'].values,
        'Distance': y_pred
    })

    fig = plt.figure()
    ax = fig.add_subplot(111)  # Create axes

    ax.set_title('Multivariate Regression')
    ax.set_xlabel('Air time (minutes)')
    ax.set_ylabel('Distance (miles)')  # Set titles

    ax.scatter(df_test['AirTime'], df_test['Distance'], marker='.')
    ax.plot(df_pred['AirTime'], df_pred['Distance'], 'r-')

    ax.axis([-100, 600, -1000, 5000])

    return ax


def fit_reg_poly(degree, alpha=1.0):
    """
    Fits a ridge regression model on "CRSDepTime" to predict "DepDelay".

    Parameters
    ----------
    degree: parameter for PolynomialFeatures
    alpha: A float denoting whether to use lasso or ridge regression

    Returns
    -------
    A tuple of (sklearn.Pipeline object, numpy.ndarray)
    """

    CRSDepTime = origin_data['CRSDepTime'].as_matrix().reshape(18,1)
    DepDelay = origin_data['DepDelay'].as_matrix().reshape(18,1)

    model = make_pipeline(PolynomialFeatures(degree=degree), Ridge(alpha=alpha))
    model.fit(CRSDepTime,DepDelay)
    prediction = model.predict(CRSDepTime)

    return model, prediction


def find_mse(degree):
    """
    Computes mean squared error of a Ridge Regression model on "df".
    Uses the "fit_reg_poly()" function.

    Parameters
    ----------
    degree: A polynomial degree for PolynomialFeatures

    Returns
    -------
    A float.
    """

    model, prediction = fit_reg_poly(degree)

    result = mean_squared_error(origin_data['DepDelay'], prediction)

    return result


def plot_reg_poly(model, prediction):
    '''
    Plots the following:
    1. A scatter plot of the "CRSDepTime" column of "df" on the x-axis
       and the "DepDelay" column of "df" on the y-axis,
    2. A line that represents a polynomial of degree "degree".

    Parameters
    ----------
    model: a pipeline from fit_reg_poly
    prediction: a numpy ndarray

    Returns
    -------
    A matplotlib.Axes object
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111) #Create axes

    ax.set_title('Ridge Regression')
    ax.set_xlabel('Depature Time')
    ax.set_ylabel('Departure Delay') #Set titles

    ax.scatter(origin_data['CRSDepTime'], origin_data['DepDelay'])
    ax.plot(origin_data['CRSDepTime'], prediction, 'r-')

    return ax


def convert_to_binary(column, cutoff):
    """
    Adds a new column in Pandas.DataFrame "df".

    The returned DataFrame as one more column than "df".
    The name of this column is in the form "column_binary".
    For example, if "column" is "DepDelay", the name of the extra column
    in the returned DataFrame is "DepDelay_binary".

    We assume that df[column] contains only ints or floats.
    If df[column] < cutoff, df[column_binary] is 0.
    If df[column] >= cutoff, df[column_binary] is 1.

    Parameters
    ----------

    column: A string.
    cutoff: An int.

    """
    global origin_data
    origin_data[column+'_binary'] = (origin_data[column]>=cutoff).astype(int)

    return


def add_dummy(add_columns, keep_columns):
    """
    Transforms categorical variables of add_columns into binary indicator variables.

    Parameters
    ----------
    df: A pandas.DataFrame
    add_columns: A list of strings
    keep_columns: A list of strings

    Returns
    -------
    A pandas.DataFrame
    """

    dummies = pd.concat([pd.get_dummies(origin_data[col],prefix=col) for col in add_columns], axis=1)

    global origin_data
    origin_data = origin_data[keep_columns].join(dummies)

    return


def add_intercept():
    '''
    Appends to "df" an "Intercept" column whose values are all 1.0.

    Parameters
    ----------
    df: A pandas.DataFrame

    Returns
    -------
    A pandas.DataFrame
    '''

    Intercept = pd.Series(len(origin_data))
    Intercept = float(1)
    global origin_data
    origin_data['Intercept']=Intercept

    return


def fit_logitistic(train_columns, test_column):
    '''
    Fits a logistic regression model on "train_columns" to predict "test_column".

    The function returns a tuple of (model ,result).
    "model" is an instance of Logit(). "result" is the result of Logit.fit() method.

    Parameters
    ----------
    train_columns: A list of strings
    test_column: A string

    Returns
    -------
    A tuple of (model, result)
    model: An object of type statsmodels.discrete.discrete_model.Logit
    result: An object of type statsmodels.discrete.discrete_model.BinaryResultsWrapper
    '''
    model = sm.Logit(origin_data[test_column], origin_data[train_columns])
    result = model.fit()

    return model, result

