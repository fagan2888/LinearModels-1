import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf


data = pd.DataFrame()


def set_data(df):
    """
    Sets global data variable to be used in all functions

    Parameters
    ----------
    df: Complete airline dataframe to be used
    """

    global data
    data = df


def plot_seaborn_reg(origin):
    """
    Uses Seaborn to create a scatter plot of "AirTime" vs "Distance" columns in "df".
    Also fits a linear regression model in the same plot.

    Parameters
    ----------
    origin: A string.

    Returns
    -------
    A matplotlib Axes object
    """

    df = data.loc[data['Origin'] == origin].dropna()

    ax = sns.regplot('AirTime', 'Distance', data=df, scatter=True)  # Create regression line
    ax.set_title('Air time vs Distance regression')  # Set title
    ax.set_xlabel('Air time (minutes)')  # Set x label
    ax.set_ylabel('Distance (miles)')  # Set y label

    return ax


def get_statsmodels_reg(df_fit, X_predict):
    """
    Trains OLS on the columns in "df_fit" and makes a prediction for "X_predict".
    Returns the predicted `DepDelay` values.

    Parameters
    ----------
    df_fit: A pandas.DataFrame. Should have "AirTime", "ArrDelay", and "Distance" columns.
    X_predict: A pandas.DataFrame. Should have "AirTime" and "ArrDelay" columns.

    Returns
    -------
    A numpy array
    """
    mod = smf.ols(formula='Distance ~ AirTime + ArrDelay', data=df_fit)  # fit a regression
    res = mod.fit()  # Get the full fit
    y_predict = res.predict(X_predict)  # predict on new X

    return y_predict


def plot_statsmodels_reg(df_test, df_pred):
    """
    Plots the following:
    1. A scatter plot of the "AirTime" column of "df_test" on the x-axis
       and the "Distance" column of "df_test" on the y-axis,
    2. A straight line (multivariate linear regression model) of the
       "AirTime" column of "df_pred" on the x-axis and the "Distance"
       column of "df_pred" on the y-axis.

    Parameters
    ----------
    df_test: A pandas.DataFrame
    df_pred: A pandas.DataFrame

    Returns
    -------
    A matplotlib.Axes object
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)  # Create axes

    ax.set_title('Multivariate Regression')
    ax.set_xlabel('Air time (minutes)')
    ax.set_ylabel('Distance (miles)')  # Set titles

    ax.scatter(df_test['AirTime'], df_test['Distance'])
    ax.plot(df_pred['AirTime'], df_pred['Distance'], 'r-')

    ax.axis([-100, 600, -1000, 5000])
    print(len(ax.lines))
    print(len(ax.collections))

    return ax
