
"""PLOT.py is a utility class for plotting dataframes with plotly

Author :
    Alexandre Bremard
Contributors :
    -
Version Control :
    1.0 - 07/06/2020 : plot_data, amount_of_variation, test
"""

# ----------------------------------------------------------------- External Imports 
import plotly.graph_objects as go
import os
import pandas as pd

# ----------------------------------------------------------------- Body 
def plot_data(data, columns, traceMode='lines'):
    """This function plots your featured data into 2D graphs

    Args:
        data (pd.dataframe): data that has already been processed, i.e. contains feature columns
        columns (str[]): columns to plot
        traceMode (str, optional): any combination of "lines", "markers", "text" joined with a "+" OR "none". Defaults to 'lines'.

    Returns:
        fig (go.Figure): plotted figure
    """
    fig = go.Figure()
    for col in columns:
        fig.add_trace(go.Scatter(x=data.index, y=data[col], mode=traceMode, name=col))
        
    return fig

def amount_of_variation(data, timeframe, exportHTML = False, exportCSV = False):
    """This function ranks the amount of variation metric for each feature.
        The plot is sorted by descending score. 
        The rule of thumb is to get rid of the tail.
        Additional options for HTML or CSV exports included.

    Args:
        data (pd.dataframe): data containing the features to be studied 
        timeframe (str): timeframe of study
        exportHTML (bool, optional): Exports plotly graph as HTML. Defaults to False.
        exportCSV (bool, optional): Export ranking dataframe as CSV. Defaults to False.

    Returns:
        fig (go.Figure): plotted figure
    """
    fig = go.Figure()
    std = data.std()
    std = std.sort_values(ascending = False)
    fig.add_trace(go.Scatter(x=std.index, y=std.values, mode='markers'))

    if exportCSV:
        baseURL = '../Data/Input/Time/' + timeframe
        directory = baseURL + "/features-selection"
        if not os.path.exists(directory):
            os.mkdir(directory)
            print("Directory " , directory ,  " Created ")
        else:    
            print("Directory " , directory ,  " already exists")
        url = directory + "/amount-of-variation.html"
        std = pd.DataFrame({'Feature':std.index, 'Score':std.values})
        std.to_csv(directory + "/amount-of-variation.csv", columns=["Feature", "Score"])

    if exportHTML:
        fig.write_html(url, auto_open=False)

    return fig


# ----------------------------------------------------------------- Test
def test():
    """This function is internal to FEATURES.py, it is meant for debugging but also serves as unit test
    """
    print("----- TEST FOR PLOT.PY -----")
    timeframe = '5min'
    timeURL = '../Data/Input/Time/' + timeframe
    data = pd.read_csv(timeURL + "/xy-array.csv")
    columns = ["SMA-20-1", "SMA-200-1"]

    # Below is the test for plot_data function
    print("----- PLOT DATA ------")
    plot = plot_data(data, columns, 'markers')
    plot.show()

    print("----- AMOUNT OF VARIATION ------")
    # Below is the test for amount_of_variation
    plot = amount_of_variation(data, timeframe)
    plot.show()
    print("---------------------------------")
