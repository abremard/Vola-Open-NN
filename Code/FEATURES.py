import plotly.graph_objects as go

import os

import pandas as pd

# timeframes = ["1min", "2min", "5min", "10min", "15min", "30min", "60min", "120min", "240min", "390min"]
timeframes = ["5min"]

def show_plot(data):
    fig = go.Figure()
    # fig.add_trace(go.Scatter(x=data.index, y=data['Open'], mode='lines', name='Open'))
    # fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))
    # fig.add_trace(go.Scatter(x=data.index, y=data['High'], mode='lines', name='High'))
    # fig.add_trace(go.Scatter(x=data.index, y=data['Low'], mode='lines', name='Low'))
    fig.add_trace(go.Scatter(x=data.index, y=data['Volume'], mode='markers', name='Volume'))
    # fig.add_trace(go.Scatter(x=data.index, y=data['Up Ticks'], mode='markers', name='Up Ticks'))
    # fig.add_trace(go.Scatter(x=data.index, y=data['Down Ticks'], mode='markers', name='Down Ticks'))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA-20'], mode='lines', name='SMA-20'))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA-100'], mode='lines', name='SMA-100'))               
    # fig.add_trace(go.Scatter(x=data.index, y=data['EMA-20'], mode='lines', name='EMA-20'))
    # fig.add_trace(go.Scatter(x=data.index, y=data['EMA-100'], mode='lines', name='EMA-100'))
    # fig.add_trace(go.Scatter(x=data.index, y=data['SMAC-20'], mode='markers', name='SMAC-20'))
    # fig.add_trace(go.Scatter(x=data.index, y=data['BOLU-20'], mode='lines', name='BOLU'))
    # fig.add_trace(go.Scatter(x=data.index, y=data['BOLD-20'], mode='lines', name='BOLD'))
    # fig.add_trace(go.Scatter(x=data.index, y=data['SD-100'], mode='lines', name='SD-20'))
    # fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD'))
    # fig.add_trace(go.Scatter(x=data.index, y=data['MACDC'], mode='markers', name='MACDC'))

    fig.show()

def amount_of_variation(data, baseURL):
    fig = go.Figure()
    std = data.std()
    std = std.sort_values(ascending = False)
    fig.add_trace(go.Scatter(x=std.index, y=std.values, mode='markers'))
    
    directory = baseURL + "/features-selection"

    if not os.path.exists(directory):
        os.mkdir(directory)
        print("Directory " , directory ,  " Created ")
    else:    
        print("Directory " , directory ,  " already exists")

    url = directory + "/amount-of-variation.html"
    std = pd.DataFrame({'Feature':std.index, 'Score':std.values})
    std.to_csv(directory + "/amount-of-variation.csv", columns=["Feature", "Score"])

    fig.write_html(url, auto_open=True)

for timeframe in timeframes:
    timeURL = 'D:/Documents/DNN-Trading/Time/' + timeframe
    data = pd.read_csv(timeURL + "/xy-array.csv")
    # show_plot(data)
    amount_of_variation(data, timeURL)