"""Utility class for generating xy_arrays
    Author :
        Alexandre Bremard
    Contributors :
        -
    Version Control :
        0.1 - 09/06/2020 : generate
"""
# ----------------------------------------------------------------- External Imports 
import pandas as pd
import numpy as np
import os
import math
import datetime
# ----------------------------------------------------------------- Parameters
timeframe = '5min'
# dataSize = 4840
dataSize = 2200
col = ["SMA-20", "SMA-200", "Volume"]
# col = ["Open", "High", "Low", "Close", "Volume", "Up Ticks", "Down Ticks", "SMA-5", "SMA-10", "SMA-15", "SMA-20", "SMA-50", "SMA-100", "SMA-200", "EMA-5", "EMA-10", "EMA-15", "EMA-20", "EMA-50", "EMA-100", "EMA-200", "BOLU-20", "BOLD-20", "SD-5", "SD-10", "SD-15", "SD-20", "SD-50", "SD-100", "MACD", "SMAC-5", "SMAC-10", "SMAC-15", "SMAC-20", "SMAC-50", "SMAC-100", "SMAC-200", "EMAC-5", "EMAC-10", "EMAC-15", "EMAC-20", "EMAC-50", "EMAC-100", "EMAC-200", "MACDC"]
nbCandles = 78
# stockNames = ["AMZN","ABT","ACN","AAPL","AMGN","ADBE","BA","CCEP","CMCSA","CSCO","CVX","DOW","FB","HD","INTC","JNJ","MO","NFLX"]
stockNames = ["AMZN","ABT","ACN","AAPL","BA","CSCO","CVX","DOW","FB","MO","NFLX"]
timeURL = "../Data/Input/Time/"
# ----------------------------------------------------------------- Body
def generate(windowSize, dayStart, dayEnd):
    """Link training set to prediction values

    Args:
        windowSize (int): periods back in time used to determine training set features
        dayStart (str "%Y-%M-%d"): lower bound
        dayEnd (str "%Y-%M-%d"): upper bound

    Output:
        xy_array (pd.dataframe) file is saved as CSV (useful for statistic analysis)
    """    
    inputNeurones = windowSize * len(col)

    # Initialize
    groupedData = np.empty((dataSize,inputNeurones,))
    groupedLabel = np.empty((dataSize,))
    i = 0

    # Volatile
    volaOpen = pd.read_csv("../Data/Output/WL/Benchmark/benchmark.csv", usecols=["Symbol", "Date", "Score"])

    endReached = False


    # Match data to label
    for stock in stockNames:
        # Initialize
        endReached = False
        day = dayStart
        # Loop into data
        while not(endReached):
            fname = timeURL + timeframe + '/' + stock + "/" + day + ".csv"
            if os.path.isfile(fname):
                data = pd.read_csv(fname, usecols=col, skiprows=range(1, nbCandles - windowSize + 2))

                day = (pd.to_datetime(day) + pd.Timedelta('1 day')).strftime('%Y-%m-%d')
                fname = timeURL + timeframe + '/' + stock + "/" + day + ".csv"    

                while(not(os.path.isfile(fname))):
                    day = (pd.to_datetime(day) + pd.Timedelta('1 day')).strftime('%Y-%m-%d')
                    fname = timeURL + timeframe + '/' + stock + "/" + day + ".csv" 
                    if day == dayEnd:
                        endReached = True
                        break

                if endReached:
                    break

                predictionValue = volaOpen.loc[(volaOpen['Date'] == day) & (volaOpen['Symbol'] == stock)].iloc[0]['Score']
                if predictionValue:
                    label = not(math.isnan(predictionValue) or predictionValue <= 0)
                    groupedLabel[i] = label
                    groupedData[i] = data.values.flatten()
                    i = i+1
            else:
                day = (pd.to_datetime(day) + pd.Timedelta('1 day')).strftime('%Y-%m-%d')
                if day == dayEnd:
                    endReached = True

    col_names = []

    # generate feature names
    for i in range(windowSize):
        for c in reversed(col):
            col_names.append(c+"-"+str(windowSize-i-1))

    groupedData = pd.DataFrame(data=groupedData, columns=col_names)
    groupedLabel = pd.DataFrame(data=groupedLabel, columns=["Label"])
    xy_array = groupedData.assign(Label = groupedLabel.values)
    xy_array.to_csv(timeURL + timeframe + "/xy-array.csv", index=False)
# ----------------------------------------------------------------- Test
def test():
    """This function is internal to GENERATE_XY.py, it is meant for debugging but also serves as unit test
    """
    dayEnd = '2020-05-30'
    dayStart = '2019-01-01'
    windowSize = 78
    generate(windowSize, dayStart, dayEnd)