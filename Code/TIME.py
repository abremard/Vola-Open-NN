"""Tools for Time-based Data processing
    Author :
        Alexandre Bremard
    Contributors :
        -
    Version Control :
        0.1 - 09/06/2020 : split
"""

# ----------------------------------------------------------------- External Imports 
import time
import os
import numpy as np
import pandas as pd
from datetime import datetime
# ----------------------------------------------------------------- Internal Imports 
import INDICATORS as idc
# ----------------------------------------------------------------- Parameters
stockNames = ["AAPL","ABT","ACN","ADBE","AMGN","AMZN","BA","CCEP","CMCSA","CSCO","CVX","DOW","FB","HD","INTC","JNJ","MO","NFLX"]
features = ["Date", "Time", "Open", "High", "Low", "Close", "Volume", "Up Ticks", "Down Ticks", "SMA-5", "SMA-10", "SMA-15", "SMA-20", "SMA-50", "SMA-100", "SMA-200", "EMA-5", "EMA-10", "EMA-15", "EMA-20", "EMA-50", "EMA-100", "EMA-200", "BOLU-20", "BOLD-20", "MACD", "SD-5", "SD-10", "SD-15", "SD-20", "SD-50", "SD-100", "SD-200", "SMAC-5", "SMAC-10", "SMAC-15", "SMAC-20", "SMAC-50", "SMAC-100", "SMAC-200", "EMAC-5", "EMAC-10", "EMAC-15", "EMAC-20", "EMAC-50", "EMAC-100", "EMAC-200", "MACDC"]
timeframes = ["5min"]
read_col = ["Date and Time", "Date", "Time", "Open", "High", "Low", "Close", "Volume", "Up Ticks", "Down Ticks"]
write_col = ["Date", "Time", "Open", "High", "Low", "Close", "Volume", "Up Ticks", "Down Ticks"]
timeURL = '../Data/Input/Time/'
# ----------------------------------------------------------------- Body
def process_data(dayStart, dayEnd):
    """First creates features using INDICATORS.PY then filters data by date and saves as separate CSV files.
        The processed data can later be used for network training purposes.
    Args:
        dayStart (str "%Y-%M-%d"): lower bound
        dayEnd (str "%Y-%M-%d"): upper bound
    """

    for timeframe in timeframes:
        
        timeframeURL = timeURL + timeframe

        for stock in stockNames:

            baseURL = timeframeURL + '/' + stock

            data = pd.read_csv(baseURL + '.csv', usecols=read_col)

            data["Date"] = pd.DatetimeIndex(data["Date and Time"]).dayofweek

            if not os.path.exists(baseURL):
                os.mkdir(baseURL)
                print("Directory " , stock ,  " Created ")
            else:    
                print("Directory " , stock ,  " already exists")

            data_array = idc.preprocess(data)

            dataframe = pd.DataFrame(data=data_array[:,1:],    # values
                                    index=data_array[:,0],    # 1st column as index
                                    columns=features)

            dataframe.index = pd.to_datetime(dataframe.index)

            filteredData = dataframe.between_time('09:30', '16:00')

            day = dayStart

            readStartTime = time.time()
            while day != dayEnd:
                intradayData = filteredData[day:day]
                if intradayData.empty == True:
                    day = (pd.to_datetime(day) + pd.Timedelta('1 day')).strftime('%Y-%m-%d')
                else:
                    # Output
                    intradayData = idc.normalize(intradayData.to_numpy())
                    intradayData = pd.DataFrame(data=intradayData,    # values
                                            columns=features)
                    intradayData.to_csv(baseURL + "/" + day + ".csv", index=False)
                    print("symbol", stock, "day", day, ", OK!")
                    day = (pd.to_datetime(day) + pd.Timedelta('1 day')).strftime('%Y-%m-%d')
            print("%s seconds" % (time.time() - readStartTime))

        directory = timeframeURL + "/features-selection"
        if not os.path.exists(directory):
            os.mkdir(directory)
            print("Directory " , directory ,  " Created ")
        else:    
            print("Directory " , directory ,  " already exists")

def combine_data(dayStart, dayEnd):
    """Used for data analytics purposes. Combine filtered and processed data into large CSV file that can later be used for feature selection.

    Args:
        dayStart (str "%Y-%M-%d"): lower bound
        dayEnd (str "%Y-%M-%d"): upper bound
    Output:
        dataframe saved as CSV
    """    
    for timeframe in timeframes:

        combined_data = pd.DataFrame()

        timeframeURL = timeURL + timeframe
        directory = timeframeURL + "/features-selection"
        
        for stockName in stockNames:

            endReached = False
            day = dayStart

            while not(endReached):
                fname = timeURL + timeframe + '/' + stockName + "/" + day + ".csv"
                if day == dayEnd:
                    endReached = True
                if os.path.isfile(fname):
                    print(fname)
                    combined_data = combined_data.append(pd.read_csv(fname))
                    day = (pd.to_datetime(day) + pd.Timedelta('1 day')).strftime('%Y-%m-%d')
                else:
                    day = (pd.to_datetime(day) + pd.Timedelta('1 day')).strftime('%Y-%m-%d')
        
        combined_data.to_csv(directory + "/processed-data.csv")
# ----------------------------------------------------------------- Test
def test():
    """This function is internal to TIME.py, it is meant for debugging but also serves as unit test
    """    
    dayEnd = datetime.today().strftime('%Y-%m-%d')
    dayStart = (pd.to_datetime(dayEnd) - pd.Timedelta('10 day')).strftime('%Y-%m-%d')
    process_data(dayStart, dayEnd)
    combine_data(dayStart, dayEnd)
