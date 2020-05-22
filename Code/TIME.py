# ********************************** WORKS FOR EVERY FRAME EXCEPT DAY FRAMES FOR NOW ********************************************

import time
import os

import numpy as np
import pandas as pd
from datetime import datetime

import INDICATORS as idc

stockNames = ["AAPL","ABT","ACN","ADBE","AMGN","AMZN","BA","CCEP","CMCSA","CSCO","CVX","DOWWI","FB","HD","INTC","JNJ","MO","NFLX"]
features = ["Date", "Time", "Open", "High", "Low", "Close", "Volume", "Up Ticks", "Down Ticks", "SMA-5", "SMA-10", "SMA-15", "SMA-20", "SMA-50", "SMA-100", "SMA-200", "EMA-5", "EMA-10", "EMA-15", "EMA-20", "EMA-50", "EMA-100", "EMA-200", "BOLU-20", "BOLD-20", "MACD", "SD-5", "SD-10", "SD-15", "SD-20", "SD-50", "SD-100", "SD-200", "SMAC-5", "SMAC-10", "SMAC-15", "SMAC-20", "SMAC-50", "SMAC-100", "SMAC-200", "EMAC-5", "EMAC-10", "EMAC-15", "EMAC-20", "EMAC-50", "EMAC-100", "EMAC-200", "MACDC"]

dayEnd = '2020-05-01'
dayStart = '2019-01-01'

# timeframes = ["1min", "2min", "5min", "10min", "15min", "30min", "60min", "120min", "240min", "390min"]
timeframes = ["1min", "2min", "5min"]
# timeframes = ["10min", "15min", "30min", "60min", "120min", "240min", "390min"]

read_col = ["Date and Time", "Date", "Time", "Open", "High", "Low", "Close", "Volume", "Up Ticks", "Down Ticks"]
write_col = ["Date", "Time", "Open", "High", "Low", "Close", "Volume", "Up Ticks", "Down Ticks"]


def process_data():

    for timeframe in timeframes:

        # features_data = pd.DataFrame(columns=features)
        
        timeURL = 'D:/Documents/DNN-Trading/Time/' + timeframe

        for dirName in stockNames:

            baseURL = timeURL + '/' + dirName

            data = pd.read_csv(baseURL + '.csv', usecols=read_col)

            data["Date"] = pd.DatetimeIndex(data["Date and Time"]).dayofweek

            if not os.path.exists(baseURL):
                os.mkdir(baseURL)
                print("Directory " , dirName ,  " Created ")
            else:    
                print("Directory " , dirName ,  " already exists")

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
                    # features_data = features_data.append(intradayData)
                    print("symbol", dirName, "day", day, ", OK!")
                    day = (pd.to_datetime(day) + pd.Timedelta('1 day')).strftime('%Y-%m-%d')
            print("%s seconds" % (time.time() - readStartTime))

        directory = timeURL + "/features-selection"
        if not os.path.exists(directory):
            os.mkdir(directory)
            print("Directory " , directory ,  " Created ")
        else:    
            print("Directory " , directory ,  " already exists")
        # features_data.to_csv(directory + "/processed-data.csv")

def combine_data():

    for timeframe in timeframes:

        combined_data = pd.DataFrame()

        timeURL = 'D:/Documents/DNN-Trading/Time/' + timeframe
        directory = timeURL + "/features-selection"
        
        for stockName in stockNames:

            endReached = False
            day = dayStart

            while not(endReached):
                fname = "D:/Documents/DNN-Trading/Time/" + timeframe + '/' + stockName + "/" + day + ".csv"
                if day == dayEnd:
                    endReached = True
                if os.path.isfile(fname):
                    print(fname)
                    combined_data = combined_data.append(pd.read_csv(fname))
                    day = (pd.to_datetime(day) + pd.Timedelta('1 day')).strftime('%Y-%m-%d')
                else:
                    day = (pd.to_datetime(day) + pd.Timedelta('1 day')).strftime('%Y-%m-%d')
        
        combined_data.to_csv(directory + "/processed-data.csv")




combine_data()
# process_data()
