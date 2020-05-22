import time
import os

import pandas as pd
from datetime import datetime

stockNames = ["AAPL", "ABT","ACN","ADBE","AMGN","AMZN","BA","BA","CCEP","CMCSA","CSCO","CVX","DOWWI","FB","HD","INTC","JNJ","MO","NFLX"]

dayEnd = '2020-05-30'
dayStart = '2019-01-01'

for dirName in stockNames:

    data = pd.read_csv('D:/Documents/DNN-Trading/Tick/' + dirName + '.csv', index_col='Date and Time', usecols=["Date and Time", "Open", "High", "Low", "Close"])
    data.columns = ["Open", "High", "Low", "Close"]

    if not os.path.exists("D:/Documents/DNN-Trading/Tick/" + dirName):
        os.mkdir("D:/Documents/DNN-Trading/Tick/" + dirName)
        print("Directory " , dirName ,  " Created ")
    else:    
        print("Directory " , dirName ,  " already exists")

    data.index = pd.to_datetime(data.index)

    filteredData = data.between_time('09:30', '16:00')

    day = dayStart

    readStartTime = time.time()
    while day != dayEnd:
        intradayData = filteredData[day:day]
        if intradayData.empty == True:
            day = (pd.to_datetime(day) + pd.Timedelta('1 day')).strftime('%Y-%m-%d')
        else:
            intradayData.to_csv("D:/Documents/DNN-Trading/Tick/" + dirName + "/" + day + ".csv")
            print("symbol", dirName, "day", day, ", OK!")
            day = (pd.to_datetime(day) + pd.Timedelta('1 day')).strftime('%Y-%m-%d')
    print("%s seconds" % (time.time() - readStartTime))