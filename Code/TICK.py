import time
import os
from datetime import datetime

import pandas as pd

stockNames = ["AMZN","ABT","ACN","AAPL","BA","CSCO","CVX","DOW","FB","MO","NFLX"]

dayEnd = datetime.today().strftime('%Y-%m-%d')
dayStart = (pd.to_datetime(dayEnd) - pd.Timedelta('10 day')).strftime('%Y-%m-%d')

print(dayStart, dayEnd)

tickURL = '../Data/Input/Tick/'

for stock in stockNames:

    stockURL = tickURL + stock

    data = pd.read_csv(stockURL + '.csv', index_col='Date and Time', usecols=["Date and Time", "Open", "High", "Low", "Close"])
    data.columns = ["Open", "High", "Low", "Close"]

    if not os.path.exists(stockURL):
        os.mkdir(stockURL)
        print("Directory " , stock ,  " Created ")
    else:    
        print("Directory " , stock ,  " already exists")

    data.index = pd.to_datetime(data.index)

    filteredData = data.between_time('09:30', '16:00')

    day = dayStart

    readStartTime = time.time()
    while day != dayEnd:
        intradayData = filteredData[day:day]
        if intradayData.empty == True:
            day = (pd.to_datetime(day) + pd.Timedelta('1 day')).strftime('%Y-%m-%d')
        else:
            intradayData.to_csv(stockURL + "/" + day + ".csv")
            print("symbol", stock, "day", day, ", OK!")
            day = (pd.to_datetime(day) + pd.Timedelta('1 day')).strftime('%Y-%m-%d')
    print("%s seconds" % (time.time() - readStartTime))