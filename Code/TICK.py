"""Splits Tick Data by date and saves as CSV
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
from datetime import datetime
import pandas as pd
# ----------------------------------------------------------------- Parameters
stockNames = ["AMZN","ABT","ACN","AAPL","BA","CSCO","CVX","DOW","FB","MO","NFLX"]
tickURL = '../Data/Input/Tick/'
# ----------------------------------------------------------------- Body
def split(dayStart, dayEnd):
    """Split into CSV files

    Args:
        dayStart (str "%Y-%M-%d"): lower bound
        dayEnd (str "%Y-%M-%d"): upper bound
    """    
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

# ----------------------------------------------------------------- Test
def test():
    """This function is internal to TICK.py, it is meant for debugging but also serves as unit test
    """    
    dayEnd = datetime.today().strftime('%Y-%m-%d')
    dayStart = (pd.to_datetime(dayEnd) - pd.Timedelta('10 day')).strftime('%Y-%m-%d')
    split(dayStart, dayEnd)