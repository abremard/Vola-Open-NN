"""Main project job that runs daily.
    1) Data is first refreshed through TickWrite Schedule
    2) Data is then split into date files using TICKDATA.PY and TIMEDATA.PY
    3) TODO WL simulation on the day before
    4) Training data is linked to prediction values into standard xy_array
    5) Prediction data is preprocessed, features are created using Time Data
    6) Neural Network is trained
    7) Predictions are sent by mail to recipients

    Author :
        Alexandre Bremard
    Version Control :
        1.0 - 07/06/2020 : preprocess, train, send message
"""

# ----------------------------------------------------------------- External Imports
import pandas as pd
import numpy as np
import os
import math
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import time
from mailjet_rest import Client
import os
import base64
import csv
import TICKDATA as tickdata
import TIMEDATA as timedata
from io import StringIO
import datetime
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
# ----------------------------------------------------------------- Internal Imports
import INDICATORS as idc
import SYMBOL as sb
import GENERATE_XY as generator
# ----------------------------------------------------------------- Parameters
timeframe = '5min'
dataSize = 2958
col = ["SMA-20", "SMA-200", "Volume"]
nbCandles = 78
windowSize = 78
stockNames = ["AMZN","ABT","ACN","AAPL","BA","CSCO","CVX","DOW","FB","MO","NFLX"]
dayStart = '2019-01-01'
dayEnd = '2020-05-30'
timeURL = "../Data/Input/Time/"
# tradingDay is current day
tradingDay = datetime.datetime.today().strftime('%Y-%m-%d')
# predictionDay is the day before (in terms of market days)
predictionDay = (pd.to_datetime(tradingDay) - pd.Timedelta('1 day')).strftime('%Y-%m-%d')
while datetime.datetime.strptime(predictionDay, '%Y-%m-%d').weekday() in (5,6):
    predictionDay = (pd.to_datetime(predictionDay) - pd.Timedelta('1 day')).strftime('%Y-%m-%d')
# ----------------------------------------------------------------- Body
def preprocess_test_data():
    """This functions will generate synthetized features i.e. indicators on the previous day set of data.
        All features are generated regardless of network usage.

        Output:
            intradayData (pd.dataframe): saved as CSV file (can be used for statistic analysis purposes)
    """    
    read_col = ["Date and Time", "Date", "Time", "Open", "High", "Low", "Close", "Volume", "Up Ticks", "Down Ticks"]
    stockNames = ["AAPL","ABT","ACN","ADBE","AMGN","AMZN","BA","CCEP","CMCSA","CSCO","CVX","DOW","FB","HD","INTC","JNJ","MO","NFLX"]
    features = ["Date", "Time", "Open", "High", "Low", "Close", "Volume", "Up Ticks", "Down Ticks", "SMA-5", "SMA-10", "SMA-15", "SMA-20", "SMA-50", "SMA-100", "SMA-200", "EMA-5", "EMA-10", "EMA-15", "EMA-20", "EMA-50", "EMA-100", "EMA-200", "BOLU-20", "BOLD-20", "MACD", "SD-5", "SD-10", "SD-15", "SD-20", "SD-50", "SD-100", "SD-200", "SMAC-5", "SMAC-10", "SMAC-15", "SMAC-20", "SMAC-50", "SMAC-100", "SMAC-200", "EMAC-5", "EMAC-10", "EMAC-15", "EMAC-20", "EMAC-50", "EMAC-100", "EMAC-200", "MACDC"]

    for stock in stockNames:

        data = pd.read_csv('../Data/Output/Production/' + stock + '.csv', usecols=read_col)
        cols = data.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        data = data[cols]

        print(data)

        data["Date"] = pd.DatetimeIndex(data["Date and Time"]).dayofweek

        data_array = idc.preprocess(data)

        dataframe = pd.DataFrame(data=data_array[:,1:],    # values
                                index=data_array[:,0],    # 1st column as index
                                columns=features)

        dataframe.index = pd.to_datetime(dataframe.index)

        filteredData = dataframe.between_time('09:30', '16:00')

        intradayData = filteredData[predictionDay:predictionDay]
        # Output
        intradayData = idc.normalize(intradayData.to_numpy())
        intradayData = pd.DataFrame(data=intradayData, columns=features)
        intradayData.to_csv("../Data/Output/Production/processed/" + stock + ".csv", index=False)
        print("symbol", stock, ", OK!")

def train(groupedData, groupedLabel):
    """Training algorithm. Binary Crossentropy with 1 hidden layer. Network remains very simplified for now and is to be upgraded in the future.

    Benchmark:
        69.8% average precision

    Args:
        groupedData (pd.dataframe): Data
        groupedLabel (pd.dataframe): Label
    """    
    inputNeurones = windowSize * len(col)

    x_train = groupedData.to_numpy()
    y_train = groupedLabel.to_numpy()
    
    model = keras.Sequential()
    model.add(keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, kernel_initializer= 'glorot_uniform', activation=tf.nn.sigmoid))

    model.compile(keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=100, batch_size=64) 

    predictionList = pd.DataFrame(columns=['Stock', 'Prediction', 'Round'])

    # Predictions
    for index, stock in enumerate(stockNames):
        data = pd.read_csv("../Data/Output/Production/processed/" + stock + ".csv", usecols=col, skiprows=range(1, nbCandles - windowSize + 2))
        groupedData = np.empty((1,inputNeurones,))
        groupedData[0] = data.values.flatten()
        prediction = model.predict(groupedData)
        predictionList.loc[index] = [stock, prediction[0][0], round(prediction[0][0])]

    print(predictionList)
    
    predictionList.to_csv("../Data/Output/Production/predictions.csv")

def send_message(recipients):
    """Sends mail to project owners using Mailjet API endpoint

    Args:
        recipients ([dict()]): JSON-like recipient array
    """    
    data = open("../Data/Output/Production/predictions.csv", 'rb').read()
    base64_encoded = base64.b64encode(data).decode('UTF-8')
    api_key = '1d63a0438536320237a4c1b853df59c9'
    api_secret = '8e9ec8191b0183573a0cd94750aa37d8'
    mailjet = Client(auth=(api_key, api_secret), version='v3.1')
    data = {
    'Messages': [
        {
        "From": {
            "Email": "mogilno.trading@gmail.com",
            "Name": "Mogilno"
        },
        "To": recipients,
        "Subject": "Prediction Vola Open pour le " + tradingDay,
        "TextPart": "Ceci est un mail automatique.\nTableau des résultats de prédictions par le réseau pour la journée du " + predictionDay + ".\nLa valeur de prédiction est comprise entre 0 et 1:\n1 étant fortement recommandé, 0 étant pas du tout recommandé.",
        "Attachments": [
            {
                "ContentType": "text/plain",
                "Filename": "predictions.csv",
                "Base64Content": base64_encoded
            }
        ],
        "CustomID": "AppPredictionData"
        }
    ]
    }
    result = mailjet.send.create(data=data)
    print (result.status_code)
    print (result.json())

def job():
    """Main job
    """
    highBound = datetime.datetime.today().strftime('%Y-%m-%d')
    lowBound = (pd.to_datetime(highBound) - pd.Timedelta('10 day')).strftime('%Y-%m-%d')
    timedata.process_data(lowBound, highBound)
    tickdata.split(lowBound, highBound)
    # generator.generate(windowSize, dayStart, dayEnd)
    xy_array = pd.read_csv(timeURL + timeframe + "/xy-array.csv")
    groupedData = xy_array.iloc[:,:-1]
    groupedLabel = xy_array["Label"]
    # sb.rename('../Data/Output/Production/')
    preprocess_test_data()
    train(groupedData, groupedLabel)
    recipients = [
            {"Email": "bremard.alexandre@gmail.com",
            "Name": "Alexandre"},
            {"Email": "philippe.bremard.idts@gmail.com",
            "Name": "Philippe"},
            ]
    send_message(recipients)

# ----------------------------------------------------------------- Test
def test():
    """This function is internal to PRODUCTION.py, it is meant for debugging but also serves as unit test
    """
    print("----- TEST FOR PRODUCTION.PY -----")
    highBound = datetime.datetime.today().strftime('%Y-%m-%d')
    lowBound = (pd.to_datetime(highBound) - pd.Timedelta('10 day')).strftime('%Y-%m-%d')
    timedata.process_data(lowBound, highBound)
    tickdata.split(lowBound, highBound)
    # generator.generate(windowSize, dayStart, dayEnd)
    xy_array = pd.read_csv(timeURL + timeframe + "/xy-array.csv")
    groupedData = xy_array.iloc[:,:-1]
    groupedLabel = xy_array["Label"]
    # sb.rename('../Data/Output/Production/')
    preprocess_test_data()
    train(groupedData, groupedLabel)
    recipients = [
            {"Email": "bremard.alexandre@gmail.com",
            "Name": "Alexandre"}
            ]
    send_message(recipients)
    print("----------------------------------")

job()
# test()