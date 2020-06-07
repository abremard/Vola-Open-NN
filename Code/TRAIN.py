import pandas as pd
import numpy as np
import os
import math
import tensorflow as tf
from tensorflow import keras

import datetime
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

# !!!!!!!!!!!!!!!!!!!!! Wait for new data and change stockNames array to split !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Variables
stockNames = ["AMZN","ABT","ACN","AAPL","AMGN","ADBE","BA","CCEP","CMCSA","CSCO","CVX","DOW","FB","HD","INTC","JNJ","MO","NFLX"]
dayEnd = '2020-05-30'
dayStart = '2019-01-01'
dataSize = 4878
trainSize = 4200
inputNeurones = 79

# Initialise
testSize = dataSize - trainSize
groupedData = np.empty((dataSize,inputNeurones,))
groupedLabel = np.empty((dataSize,))
trainData = np.empty((trainSize,inputNeurones,))
trainLabel = np.empty((trainSize,))
testData = np.empty((testSize,inputNeurones,))
testLabel = np.empty(testSize,)
i = 0

# Volatile
volaOpen = pd.read_csv("../Data/Output/WL/Benchmark/benchmark.csv", usecols=["Symbol", "Date", "Score"])

endReached = False

timeURL = "../Data/Input/Time/"

# Match data to label
for stock in stockNames:
    # Initialise
    endReached = False
    day = dayStart
    # Loop into data
    while not(endReached):
        fname = timeURL + stock + "/" + day + ".csv"
        if os.path.isfile(fname):
            data = pd.read_csv(fname, usecols=['Close'], nrows=100)

            day = (pd.to_datetime(day) + pd.Timedelta('1 day')).strftime('%Y-%m-%d')
            fname = timeURL + stock + "/" + day + ".csv"    

            while(not(os.path.isfile(fname))):
                day = (pd.to_datetime(day) + pd.Timedelta('1 day')).strftime('%Y-%m-%d')
                fname = timeURL + stock + "/" + day + ".csv" 
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

n_split=5

for train_index,test_index in KFold(n_split).split(groupedData):
    x_train,x_test=groupedData[train_index],groupedData[test_index]
    y_train,y_test=groupedLabel[train_index],groupedLabel[test_index]
    
    x_train = keras.utils.normalize(x_train, axis = 1)
    x_test = keras.utils.normalize(x_test, axis = 1)

    model = keras.Sequential()

    model.add(keras.layers.Dense(32, kernel_constraint=keras.constraints.max_norm(3), bias_constraint=keras.constraints.max_norm(3)))

    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print('---------------------TRAIN--------------------')
    history = model.fit(x_train, y_train, epochs=50, validation_split = 0.2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    print('---------------------TEST--------------------')
    model.evaluate(x_test,y_test)

# trainData = groupedData[:trainSize]
# trainLabel = groupedLabel[:trainSize]
# testData = groupedData[trainSize:]
# testLabel = groupedLabel[trainSize:]

# trainData = keras.utils.normalize(trainData, axis = 1)
# testData = keras.utils.normalize(testData, axis = 1)

# print(trainData)

# mean = 0
# mean0 = 0
# mean1 = 0

# for a in range(1):

#     model = keras.Sequential()

#     model.add(keras.layers.Dense(32, input_dim=79, activation='relu'))
#     model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#     print('---------------------TRAIN--------------------')
#     history = model.fit(trainData, trainLabel, epochs=400, validation_split = 0.5)
#     plt.plot(history.history['accuracy'])
#     plt.plot(history.history['val_accuracy'])
#     plt.title('model accuracy')
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.legend(['Train', 'Validation'], loc='upper left')
#     plt.show()
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('Model loss')
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Validation'], loc='upper left')
#     plt.show()

#     print('---------------------TEST--------------------')
#     test_loss, test_acc = model.evaluate(testData, testLabel)
#     acc0 = 0
#     tot0 = 0
#     acc1 = 0
#     tot1 = 0
#     predictions = model.predict(testData)
#     print('Test accuracy', test_acc)
#     print('Test loss', test_loss)

#     for i in range(len(predictions)):
#         # print(int(round(predictions[i][0])), testLabel[i])
#         if round(predictions[i][0]) == testLabel[i] :
#             if round(predictions[i][0]) == 0 :
#                 acc0 = acc0+1
#                 tot0 = tot0+1
#             else:
#                 acc1 = acc1+1
#                 tot1 = tot1+1
#         else:
#             if round(predictions[i][0]) == 0 :
#                 tot0 = tot0+1
#             else:
#                 tot1 = tot1+1

#     mean0 = mean0 + acc0*100/tot0
#     mean1 = mean1 + acc1*100/tot1
#     mean = mean + test_acc

# print(mean/100)
# print(mean0/100)
# print(mean1/100)
# print('Nothing to report!')
