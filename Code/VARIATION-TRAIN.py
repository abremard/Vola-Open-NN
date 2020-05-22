import pandas as pd
import numpy as np
import os
import math
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix

import datetime
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, LeakyReLU, BatchNormalization, Dropout
from keras.activations import relu, sigmoid
from sklearn.preprocessing import StandardScaler


# !!!!!!!!!!!!!!!!!!!!! Wait for new data and change stockNames array to split !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

timeframe = '5min'
# dataSize = 4840
dataSize = 2958
col = ["SMA-20", "SMA-200", "Volume"]
# col = ["Open", "High", "Low", "Close", "Volume", "Up Ticks", "Down Ticks", "SMA-5", "SMA-10", "SMA-15", "SMA-20", "SMA-50", "SMA-100", "SMA-200", "EMA-5", "EMA-10", "EMA-15", "EMA-20", "EMA-50", "EMA-100", "EMA-200", "BOLU-20", "BOLD-20", "SD-5", "SD-10", "SD-15", "SD-20", "SD-50", "SD-100", "MACD", "SMAC-5", "SMAC-10", "SMAC-15", "SMAC-20", "SMAC-50", "SMAC-100", "SMAC-200", "EMAC-5", "EMAC-10", "EMAC-15", "EMAC-20", "EMAC-50", "EMAC-100", "EMAC-200", "MACDC"]
nbCandles = 78
windowSizes = [78]
# stockNames = ["AMZN","ABT","ACN","AAPL","AMGN","ADBE","BA","CCEP","CMCSA","CSCO","CVX","DOWWI","FB","HD","INTC","JNJ","MO","NFLX"]
stockNames = ["AMZN","ABT","ACN","AAPL","BA","CSCO","CVX","DOWWI","FB","MO","NFLX"]

timeURL = "../Data/Input/Time/"


def annotation(windowSize):
    # col = ["Up Ticks", "Down Ticks", "SD-20", "MACD", "SD-100", "Volume", "SMA-50", "EMA-50", "SMA-100", "EMA-100"]
    inputNeurones = windowSize * len(col)

    # Variables
    # timeframes = ["1min", "2min", "5min", "10min", "15min", "30min", "60min", "120min", "240min", "390min"]
    dayEnd = '2020-05-30'
    dayStart = '2019-01-01'

    # Initialize
    groupedData = np.empty((dataSize,inputNeurones,))
    groupedLabel = np.empty((dataSize,))
    i = 0

    # Volatile
    volaOpen = pd.read_csv("../Data/Output/WL/Benchmark/benchmark.csv", usecols=["Symbol", "Date", "Score"])

    endReached = False


    # Match data to label
    for stock in stockNames:
        # Initialise
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
    groupedData = pd.DataFrame(data=groupedData)
    groupedLabel = pd.DataFrame(data=groupedLabel, columns=["Label"])
    xy_array = groupedData.assign(Label = groupedLabel.values)
    print(xy_array)
    xy_array.to_csv(timeURL + timeframe + "/xy-array.csv", index=False)

def train(groupedData, groupedLabel):
    # print(groupedData)
    # print(groupedLabel)
    n_split = 5
    n_runs = 20

    avgIncreaseLongRun = 0
    avgAccLongRun = 0
    
    # Runs
    for i in range(n_runs):

        benchmark = []

        # Cross-validation
        for train_index,test_index in KFold(n_split).split(groupedData):
            x_train,x_test=groupedData[train_index],groupedData[test_index]
            y_train,y_test=groupedLabel[train_index],groupedLabel[test_index]
            model = keras.Sequential()
            model.add(keras.layers.Dense(64, activation=tf.nn.relu))
            model.add(keras.layers.Dense(16, activation=tf.nn.relu))
            model.add(keras.layers.Dense(1, kernel_initializer= 'glorot_uniform', activation=tf.nn.sigmoid))

            model.compile(keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

            # print('---------------------TRAIN--------------------')
            history = model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=0) 
            # plt.plot(history.history['accuracy'])
            # plt.plot(history.history['val_accuracy'])
            # plt.title('model accuracy')
            # plt.ylabel('accuracy')
            # plt.xlabel('epoch')
            # plt.legend(['Train', 'Validation'], loc='upper left')
            # plt.show()
            # plt.plot(history.history['loss'])
            # plt.plot(history.history['val_loss'])
            # plt.title('Model loss')
            # plt.ylabel('Loss')
            # plt.xlabel('Epoch')
            # plt.legend(['Train', 'Validation'], loc='upper left')
            # plt.show()

            # print('---------------------TEST--------------------')
            model.evaluate(x_test, y_test, verbose = 0)
            acc0 = tot0 = true0 = acc1 = tot1 = true1 = 0
            predictions = model.predict(x_test)

            for i in range(len(predictions)):
                if y_test[i]:
                    true1 = true1 + 1
                else:
                    true0 = true0 + 1
                if round(predictions[i][0]) == y_test[i] :
                    if round(predictions[i][0]) == 0 :
                        acc0 = acc0+1
                        tot0 = tot0+1
                    else:
                        acc1 = acc1+1
                        tot1 = tot1+1
                else:
                    if round(predictions[i][0]) == 0 :
                        tot0 = tot0+1
                    else:
                        tot1 = tot1+1

            # print("acc0", acc0, "tot0", tot0, "true0", true0, "acc1", acc1, "tot1", tot1, "true1", true1)
            withNN = acc1*100/(tot1)
            withoutNN = true1*100/(true0+true1)
            benchmark.append(dict(acc=(acc0 + acc1)*100/(tot0 + tot1), withNN=withNN, withoutNN=withoutNN, increase=round(withNN-withoutNN, 4)))

        averageIncrease = 0
        averageAcc = 0

        for bm in benchmark:
            averageIncrease = averageIncrease + bm['increase']
            averageAcc = averageAcc + bm['withNN']
            print(bm)

        averageIncrease = averageIncrease / n_split
        averageAcc = averageAcc / n_split
        avgIncreaseLongRun = avgIncreaseLongRun + averageIncrease
        avgAccLongRun = avgAccLongRun + averageAcc

        print("Average increase", averageIncrease)
        print("Average accuracy", averageAcc)

    avgIncreaseLongRun = avgIncreaseLongRun / n_runs
    avgAccLongRun = avgAccLongRun / n_runs
    return avgIncreaseLongRun, avgAccLongRun

# *************************** PCA ******************

def PCA_train(x_train):
    from sklearn.decomposition import PCA
    pca_data = PCA(n_components=10)
    principalComponents_data = pca_data.fit_transform(x_train.iloc[:,:-1])
    # principal_data_df = pd.DataFrame(data = principalComponents_data, columns = ['principal component 1', 'principal component 2'])
    principal_data_df = pd.DataFrame(data = principalComponents_data)

    print(principal_data_df.tail())
    print('Explained variation per principal component: {}'.format(pca_data.explained_variance_ratio_))

    # plt.figure()
    # plt.figure(figsize=(10,10))
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=14)
    # plt.xlabel('Principal Component - 1',fontsize=20)
    # plt.ylabel('Principal Component - 2',fontsize=20)
    # plt.title("Principal Component Analysis of Intraday Stock Dataset",fontsize=20)
    # targets = [1, 0]
    # colors = ['g', 'r']
    # for target, color in zip(targets,colors):
    #     indicesToKeep = x_train['Label'] == target
    #     plt.scatter(principal_data_df.loc[indicesToKeep, 'principal component 1']
    #             , principal_data_df.loc[indicesToKeep, 'principal component 5'], c = color, s = 50)

    # plt.legend(targets,prop={'size': 15})

    # plt.show()

    return principal_data_df

# *************************** Grid Search optimizer ******************

def create_model(layers, activation):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes,input_dim=inputNeurones))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
            
    model.add(Dense(units = 1, kernel_initializer= 'glorot_uniform', activation = 'sigmoid')) # Note: no activation beyond this point
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("layers", layers, "activation", activation)

    return model

def grid_opti(x_train, y_train):

    # Feature Scaling
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)

    model = KerasClassifier(build_fn=create_model)

    layers = [[20], [40, 20], [45, 30, 15]]
    activations = ['sigmoid', 'relu']
    param_grid = dict(layers=layers, activation=activations, batch_size = [128, 256], epochs=[50])
    grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=5)

    grid_result = grid.fit(x_train, y_train, validation_split = 0.2)

    [grid_result.best_score_,grid_result.best_params_]

# *************************** MAIN ***********************************

for windowSize in windowSizes:
    # annotation(windowSize)
    xy_array = pd.read_csv(timeURL + timeframe + "/xy-array.csv")
    groupedData = xy_array.iloc[:,:-1].to_numpy()
    groupedLabel = xy_array["Label"].to_numpy()
    averageIncrease, averageAcc = train(groupedData, groupedLabel)
    print("Timeframe", timeframe, "features", col, "Window size", windowSize, "Average increase", averageIncrease, "Average accuracy", averageAcc)

    # pca_data = PCA_train(xy_array)
    # averageIncrease, averageAcc = train(pca_data.to_numpy(), groupedLabel)
    # print("Timeframe", timeframe, "features", col, "Window size", windowSize, "Average increase", averageIncrease, "Average accuracy", averageAcc)

    # grid_opti(groupedData, groupedLabel)
