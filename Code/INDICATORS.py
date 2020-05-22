import numpy as np
import pandas as pd

import time

def create_std_deviation(price_array, window):
    length = len(price_array)
    if len(price_array) < window:
        return None
    std = np.zeros(len(price_array))
    mean = create_sma(price_array, window)[window:,0]
    for i in range(window):
        std[window:] = np.add(std[window:], np.multiply(np.subtract(price_array[i:i-window], mean), np.subtract(price_array[i:i-window], mean)))
    std = np.sqrt(std / window)
    return std[:, np.newaxis]

def create_typical_price(high_array, low_array, close_array):
    typ = np.add(np.add(high_array, low_array), close_array)/3
    return typ

def create_sma(price_array, window):
    if len(price_array) < window:
        return None
    sma = np.zeros(len(price_array))
    for i in range(window):
        sma[window:] = np.add(sma[window:], price_array[i:i-window])
    sma = sma / window
    return sma[:, np.newaxis]

def create_ema(price_array, sma, window):
    if len(price_array) < window:
        return None
    c = 2./float(window + 1)
    ema = np.zeros(len(price_array))
    for i in range(window, len(price_array)):
        if i == window:
            ema[i] = sma[i]
        else:
            ema[i] = c*(price_array[i] - ema[i-1]) + ema[i-1]
    return ema[:, np.newaxis]

def create_mom(price_array, window):
    mom =  np.zeros(len(price_array))
    for i in range(window, len(price_array)):
        mom[i] = price_array[i] - price_array[i-window]
    return mom

def create_bollband(high_array, low_array, close_array, window = 20, nb_std_dev = 2):
    typical_array = create_typical_price(high_array, low_array, close_array)
    std_deviation = create_std_deviation(typical_array, window)[:,0] * nb_std_dev
    ma_array = create_sma(typical_array, window)[:,0]
    bolu_array = np.zeros(len(high_array))
    bolu_array[window:] = np.add(ma_array[window:], std_deviation[window:])
    bold_array = np.zeros(len(high_array))
    bold_array[window:] = np.subtract(ma_array[window:], std_deviation[window:])
    return bold_array[:, np.newaxis], bolu_array[:, np.newaxis]

def create_macd(price_array, window = [12, 26]):
    sma_12 = create_sma(price_array, window[0])[:,0]
    sma_26 = create_sma(price_array, window[1])[:,0]
    ema_12 = create_ema(price_array, sma_12, window[0])[:,0]
    ema_26 = create_ema(price_array, sma_26, window[1])[:,0]
    diff_ema = ema_12 - ema_26
    sma_9 = create_sma(diff_ema, 9)[:,0]
    v = create_ema(diff_ema, sma_9, 9)[:,0]
    return np.subtract(diff_ema, v)[:, np.newaxis]

def create_return(price_array, window):
    output = np.zeros(len(price_array))
    for i in range(window, len(price_array)):
        output[i] = float(price_array[i+1] - price_array[i+1-window])/float(price_array[i+1-window])
        if i+2 == len(price_array): break
    return output

def create_up_down(price_array, window):
    pastUD = np.zeros(len(price_array))
    for i in range(window+1, len(price_array)):
        pastUD[i] = window - 2*np.sum(price_array[i-window:i] < price_array[i-window-1:i-1])
    return pastUD

def create_day_since_cross(cross_array):
    day_since_cross = np.zeros(len(cross_array))
    num = 0
    for i in range(len(cross_array)):
        if cross_array[i] == 0:
            num += 1
        else:
            num = 0
        day_since_cross[i] = num
    return day_since_cross

def create_macd_cross(macd):
    macd_cross = np.zeros(len(macd))
    for i in range(1, len(macd)):
        if macd[i-1] < 0 and macd[i] > 0:
            macd_cross[i] = 1
        elif macd[i-1] > 0 and macd[i] < 0:
            macd_cross[i] = -1
        else:
            macd_cross[i] = 0
    return macd_cross[:, np.newaxis]

def create_ma_cross(ma, price_array):
    ma_cross = np.zeros(len(ma))
    for i in range(1, len(ma)):
        if ma[i-1] < price_array[i-1] and ma[i] > price_array[i]:
            ma_cross[i] = -1
        elif ma[i-1] > price_array[i-1] and ma[i] < price_array[i]:
            ma_cross[i] = 1
        else:
            ma_cross[i] = 0
    return ma_cross[:, np.newaxis]

def create_rsi(series, period = 14):
    delta = series.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[period-1]] = np.mean( u[:period] ) #first value is sum of avg gains
    u = u.drop(u.index[:(period-1)])
    d[d.index[period-1]] = np.mean( d[:period] ) #first value is sum of avg losses
    d = d.drop(d.index[:(period-1)])
    rs = pd.stats.moments.ewma(u, com=period-1, adjust=False) / \
    pd.stats.moments.ewma(d, com=period-1, adjust=False)
    return 100 - 100 / (1 + rs)

def create_sto(close, low, high, n = 14): 
    STOK = ((close - pd.rolling_min(low, n)) / (pd.rolling_max(high, n) - pd.rolling_min(low, n))) * 100
    STOD = pd.rolling_mean(STOK, 3)
    return STOK[:, np.newaxis], STOD[:, np.newaxis]

def create_class(price_array):
    output = np.zeros(len(price_array))
    for i in range(len(price_array)):
        if price_array[i+1] > price_array[i]:
            output[i] = 1
        if i+2 == len(price_array): break
    return output

def normalize(array):
    # output = np.std(array)
    for i in range(2,32):
        mean = np.mean(array[:,i])
        std = np.std(array[:,i])
        sub = np.subtract(array[:,i], mean)
        array[:,i] = np.divide(sub, std, out=np.zeros_like(sub), where=std!=0)
    # for i in range(array.shape[1]):
    #     mean = np.mean(array[:,i])
    #     print(mean)
    #     std = np.std(array[:,i])
    #     output[:,i] = np.divide(np.subtract(array[:,i], mean), std)
    return array

def preprocess(data):
    data_array = data.to_numpy()

    datetime = data_array[:,0][:, np.newaxis]
    date = data_array[:,1][:, np.newaxis]
    timeT = data_array[:,2][:, np.newaxis]

    open_price = data_array[:,3]
    high_price = data_array[:,4]
    low_price = data_array[:,5]
    close_price = data_array[:,6]

    readStartTime = time.time()
    typical_price = create_typical_price(high_price, low_price, close_price)
    print("%s seconds for typical_price" % (time.time() - readStartTime))

    readStartTime = time.time()
    sma5 = create_sma(close_price,5)
    sma10 = create_sma(close_price,10)
    sma15 = create_sma(close_price,15)
    sma20 = create_sma(close_price,20)
    sma50 = create_sma(close_price,50)
    sma100 = create_sma(close_price,100)
    sma200 = create_sma(close_price,200)

    print("%s seconds for sma" % (time.time() - readStartTime))

    readStartTime = time.time()
    ema5 = create_ema(close_price, sma5, 5)
    ema10 = create_ema(close_price, sma10, 10)
    ema15 = create_ema(close_price, sma15, 15)
    ema20 = create_ema(close_price, sma20, 20)
    ema50 = create_ema(close_price, sma50, 50)
    ema100 = create_ema(close_price, sma100, 100)
    ema200 = create_ema(close_price, sma200, 200)

    print("%s seconds for ema" % (time.time() - readStartTime))

    readStartTime = time.time()
    smac5 = create_ma_cross(sma5, close_price)
    smac10 = create_ma_cross(sma10, close_price)
    smac15 = create_ma_cross(sma15, close_price)
    smac20 = create_ma_cross(sma20, close_price)
    smac50 = create_ma_cross(sma50, close_price)
    smac100 = create_ma_cross(sma100, close_price)
    smac200 = create_ma_cross(sma200, close_price)
    print("%s seconds for smac" % (time.time() - readStartTime))

    readStartTime = time.time()
    emac5 = create_ma_cross(ema5, close_price)
    emac10 = create_ma_cross(ema10, close_price)
    emac15 = create_ma_cross(ema15, close_price)
    emac20 = create_ma_cross(ema20, close_price)
    emac50 = create_ma_cross(ema50, close_price)
    emac100 = create_ma_cross(ema100, close_price)
    emac200 = create_ma_cross(ema200, close_price)
    print("%s seconds for emac" % (time.time() - readStartTime))

    readStartTime = time.time()
    bold, bolu = create_bollband(high_price, low_price, close_price)
    print("%s seconds for bollinger" % (time.time() - readStartTime))

    readStartTime = time.time()
    sd5 = create_std_deviation(typical_price, 5)
    sd10 = create_std_deviation(typical_price, 10)
    sd15 = create_std_deviation(typical_price, 15)
    sd20 = create_std_deviation(close_price, 20)
    sd50 = create_std_deviation(typical_price, 50)
    sd100 = create_std_deviation(typical_price, 100)
    sd200 = create_std_deviation(typical_price, 200)
    
    print("%s seconds for sd" % (time.time() - readStartTime))

    readStartTime = time.time()
    macd = create_macd(typical_price)
    macdc = create_macd_cross(macd)
    print("%s seconds for macd" % (time.time() - readStartTime))

    readStartTime = time.time()
    data_array = np.append(data_array, sma5, 1)
    data_array = np.append(data_array, sma10, 1)
    data_array = np.append(data_array, sma15, 1)
    data_array = np.append(data_array, sma20, 1)
    data_array = np.append(data_array, sma50, 1)
    data_array = np.append(data_array, sma100, 1)
    data_array = np.append(data_array, sma200, 1)
    print("%s seconds for sma" % (time.time() - readStartTime))

    readStartTime = time.time()
    data_array = np.append(data_array, ema5, 1)
    data_array = np.append(data_array, ema10, 1)
    data_array = np.append(data_array, ema15, 1)
    data_array = np.append(data_array, ema20, 1)
    data_array = np.append(data_array, ema50, 1)
    data_array = np.append(data_array, ema100, 1)
    data_array = np.append(data_array, ema200, 1)
    print("%s seconds for ema" % (time.time() - readStartTime))

    readStartTime = time.time()
    data_array = np.append(data_array, bolu, 1)
    data_array = np.append(data_array, bold, 1)
    print("%s seconds for bollinger" % (time.time() - readStartTime))

    readStartTime = time.time()
    data_array = np.append(data_array, macd, 1)
    print("%s seconds for macd" % (time.time() - readStartTime))

    readStartTime = time.time()
    data_array = np.append(data_array, sd5, 1)
    data_array = np.append(data_array, sd10, 1)
    data_array = np.append(data_array, sd15, 1)
    data_array = np.append(data_array, sd20, 1)
    data_array = np.append(data_array, sd50, 1)
    data_array = np.append(data_array, sd100, 1)
    data_array = np.append(data_array, sd200, 1)
    print("%s seconds for sd" % (time.time() - readStartTime))

    readStartTime = time.time()
    data_array = np.append(data_array, smac5, 1)
    data_array = np.append(data_array, smac10, 1)
    data_array = np.append(data_array, smac15, 1)
    data_array = np.append(data_array, smac20, 1)
    data_array = np.append(data_array, smac50, 1)
    data_array = np.append(data_array, smac100, 1)
    data_array = np.append(data_array, smac200, 1)
    print("%s seconds for smac" % (time.time() - readStartTime))

    readStartTime = time.time()
    data_array = np.append(data_array, emac5, 1)
    data_array = np.append(data_array, emac10, 1)
    data_array = np.append(data_array, emac15, 1)
    data_array = np.append(data_array, emac20, 1)
    data_array = np.append(data_array, emac50, 1)
    data_array = np.append(data_array, emac100, 1)
    data_array = np.append(data_array, emac200, 1)
    print("%s seconds for emac" % (time.time() - readStartTime))

    readStartTime = time.time()
    data_array = np.append(data_array, macdc, 1)
    print("%s seconds for macdc" % (time.time() - readStartTime))

    return data_array