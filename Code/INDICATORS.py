"""Toolbox to generate indicators based on OHLC-Volume, please refer to Investopedia for indicators detail and formulas
    https://www.investopedia.com/

    Author :
        Alexandre Bremard
    Citations :
        "Machine Learning in Intraday Stock Trading" : Art Paspanthong, Nick Tantivasadakarn, Will Vithayapalert form Stanford University
        http://cs229.stanford.edu/proj2019spr/report/28.pdf
    Version Control :
        1.0 - 07/06/2020 : std deviation, typical price, sma, ema, momentum, bollband, macd, days since cross, macd cross, ma cross, rsi, stochastic, normalization
"""

# ----------------------------------------------------------------- External Imports
import numpy as np
import pandas as pd
import time
# ----------------------------------------------------------------- Body
def create_std_deviation(price_array, window):
    """Standard price deviation for a given time window

    Args:
        price_array (np.array): typical price most of the case
        window (int): number of periods back in time 

    Returns:
        std (np.array): std deviation array
    """    
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
    """Typical price as defined in investopedia

    Args:
        high_array (np.array): High
        low_array (np.array): Low
        close_array (np.array): Close

    Returns:
        typ (np.array): Typical Price
    """    
    typ = np.add(np.add(high_array, low_array), close_array)/3
    return typ

def create_sma(price_array, window):
    """Simple Moving Average Generator on a given period, cumulative sum is being used for time efficiency

    Args:
        price_array (np.array): closing price most of the case
        window (int): number of periods back in time 

    Returns:
        sma (np.array): SMA
    """    
    if len(price_array) < window:
        return None
    sma = np.zeros(len(price_array))
    for i in range(window):
        sma[window:] = np.add(sma[window:], price_array[i:i-window])
    sma = sma / window
    return sma[:, np.newaxis]

def create_ema(price_array, sma, window):
    """Exponential Moving Average

    Args:
        price_array (np.array): closing price most of the case
        sma (np.array): associated simple moving average
        window (int): number of periods back in time 

    Returns:
        ema (np.array): EMA
    """    
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
    """Momentum

    Args:
        price_array (np.array): closing price most of the case
        window (int): number of periods back in time 

    Returns:
        mom (np.array): Momentum
    """    
    mom =  np.zeros(len(price_array))
    for i in range(window, len(price_array)):
        mom[i] = price_array[i] - price_array[i-window]
    return mom

def create_bollband(high_array, low_array, close_array, window = 20, nb_std_dev = 2):
    """Bollinger Band up and down as defined in Investopedia

    Args:
        high_array (np.array): High
        low_array (np.array): Low
        close_array (np.array): Close
        window (int, optional): periods back in time. Defaults to 20.
        nb_std_dev (int, optional): std dev back in time. Defaults to 2.

    Returns:
        bold_array (np.array), bolu_array (np.array): Down and Up
    """    
    typical_array = create_typical_price(high_array, low_array, close_array)
    std_deviation = create_std_deviation(typical_array, window)[:,0] * nb_std_dev
    ma_array = create_sma(typical_array, window)[:,0]
    bolu_array = np.zeros(len(high_array))
    bolu_array[window:] = np.add(ma_array[window:], std_deviation[window:])
    bold_array = np.zeros(len(high_array))
    bold_array[window:] = np.subtract(ma_array[window:], std_deviation[window:])
    return bold_array[:, np.newaxis], bolu_array[:, np.newaxis]

def create_macd(price_array, window = [12, 26]):
    """Moving Average Convergence Divergence

    Args:
        price_array (np.array): typical price most of the case
        window (list, optional): periods used for Moving Averages. Defaults to [12, 26].

    Returns:
        macd (np.array): MACD
    """    
    sma_12 = create_sma(price_array, window[0])[:,0]
    sma_26 = create_sma(price_array, window[1])[:,0]
    ema_12 = create_ema(price_array, sma_12, window[0])[:,0]
    ema_26 = create_ema(price_array, sma_26, window[1])[:,0]
    diff_ema = ema_12 - ema_26
    sma_9 = create_sma(diff_ema, 9)[:,0]
    v = create_ema(diff_ema, sma_9, 9)[:,0]
    macd = np.subtract(diff_ema, v)[:, np.newaxis]
    return macd

def create_return(price_array, window):
    """Relative variation of price between n+1 and n

    Args:
        price_array (np.array): closing price most of the case
        window (int): periods back in time

    Returns:
        output (np.array): Relative variation
    """    
    output = np.zeros(len(price_array))
    for i in range(window, len(price_array)):
        output[i] = float(price_array[i+1] - price_array[i+1-window])/float(price_array[i+1-window])
        if i+2 == len(price_array): break
    return output

def create_up_down(price_array, window):
    """Consecutive Price Trends

    Args:
        price_array (np.array): closing price most of the case
        window (int): periods back in time

    Returns:
        pastUD (np.array): Consecutive Price Trends
    """    
    pastUD = np.zeros(len(price_array))
    for i in range(window+1, len(price_array)):
        pastUD[i] = window - 2*np.sum(price_array[i-window:i] < price_array[i-window-1:i-1])
    return pastUD

def create_period_since_cross(cross_array):
    """Period Since Cross

    Args:
        cross_array (np.array): Price-MA cross

    Returns:
        period_since_cross (np.array): Period Since MA and price cross 
    """    
    period_since_cross = np.zeros(len(cross_array))
    num = 0
    for i in range(len(cross_array)):
        if cross_array[i] == 0:
            num += 1
        else:
            num = 0
        period_since_cross[i] = num
    return period_since_cross

def create_macd_cross(macd):
    """MACD-y=0 cross

    Args:
        macd (np.array): MACD array

    Returns:
        macd_cross (np.array): MACD cross 
    """    
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
    """Moving Average Cross

    Args:
        ma (np.array): moving average
        price_array (np.array): closing price most of the time

    Returns:
        ma_cross (np.array): MA-price cross
    """    
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
    """Relative Strength Index

    Args:
        series (np.array): price array
        period (int, optional): periods back in time. Defaults to 14.

    Returns:
        rsi (np.array): RSI 
    """    
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
    rsi = 100 - 100 / (1 + rs)
    return rsi

def create_sto(close, low, high, n = 14):
    """Stochastic Oscillator

    Args:
        close (np.array): Close
        low ([type]): Low
        high ([type]): High
        n (int, optional): periods back in time. Defaults to 14.

    Returns:
        STOK (np.array), STOD (np.array): Slow and fast stochastics
    """    
    STOK = ((close - pd.rolling_min(low, n)) / (pd.rolling_max(high, n) - pd.rolling_min(low, n))) * 100
    STOD = pd.rolling_mean(STOK, 3)
    return STOK[:, np.newaxis], STOD[:, np.newaxis]

def normalize(array):
    """Normalization is necessary when features are in different scales

    Args:
        array (np.array): preprocessed data 

    Returns:
        array (np.array): normalized data 
    """    
    for i in range(2,32):
        mean = np.mean(array[:,i])
        std = np.std(array[:,i])
        sub = np.subtract(array[:,i], mean)
        array[:,i] = np.divide(sub, std, out=np.zeros_like(sub), where=std!=0)
    # Below is a naive implementation for normalization, returns the same results in much more time
    # output = np.std(array)
    # for i in range(array.shape[1]):
    #     mean = np.mean(array[:,i])
    #     print(mean)
    #     std = np.std(array[:,i])
    #     output[:,i] = np.divide(np.subtract(array[:,i], mean), std)
    #     array = output
    return array

def preprocess(data):
    """Preprocessing function that synthesizes all necessary features

    Args:
        data (pd.dataframe): Raw OCHL-Volume data

    Returns:
        data_array (np.array): Processed features based on trading indicators
    """    
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
# ----------------------------------------------------------------- Test
def test():
    """This function is internal to INDICATORS.py, it is meant for debugging but also serves as unit test
    """
    print("----- TEST FOR INDICATORS.PY -----")
    read_col = ["Date and Time", "Date", "Time", "Open", "High", "Low", "Close", "Volume", "Up Ticks", "Down Ticks"]
    data = pd.read_csv('../Data/Output/Production/AAPL.csv', usecols=read_col)
    cols = data.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    data = data[cols]
    data["Date"] = pd.DatetimeIndex(data["Date and Time"]).dayofweek
    print("----- PREPROCESS ------")
    preprocessedData = preprocess(data)
    print(preprocessedData)
    print("----- NORMALIZE ------")
    features = ["Date", "Time", "Open", "High", "Low", "Close", "Volume", "Up Ticks", "Down Ticks", "SMA-5", "SMA-10", "SMA-15", "SMA-20", "SMA-50", "SMA-100", "SMA-200", "EMA-5", "EMA-10", "EMA-15", "EMA-20", "EMA-50", "EMA-100", "EMA-200", "BOLU-20", "BOLD-20", "MACD", "SD-5", "SD-10", "SD-15", "SD-20", "SD-50", "SD-100", "SD-200", "SMAC-5", "SMAC-10", "SMAC-15", "SMAC-20", "SMAC-50", "SMAC-100", "SMAC-200", "EMAC-5", "EMAC-10", "EMAC-15", "EMAC-20", "EMAC-50", "EMAC-100", "EMAC-200", "MACDC"]
    dataframe = pd.DataFrame(data=preprocessedData[:,1:],    # values
                            index=preprocessedData[:,0],    # 1st column as index
                            columns=features)
    dataframe.index = pd.to_datetime(dataframe.index)
    normalizedData = normalize(dataframe.to_numpy())
    print(normalizedData)
    print("----------------------------------")