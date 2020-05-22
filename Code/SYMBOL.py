import os

base = 'C:/Users/Alexandre Bremard/DNN-trading/Vola-Open/Production'
timeframes = ['5min', '10min', '30min', '60min', '120min', '240min', '390min']
ref = ['24|AAPL[20500101],62|ABT[20500101],114|ADBE[20500101],368|AMGN[20500101],698|BA[20500101],1332|CCEP[20500101],1637|CMCSA[20500101],1900|CSCO[20500101],3496|HD[20500101],3951|INTC[20500101],4151|JNJ[20500101],4954|MO[20500101],16841|AMZN[20500101],23112|CVX[20500101],23709|NFLX[20500101],25555|ACN[20500101],42950|FB[20500101],52991|DOWWI[20500101]']
numbers = ['24','62','114','368','698','1332','1637','1900','3496','3951','4151','4954','16841','23112','23709','25555','42950','52991']
symbols = ['AAPL','ABT','ADBE','AMGN','BA','CCEP','CMCSA','CSCO','HD','INTC','JNJ','MO','AMZN','CVX','NFLX','ACN','FB','DOW']

# for time in timeframes:
#     for i in range(len(numbers)):
#         old_url = base + time + '/' + numbers[i] + '.csv'
#         new_url = base + time + '/' + symbols[i] + '.csv'
#         os.rename(old_url, new_url)

def rename():
    for i in range(len(numbers)):
        old_url = base + '/' + numbers[i] + '.csv'
        new_url = base + '/' + symbols[i] + '.csv'
        os.rename(old_url, new_url)