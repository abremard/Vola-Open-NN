import plotly.graph_objects as go

import pandas as pd
import time

import dask.dataframe as dd # Dask is meant for performance impovement, not implemented yet

import glob
import os

stockNames = ["AMZN","AAPL","ABT","ACN","AMGN","ADBE","BA","CCEP","CMCSA","CSCO","CVX","DOW","FB","HD","INTC","JNJ","MO","NFLX"]

tickURL = "../Data/Input/Tick/"

def division(n, d):
    return n / d if d else 0

nbDays = 0
totalProfit = 0
totalPositions = 0
totalBuyPositions = totalSellPositions = 0
totalBuyLoss = totalBuyWin = 0
totalBuyWinProfit = totalBuyLossProfit = 0
totalSellLoss = totalSellWin = 0
totalSellWinProfit = totalSellLossProfit = 0

benchmarkDict = {}

for stock in stockNames:
    path = tickURL + stock + "/*.csv"
    for fname in glob.glob(path):
        currentDay = os.path.basename(fname).split(".")[0]
        data = pd.read_csv(fname, usecols=["Date and Time", "Open", "High", "Low", "Close"], index_col="Date and Time")
        data.index.names = ["Date"]
        data.columns = ["Open", "High", "Low", "Close"]
        trueSize = len(data)

        # Setting data
        data.index = pd.to_datetime(data.index)

        meanVolatility = 0

        day = data

        firstHourDF = day.groupby(['Date']).agg({'Open': "first", 'High': "max", 'Low': "min",'Close': "last"}).head(60)
        minimum = firstHourDF["Low"].min()
        maximum = firstHourDF["High"].max()
        volatility = maximum - minimum
        renko = volatility * 0.05
        # print(renko)
        # print(minimum, maximum, volatility)
        # meanVolatility = meanVolatility + volatility
        nbDays = nbDays + 1
        day = day.between_time('10:31', '16:00')
        if not(day.empty):
            startingPoint = day["Close"].iloc[0]
            id = 0
            WLDataframe = pd.DataFrame({"Date":[currentDay], "Open":[startingPoint - renko/3], "Close":[startingPoint + renko], "High":[startingPoint + renko], "Low":[startingPoint - renko/3]})
            tmpDict = {}
            highTarget = day["Close"].iloc[0] + 2*renko
            lowTarget = day["Close"].iloc[0] - renko
            # print("renko", renko, "highTarget", highTarget, "lowTarget", lowTarget)
            for row in day.itertuples():
                close = row.Close
                if close > highTarget:
                    id = id + 1
                    tmpDict[id] = {"Date":row.Index, "Open":highTarget - renko*4/3, "Close":highTarget, "High":highTarget, "Low":highTarget - renko*4/3}
                    highTarget = highTarget + renko
                    lowTarget = lowTarget + renko
                elif close < lowTarget:
                    id = id + 1
                    tmpDict[id] = {"Date":row.Index, "Open":lowTarget + renko*4/3, "Close":lowTarget, "High":lowTarget + renko*4/3, "Low":lowTarget}
                    highTarget = highTarget - renko
                    lowTarget = lowTarget - renko
            tmpDF = pd.DataFrame.from_dict(tmpDict, "index")
            WLDataframe = WLDataframe.append(tmpDF)

            ovRangeLow = WLDataframe['Low'].iloc[0]
            ovRangeHigh = WLDataframe['High'].iloc[0]

            for i in range(1,4):
                if WLDataframe['High'].iloc[i] > ovRangeHigh:
                    ovRangeHigh = WLDataframe['High'].iloc[i]
                if WLDataframe['Low'].iloc[i] < ovRangeLow:
                    ovRangeLow = WLDataframe['Low'].iloc[i]

            firstTargetHigh = ovRangeHigh + volatility * 0.3
            firstTargetLow = ovRangeLow - volatility * 0.3
            secondTargetHigh = ovRangeHigh + volatility * 1.9
            secondTargetLow = ovRangeLow - volatility * 1.9

            nRows, nCols = WLDataframe.shape

            entryHigh = entryLow = None

            countSellPosition = 0
            countBuyPosition = 0

            trend = True

            fig = go.Figure(data=[go.Candlestick(x=WLDataframe.index,
            open=WLDataframe["Open"],
            high=WLDataframe["High"],
            low=WLDataframe["Low"],
            close=WLDataframe["Close"])], layout=dict(xaxis=dict(title='Date', tickmode='array', tickvals=WLDataframe.index, ticktext=WLDataframe["Date"], type='category')))

            fig.update_layout(
                xaxis_rangeslider_visible=False,
                yaxis_title= stock + " Stock",
                shapes = [
                    dict(type="line", x0=0, x1=id, y0=ovRangeHigh, y1=ovRangeHigh, line=dict(color="RoyalBlue",width=1)),
                    dict(type="line", x0=0, x1=id, y0=ovRangeLow, y1=ovRangeLow, line=dict(color="RoyalBlue",width=1)),
                    dict(type="line", x0=0, x1=id, y0=firstTargetHigh, y1=firstTargetHigh, line=dict(color="DarkOrange",width=1)),
                    dict(type="line", x0=0, x1=id, y0=firstTargetLow, y1=firstTargetLow, line=dict(color="DarkOrange",width=1)),
                    dict(type="line", x0=0, x1=id, y0=secondTargetHigh, y1=secondTargetHigh, line=dict(color="LightSeaGreen",width=1)),
                    dict(type="line", x0=0, x1=id, y0=secondTargetLow, y1=secondTargetLow, line=dict(color="LightSeaGreen",width=1)),
                    ],
                annotations=[]
            )

            nbPosition = 0
            nbBuyPosition = nbSellPosition = 0
            nbBuyWin = nbBuyLoss = 0
            buyWinProfit = buyLossProfit = 0
            sellWinProfit = sellLossProfit = 0
            nbSellWin = nbSellLoss = 0
            previousLow = WLDataframe['Low'].iloc[0]
            previousHigh = WLDataframe['High'].iloc[0]

            positions = []

            profit = 0

            for row in WLDataframe.itertuples():    

                if countBuyPosition > 0 or countSellPosition > 0:

                    for pos in positions:

                        if pos['inPosition'] == True:
                                
                            if pos['order'] == 'buy':
                                # Stop-loss triggered
                                if row.Low < pos['stopLoss'] or row.Index == len(WLDataframe) - 1:
                                    fig.add_shape(dict(type="rect", x0=row.Index, x1=row.Index+1, yref="paper", y0=0, y1=1, fillcolor="Red", opacity=0.3, line_width=0))
                                    pos['exitPrice'] = pos['stopLoss']
                                    if pos['firstObjectiveReached']:
                                        pos['profit'] = pos['profit'] + (pos['exitPrice'] - pos['entryPrice']) * pos['sizing'] * 2 / 3
                                    else:
                                        pos['profit'] = pos['profit'] + (pos['exitPrice'] - pos['entryPrice']) * pos['sizing']
                                    profit = profit + pos['profit']
                                    if pos['profit'] >= 0:
                                        nbBuyWin = nbBuyWin + 1
                                        buyWinProfit = buyWinProfit + pos['profit']
                                    else: 
                                        nbBuyLoss = nbBuyLoss + 1
                                        buyLossProfit = buyLossProfit + pos['profit']
                                    pos['inPosition'] = False
                                # First objective reached triggered
                                if not(pos['firstObjectiveReached']):
                                    if row.High > firstTargetHigh:
                                        pos['stopLoss'] = ovRangeHigh
                                        fig.add_shape(dict(type="rect", x0=row.Index, x1=row.Index+1, yref="paper", y0=0, y1=1, fillcolor="Yellow", opacity=0.3, line_width=0))
                                        pos['firstObjectiveReached'] = True
                                        pos['profit'] = pos['profit'] + (firstTargetHigh - pos['entryPrice']) * pos['sizing'] / 3
                                        profit = profit + pos['profit']
                                # Second objective reached triggered
                                if not(pos['secondObjectiveReached']):
                                    if row.High > secondTargetHigh:
                                        fig.add_shape(dict(type="rect", x0=row.Index, x1=row.Index+1, yref="paper", y0=0, y1=1, fillcolor="Blue", opacity=0.3, line_width=0))
                                        pos['secondObjectiveReached'] = True
                                        pos['profit'] = pos['profit'] + (firstTargetHigh - pos['entryPrice']) * pos['sizing'] * 2 / 3
                                        profit = profit + pos['profit']
                                        pos['inPosition'] = False
                                # Price is over first objective
                                elif row.High > firstTargetHigh :
                                    if trend:
                                        if row.High > previousHigh:
                                            top = row.High
                                        else:
                                            bottom = top
                                            trend = False
                                    else:
                                        if row.Low < bottom:
                                            bottom = row.Low
                                        if row.High > top:
                                            trend = True
                                            pos['stopLoss'] = bottom
                                            fig.add_shape(dict(type="rect", x0=row.Index, x1=row.Index+1, yref="paper", y0=0, y1=1, fillcolor="LightSalmon", opacity=0.3, line_width=0))

                            elif pos['order'] == 'sell':
                                # Stop-loss triggered
                                if row.Low > pos['stopLoss'] or row.Index == len(WLDataframe) - 1:
                                    fig.add_shape(dict(type="rect", x0=row.Index, x1=row.Index+1, yref="paper", y0=0, y1=1, fillcolor="Red", opacity=0.3, line_width=0))
                                    pos['exitPrice'] = pos['stopLoss']
                                    if pos['firstObjectiveReached']:
                                        pos['profit'] = pos['profit'] + (pos['entryPrice'] - pos['exitPrice']) * pos['sizing'] * 2 / 3
                                    else:
                                        pos['profit'] = pos['profit'] + (pos['entryPrice'] - pos['exitPrice']) * pos['sizing']
                                    profit = profit + pos['profit']
                                    if pos['profit'] >= 0:
                                        nbSellWin = nbSellWin + 1
                                        sellWinProfit = sellWinProfit + pos['profit']
                                    else: 
                                        nbSellLoss = nbSellLoss + 1
                                        sellLossProfit = sellLossProfit + pos['profit']
                                    pos['inPosition'] = False
                                # First objective reached triggered
                                if not(pos['firstObjectiveReached']):
                                    if row.Low < firstTargetLow:
                                        pos['stopLoss'] = ovRangeLow
                                        fig.add_shape(dict(type="rect", x0=row.Index, x1=row.Index+1, yref="paper", y0=0, y1=1, fillcolor="Yellow", opacity=0.3, line_width=0))
                                        pos['firstObjectiveReached'] = True
                                        pos['profit'] = pos['profit'] + (pos['entryPrice'] - firstTargetLow) * pos['sizing'] / 3
                                        profit = profit + pos['profit']
                                # Second objective reached triggered
                                if not(pos['secondObjectiveReached']):
                                    if row.Low < secondTargetLow:
                                        fig.add_shape(dict(type="rect", x0=row.Index, x1=row.Index+1, yref="paper", y0=0, y1=1, fillcolor="Blue", opacity=0.3, line_width=0))
                                        pos['secondObjectiveReached'] = True
                                        pos['profit'] = pos['profit'] + (pos['entryPrice'] - secondTargetLow) * pos['sizing'] * 2 / 3
                                        profit = profit + pos['profit']
                                        pos['inPosition'] = False
                                # Price is below first objective
                                elif row.Low < firstTargetLow :
                                    if trend:
                                        if row.Low < previousLow:
                                            bottom = row.Low
                                        else:
                                            top = bottom
                                            trend = False
                                    else:
                                        if row.High > top:
                                            top = row.High
                                        if row.Low < bottom:
                                            trend = True
                                            pos['stopLoss'] = top
                                            fig.add_shape(dict(type="rect", x0=row.Index, x1=row.Index+1, yref="paper", y0=0, y1=1, fillcolor="LightSalmon", opacity=0.3, line_width=0))

                if countBuyPosition < 2 :
                    # enter buy position
                    if (row.Close > ovRangeHigh) and (previousHigh <= ovRangeHigh) :
                        fig.add_shape(dict(type="rect", x0=row.Index, x1=row.Index+1, yref="paper", y0=0, y1=1, fillcolor="LightSeaGreen", opacity=0.3, line_width=0))
                        tmpSL = firstTargetLow + 0.1
                        tmpRisk = abs(row.High - tmpSL)
                        positions.append({
                            'entryPrice': row.High,
                            'stopLoss': tmpSL,
                            'risk': tmpRisk,
                            'sizing': 100 / tmpRisk,
                            'order': 'buy',
                            'profit': 0,
                            'exitPrice': None,
                            'inPosition': True,
                            'firstObjectiveReached': False,
                            'secondObjectiveReached': False
                        })
                        bottom = row.Low
                        top = row.High
                        countBuyPosition += 1
                        nbBuyPosition = nbBuyPosition + 1
                        nbPosition = nbPosition + 1
                if countSellPosition < 2 :
                    # enter sell position
                    if (row.Close < ovRangeLow) and (previousLow >= ovRangeLow) :
                        fig.add_shape(dict(type="rect", x0=row.Index, x1=row.Index+1, yref="paper", y0=0, y1=1, fillcolor="LightSeaGreen", opacity=0.3, line_width=0))
                        tmpSL = firstTargetHigh - 0.1
                        tmpRisk = abs(row.Low - tmpSL)
                        positions.append({
                            'entryPrice': row.Low,
                            'stopLoss': tmpSL,
                            'risk': tmpRisk,
                            'sizing': 100 / tmpRisk,
                            'order': 'sell',
                            'profit': 0,
                            'exitPrice': None,
                            'inPosition': True,
                            'firstObjectiveReached': False,
                            'secondObjectiveReached': False
                        })
                        bottom = row.Low
                        top = row.High
                        countSellPosition += 1
                        nbSellPosition = nbSellPosition + 1
                        nbPosition = nbPosition + 1

                previousHigh = row.High
                previousLow = row.Low

            # A corriger... mais pas urgent
            print(nbDays, '. riskRange', ovRangeHigh - firstTargetLow, 'risk', pos['risk'], 'sizing', pos['sizing'], 'profit', profit, 'nbPosition', nbPosition)
            totalProfit = totalProfit + profit
            totalPositions = totalPositions + nbPosition
            totalSellPositions = totalSellPositions + nbSellPosition
            totalSellWin = totalSellWin + nbSellWin
            totalSellWinProfit = totalSellWinProfit + sellWinProfit
            totalSellLoss = totalSellLoss + nbSellLoss
            totalSellLossProfit = totalSellLossProfit + sellLossProfit
            totalBuyPositions = totalBuyPositions + nbBuyPosition
            totalBuyWin = totalBuyWin + nbBuyWin
            totalBuyWinProfit = totalBuyWinProfit + buyWinProfit
            totalBuyLoss = totalBuyLoss + nbBuyLoss
            totalBuyLossProfit = totalBuyLossProfit + buyLossProfit

        benchmarkDict[nbDays] = {
            "Symbol":stock,
            "Date":currentDay,
            "Sizing":str(pos['sizing']),
            "Profit":str(profit),
            "Sell Positions":str(nbSellPosition),
            "Sell Win":str(nbSellWin),
            "Sell Loss":str(nbSellLoss),
            "Buy Positions":str(nbBuyPosition),
            "Buy Win":str(nbBuyWin),
            "Buy Loss":str(nbBuyLoss),
            "Score":nbBuyWin+nbSellWin-nbBuyLoss-nbSellLoss,
            "Positions":str(nbPosition)
            }

        fig.update_layout(
            title="Vola Open Strategy --- " + currentDay + " --- sizing : " + str(pos['sizing']) + " --- profit : " + str(profit) + " --- positions : " + str(nbPosition),
        )
        # fig.write_html("../Data/Output/WL/HTML/" + str(nbDays) + ".html", auto_open=False)

benchmarkDict['Total'] = {
    "Sell Positions":str(totalSellPositions),
    "Sell Win":str(totalSellWin),
    "Sell Loss":str(totalSellLoss),
    "Buy Positions":str(totalBuyPositions),
    "Buy Win":str(totalBuyWin),
    "Buy Loss":str(totalBuyLoss),
    "Positions":str(totalPositions)
    }

benchmarkDict['Sum'] = {
    "Profit":str(totalProfit),
    "Sell Positions":str(totalSellLossProfit + totalSellWinProfit),
    "Sell Win":str(totalSellWinProfit),
    "Sell Loss":str(totalSellLossProfit),
    "Buy Positions":str(totalBuyLossProfit + totalBuyWinProfit),
    "Buy Win":str(totalBuyWinProfit),
    "Buy Loss":str(totalBuyLossProfit),
    }

benchmarkDict['Average'] = {
    "Profit":str(division(totalProfit, totalPositions)),
    "Sell Positions":str(division((totalSellLossProfit + totalSellWinProfit), totalSellPositions)),
    "Sell Win":str(division(totalSellWinProfit, totalSellWin)),
    "Sell Loss":str(division(totalSellLossProfit, totalSellLoss)),
    "Buy Positions":str(division((totalBuyLossProfit + totalBuyWinProfit), totalBuyPositions)),
    "Buy Win":str(division(totalBuyWinProfit, totalBuyWin)),
    "Buy Loss":str(division(totalBuyLossProfit, totalBuyLoss)),
    }

benchmarkDF = pd.DataFrame(benchmarkDict)

benchmarkDF.T.to_csv('../Data/Output/WL/Benchmark/benchmark-test.csv')

print("totalProfit", totalProfit)
