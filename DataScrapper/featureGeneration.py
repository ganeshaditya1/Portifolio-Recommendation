import numpy as np
import talib
from talib.abstract import *
data = np.genfromtxt('AAPL2.txt', delimiter=',')
data = np.flipud(data);

inputs = {'open' : data[:, 0], 'high': data[:, 1], 'low': data[:, 2], 'close': data[:, 3], 'volume': data[:, 4]}

TodayCloseMinusPreviousClose = np.transpose([np.append([0], data[:5018, 3]) - data[:, 3]])

PreviousClosePrice = np.transpose([np.append([0], data[:5018, 3])])
PreviousLowPrice = np.transpose([np.append([0], data[:5018, 2])])
PreviousHighPrice = np.transpose([np.append([0], data[:5018, 1])])
PreviousOpenPrice = np.transpose([np.append([0], data[:5018, 0])])

FiveDaySMAClosingPrices = np.transpose([SMA({'close': data[:, 3]}, timeperiod = 5)])
SixDaySMAClosingPrices = np.transpose([SMA({'close': data[:, 3]}, timeperiod = 6)])
TenDaySMAClosingPrices = np.transpose([SMA({'close': data[:, 3]}, timeperiod = 10)])
TwentyDaySMAClosingPrices = np.transpose([SMA({'close': data[:, 3]}, timeperiod = 20)])


TwentyDayEMAClosingPrices = np.transpose([EMA({'close': data[:, 3]}, timeperiod = 20)])
TenDayEMAClosingPrices = np.transpose([EMA({'close': data[:, 3]}, timeperiod = 10)])
SixDayEMAClosingPrices = np.transpose([EMA({'close': data[:, 3]}, timeperiod = 6)])
fiveDayEMAClosingPrices = np.transpose([EMA({'close': data[:, 3]}, timeperiod = 5)])

TwentyDayTRIMAClosingPrices = np.transpose([TRIMA({'close': data[:, 3]}, timeperiod = 20)])
TenDayTRIMAClosingPrices = np.transpose([TRIMA({'close': data[:, 3]}, timeperiod = 10)])
SixDayTRIMAClosingPrices = np.transpose([TRIMA({'close': data[:, 3]}, timeperiod = 6)])
fiveDayTRIMAClosingPrices = np.transpose([TRIMA({'close': data[:, 3]}, timeperiod = 5)])

AbsolutePriceOscillator = np.transpose([APO({'close': data[:, 3]})])

macd_data = MACD({'close': data[:, 3]})
macd_data = np.transpose([macd_data[0]])

MomentumOpeningPrice = np.transpose([MOM({'close': data[:, 0]})])
MomentumHighestPrice = np.transpose([MOM({'close': data[:, 1]})])
MomentumLowestPrice = np.transpose([MOM({'close': data[:, 2]})])
MomentumClosestPrice = np.transpose([MOM({'close': data[:, 3]})])


ChaikinVolatality = np.transpose([ADOSC(inputs)])

FastK, FastD = STOCHF(inputs, prices=['high', 'low', 'open'])
FastK = np.transpose([FastK])
FastD = np.transpose([FastD])
SlowK, SlowD = STOCH(inputs, prices=['high', 'low', 'open'])
SlowK = np.transpose([SlowK])
SlowD = np.transpose([SlowD])

Wr = np.transpose([WILLR(inputs)])

RelativeStrengthIndex = np.transpose([RSI(inputs)])

UpperBBand, MiddleBBand, LowerBBand = talib.BBANDS(data[:, 3], matype=talib.MA_Type.T3)
UpperBBand = np.transpose([UpperBBand])
MiddleBBand = np.transpose([MiddleBBand])
LowerBBand = np.transpose([LowerBBand])

priceRateOfChange = np.transpose([ROC(inputs)])

MedianPrice = np.transpose([MEDPRICE(inputs)])
TypicalPrice = np.transpose([TYPPRICE(inputs)])
WeightedClose = np.transpose([WCLPRICE(inputs)])

maxHigh = np.array([])
for i in range(0, 5019):
    maxHigh = np.append([maxHigh], [np.max(data[0:i + 1, 1])])
maxHigh = np.transpose([maxHigh])

minLow = np.array([])
for i in range(0, 5019):
    minLow = np.append([minLow], [np.min(data[0:i + 1, 2])])
minLow = np.transpose([minLow])

result = np.concatenate((TodayCloseMinusPreviousClose, PreviousClosePrice, PreviousHighPrice, PreviousLowPrice,
                        PreviousOpenPrice, FiveDaySMAClosingPrices, SixDaySMAClosingPrices, TenDaySMAClosingPrices,
                        TwentyDaySMAClosingPrices, fiveDayEMAClosingPrices, SixDayEMAClosingPrices,
                        TenDayEMAClosingPrices, TwentyDayEMAClosingPrices, fiveDayTRIMAClosingPrices,
                        SixDayTRIMAClosingPrices, TenDayTRIMAClosingPrices, TwentyDayTRIMAClosingPrices,
                        AbsolutePriceOscillator, macd_data, MomentumClosestPrice, MomentumHighestPrice,
                        MomentumLowestPrice, MomentumOpeningPrice, ChaikinVolatality, FastK, FastD, SlowK, SlowD, Wr,
                        RelativeStrengthIndex, UpperBBand, MiddleBBand, LowerBBand, priceRateOfChange, MedianPrice,
                        TypicalPrice, WeightedClose, maxHigh, minLow), axis = 1)

#print(np.shape(result))

