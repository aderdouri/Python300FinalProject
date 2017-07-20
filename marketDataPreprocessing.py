"""
Date: Monday, 24 July 2017
File name: 
Version: 1 
Author: Abderrazak DERDOURI
Subject: CQF Final Project

Description: 

Notes:
Revision History:
"""

import LiteLibrary
import capletVolatilityStripping
import pandas as pd
import numpy as np
from datetime import date
import re
#pip install xlrd
from sys import exit
from collections import OrderedDict


from scipy.optimize import root, fsolve
import matplotlib.pyplot as plt
import xlsxwriter

def load_data():    
    marketData = pd.read_excel('MarketData.xls', sheetname='MarketData', skiprows=1)
    marketData.columns = ['TenorTi', 'Date', 'DiscountFactor', 'CapVolatility']
    return marketData

def updateDate(df):
    for index, row in df.iterrows():
        if (pd.isnull(row['Date'])):
            year, period = row['TenorTi'].split('Y')
            if (period in ['3M','6M','9M']):
                prevDate = df[df['TenorTi']=='T'+year]['Date']
                Date = LiteLibrary.getNextDate( pd.datetime(prevDate.dt.year, prevDate.dt.month, prevDate.dt.day), period)
                df.loc[index, 'Date'] = Date
            else:
                year, period = row['TenorTi'].split('Y')
                prevDate = df[df['TenorTi']=='T'+ str(int(year)-1)+'Y']['Date']
                Date = LiteLibrary.getNextDate( pd.datetime(prevDate.dt.year, prevDate.dt.month, prevDate.dt.day), period)
                df.loc[index, 'Date'] = Date
    return df


def plot(df):
    tenor = df['TenorTi']
    CapVolatility    = df['CapVolatility']
    CapletVolatility    = df['CapletVolatility']
    plt.plot(tenor, CapVolatility)
    plt.show()
        
def preProcessMarketData():
    marketData = load_data()

    data = OrderedDict()
    tenorList = ['t0','T0','TSN','TSW','T2W'
                 ,'T1M','T2M','T3M','T6M','T9M'
                 ,'T1Y','T1Y3M','T1Y6M','T1Y9M'
                 ,'T2Y','T2Y3M','T2Y6M','T2Y9M'
                 ,'T3Y','T3Y3M','T3Y6M','T3Y9M'
                 ,'T4Y','T4Y3M','T4Y6M','T4Y9M'
                 ,'T5Y','T5Y3M','T5Y6M','T5Y9M'
                 ,'T6Y','T6Y3M','T6Y6M','T6Y9M'
                 ,'T7Y','T7Y3M','T7Y6M','T7Y9M'
                 ,'T8Y','T8Y3M','T8Y6M','T8Y9M'
                 ,'T9Y','T9Y3M','T9Y6M','T9Y9M'
                 ,'T10Y','T10Y3M','T10Y6M','T10Y9M'
                 ,'T11Y','T11Y3M','T11Y6M','T11Y9M'
                 ,'T12Y','T12Y3M','T12Y6M','T12Y9M'
                 ,'T13Y','T13Y3M','T13Y6M','T13Y9M'
                 ,'T14Y','T14Y3M','T14Y6M','T14Y9M'
                 ,'T15Y','T15Y3M','T15Y6M','T15Y9M'
                 ,'T16Y','T16Y3M','T16Y6M','T16Y9M'
                 ,'T17Y','T17Y3M','T17Y6M','T17Y9M'
                 ,'T18Y','T18Y3M','T18Y6M','T18Y9M'
                 ,'T19Y','T19Y3M','T19Y6M','T19Y9M'
                 ,'T20Y'
                 ]

    size = len(tenorList)
    data = OrderedDict()

    data['TenorTi'] = tenorList
    data['Date'] = size*[None]
    data['DiscountFactor'] = size*[None]
    data['CapVolatility'] = size*[None]
   
    df = pd.DataFrame(data)
    df = pd.merge(df, marketData, how='outer', on='TenorTi', suffixes=('_1', '_2'))

    columns_to_keep = ['TenorTi', 'Date', 'DiscountFactor', 'CapVolatility']
    df.rename(columns={'Date_2': 'Date', 
                             'DiscountFactor_2': 'DiscountFactor', 
                             'CapVolatility_2': 'CapVolatility'}, inplace=True)

    df = df[columns_to_keep]
    s = df['Date']
    valid_indx = s.index.get_indexer(s.index[~s.isnull()])
    nan_indx = s.index.get_indexer(s.index[s.isnull()])

    T1Y = df[df['TenorTi']=='T1Y']['Date']

    for idx in nan_indx:
        res = re.split(';|Y|M|T', df['TenorTi'].iloc[idx])
        year = int(res[1])
        month = (int(res[2]) if len(res[2])>0 else 0)
        currentDate = pd.datetime(T1Y.dt.year + year-1, T1Y.dt.month + month, T1Y.dt.day)
        if (5==currentDate.weekday()):
            day += 2
        elif (6==currentDate.weekday()):
            day += 1

        year = currentDate.year
        month = currentDate.month
        day = currentDate.day
        df['Date'].iloc[idx] = pd.datetime(year, month, day)


    s = df['DiscountFactor']
    valid_indx = s.index.get_indexer(s.index[~s.isnull()])
    nan_indx = s.index.get_indexer(s.index[s.isnull()])

    df['DF'] = df['DiscountFactor']

    for idx in nan_indx:
        prev = np.where(valid_indx<idx)
        next = np.where(valid_indx>idx)
        valid_prev_idx = valid_indx[prev[0][-1]]
        valid_next_idx = valid_indx[next[0][0]]

        prevDF = df['DF'].iloc[valid_prev_idx]
        nextDF = df['DF'].iloc[valid_next_idx]

        currentDate = pd.to_datetime(df['Date'].iloc[idx])
        prevDate = pd.to_datetime(df['Date'].iloc[valid_prev_idx])
        nextDate = pd.to_datetime(df['Date'].iloc[valid_next_idx])

        df['DF'].iloc[idx] = LiteLibrary.interpolateDF(prevDF, nextDF,  currentDate, prevDate, nextDate)  

    DF0 = df['DF'].iloc[1]    
    df['DF'] = df.apply(lambda row: row['DF']/DF0, axis=1)        
    df['PrevDF'] = df['DF'].shift(1)        

    df['PrevDate'] = df['Date'].shift(1)
    df['Delta'] = df.apply(LiteLibrary.date_diff, axis=1)
    dateT0 = pd.to_datetime('25-01-2005')
    
    df['YearFraction*DF'] = df.apply(lambda row: row['DF']*row['Delta'], axis=1)        

    df['LT0Ti-3MTi'] = df.apply(lambda row: (row['PrevDF']/row['DF']-1)/row['Delta'], axis=1)       
    df['delta*L^2'] = df.apply(lambda row: row['Delta']*row['LT0Ti-3MTi']*row['LT0Ti-3MTi'], axis=1) 


    T6M = df[df['TenorTi']=='T6M']['YearFraction*DF']
    start_idx = T6M.index[0]
    max_idx = len(df.index)
    df['CumulativeSum'] = np.NaN
   
    for idx in range(start_idx, max_idx):
        myrange = df['YearFraction*DF'].iloc[start_idx:idx+1] 
        df.loc[idx, 'CumulativeSum'] = myrange.sum()


    df['DifferenceDF'] = np.NaN
    DF_T3M_idx = df[df['TenorTi']=='T3M']['DF'].index[0]
    DF_T3M_value = df['DF'].iloc[DF_T3M_idx]


    df['DifferenceDF'] = df.apply(lambda row: DF_T3M_value-row['DF'], axis=1)        
    df['ForwardSwapRate'] = df.apply(lambda row: row['DifferenceDF']/row['CumulativeSum'], axis=1)        


    s = df['CapVolatility']
    valid_indx = s.index.get_indexer(s.index[~s.isnull()])
    nan_indx = s.index.get_indexer(s.index[s.isnull()])
    nan_indx = [x for x in nan_indx if x>10]

    df['CapVolatility'].iloc[8] = df['CapVolatility'].iloc[10]
    df['CapVolatility'].iloc[9] = df['CapVolatility'].iloc[10]

    for idx in nan_indx:
        prev = np.where(valid_indx<idx)
        next = np.where(valid_indx>idx)
        valid_prev_idx = valid_indx[prev[0][-1]]
        valid_next_idx = valid_indx[next[0][0]]

        prevVol = df['CapVolatility'].iloc[valid_prev_idx]
        nextVol = df['CapVolatility'].iloc[valid_next_idx]

        currentDate = pd.to_datetime(df['Date'].iloc[idx])
        prevDate = pd.to_datetime(df['Date'].iloc[valid_prev_idx])
        nextDate = pd.to_datetime(df['Date'].iloc[valid_next_idx])

        df['CapVolatility'].iloc[idx] = LiteLibrary.interpolateVol(prevVol, nextVol,  currentDate, prevDate, nextDate)  

    
    df['CapletVolatility'] = np.NaN
    
    DF_T6M_idx = df[df['TenorTi']=='T6M'].index[0]    
    DF_T6M_value = df['CapVolatility'].iloc[DF_T6M_idx]
    df['CapletVolatility'].iloc[DF_T6M_idx] = DF_T6M_value
    
    s = df[df['TenorTi']=='T9M']
    df['DeltaT0Ti'] = df.apply(lambda row: LiteLibrary.year_fraction(dateT0, row['Date']), axis=1)
    df['DeltaT0Ti_365'] = df.apply(lambda row: LiteLibrary.year_fraction365(dateT0, row['Date']), axis=1)


    df = capletVolatilityStripping.resolveForCapletVolatility(df)

    df['SigmaCaplet^2*TimeToMaturity'] = df.apply(lambda row: np.power(row['CapletVolatility'], 2)*row['DeltaT0Ti'] , axis=1)

    return df

if __name__ == '__main__':
    df = preProcessMarketData()
    LiteLibrary.writeDataFrame(df)

    tenor = df['TenorTi']
    CapVolatility    = df['CapVolatility']
    CapletVolatility    = df['CapletVolatility']
    plt.plot(CapVolatility)
    plt.plot(CapletVolatility)
    plt.show()
    
