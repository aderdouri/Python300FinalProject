"""
Date: Monday, 24 July 2017
File name: capletVolatilityStripping.py
Version: 1 
Author: Abderrazak DERDOURI
Subject: CQF Final Project

Description: Perform caplet volatility stripping as decribed in paragraph (3.4.2)

Notes: AbderrazakDerdouriCQFFinalProject.pdf

Run: python -m unittest capletVolatilityStripping.CplVolStripTests.testCplVolStrip

requirement : must be run after  
              python -m unittest lmmCalibration.CapATMStrikeTests.testATMStrike
              to have lmmCalibration sheet already prepared
"""

import LiteLibrary
from scipy.optimize import root, fsolve
from unittest import TestCase
import numpy as np
import pandas as pd
import lmmCalibration
import matplotlib.pyplot as plt

def load_lmmCalibration():
    """
    Load preProcessed market data (sheet lmmCalibration from lmmData.xlsx file)
    The output saved to the CapletVolatility in the same lmmCalibration sheet 
    """
    lmmCalibration = pd.read_excel('lmmData.xlsx', sheetname='lmmCalibration', skiprows=0)
    return lmmCalibration

def plotStrippedCapletVolatilities(df):
    """
    Plot Stripped Caplet Volatilities
    """
    strippedCapletVolatilities = df[['CapVolatility', 'CapletVolatility']][8:] # start  from  T6M
    index = strippedCapletVolatilities.index
    myRange = 8+np.linspace(1, 82, 10)
    myRange[0] = 12
    myRange[9] = 86
    myRange = np.append(8, myRange)

    xticks = list(df['TenorTi'].iloc[myRange])        
    ax = strippedCapletVolatilities.plot(title='Cap and caplet stripped volatilities')
    fig = ax.get_figure()
    ax.set_xticks(myRange);
    ax.set_xticklabels(xticks, rotation=45)
    plt.savefig('StrippedCapletVolatilies')
    plt.show()        

def resolveForCapletVolatility(df):
    """
    Resolve for caplet volatility using algorith described in paragraph (3.4.2)
    """
    DF_T6M_idx = df[df['TenorTi']=='T6M'].index[0]    
    start_idx = DF_T6M_idx
    max_idx = len(df.index)

    for row_idx in range(start_idx +1, max_idx):
        capValue = 0.0
        capletValue  = 0.0
        for idx in range(start_idx, row_idx+1):
            DF_T0_Ti_1 = df['DF'].iloc[idx-1] # D(T0, Ti-1)
            DF_T0_Ti = df['DF'].iloc[idx]     # D(T0, Ti)
            Tau_i =  df['Delta'].iloc[idx]    # time(T3M, Ti)

            Strike = df['ForwardSwapRate'].iloc[idx] # S(T0, T3M, Ti)
            L_T0_Ti_1_Ti = LiteLibrary.liborRate(DF_T0_Ti_1, DF_T0_Ti, Tau_i) # L(Ti-1, Ti-1, Ti) = F(T0, Ti-1, Ti)
            Ti_1 = df['DeltaT0Ti'].iloc[idx]
            r = 0.0
            sigma_cap = df['CapVolatility'].iloc[row_idx]
            capValue = capValue + LiteLibrary.blackCapletValue(DF_T0_Ti, Tau_i, L_T0_Ti_1_Ti, Strike, Ti_1, sigma_cap)

            if (idx==row_idx):
                def func(x):
                    return ( capValue 
                            - capletValue
                            - LiteLibrary.blackCapletValue(DF_T0_Ti, Tau_i, L_T0_Ti_1_Ti, Strike, Ti_1, x)
                        )
                """
                solve for caplet volatility 
                starting from initial value sigma_cpl0 
                """
                sigma_cpl0 = 1.0
                sigma_cpl = fsolve (func, sigma_cpl0)
                #print ('CapletVolatility: {0}'.format(sigma_cpl))
                df['CapletVolatility'].iloc[idx] = sigma_cpl[0]
            else:
                sigma_caplet = df['CapletVolatility'].iloc[idx]
                capletValue = capletValue + LiteLibrary.blackCapletValue(DF_T0_Ti, Tau_i, L_T0_Ti_1_Ti, Strike, Ti_1, sigma_caplet)
        
    return df

class CplVolStripTests(TestCase):
    def testCplVolStrip(self):
        """
        Test caplet volatilities stripping algorithm 
        and call of resolveForCapletVolatility function
        """
        print('Caplet Volatility Stripping test')

        df = load_lmmCalibration()
        df = resolveForCapletVolatility(df)
        df['SigmaCaplet2*TimeToMaturity'] = df.apply(lambda row: np.power(row['CapletVolatility'], 2)*row['DeltaT0Ti'] , axis=1)

        LiteLibrary.writeDataFrame(df, 'lmmCalibration')
        print('See column CapletVolatility from the lmmCalibration sheet in the lmmData.xlsx file.')
        plotStrippedCapletVolatilities(df)                