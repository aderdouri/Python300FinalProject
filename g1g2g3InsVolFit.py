"""
Date: Monday, 24 July 2017
File name: g1g2g3InstantaneousVolatilityFitting.py
Version: 1 
Author: Abderrazak DERDOURI
Subject: CQF Final Project

Description: Estimate parameters g1, g2, g3 as described in paragraph 3.5.1

Notes: AbderrazakDerdouriCQFFinalProject.pdf

Run: python -m unittest g1g2g3InsVolFit.g1g2g2FitTests.testParamsg1g2g2Fit
     python -m unittest g1g2g3InsVolFit.g1g2g2FitTests.testParametricCorrInstVol

Revision History:
"""

import math
import pandas as pd
import numpy as np
import scipy.optimize as optimize
from scipy.integrate import quad
from unittest import TestCase
import lmmCalibration
import LiteLibrary
import matplotlib.pyplot as plt


def plotParametricCorrCapletVolatiliy(df):
    """
    Plot Market Data Stripped Caplet volatilities vs and Parametric volatilities
    """
    ParametricCapletVolatiliy = df[['SigmaCaplet2*TimeToMaturity', 'ParametricCorrCapletVolatiliy']][8:46] # start  from  T6M-10Y
    myRange = np.array([8, 10, 15, 20, 25, 30, 35, 40, 42, 46])
    xticks = list(df['TenorTi'].iloc[myRange])        
    ax = ParametricCapletVolatiliy.plot(title='Market Data Stripped Caplet volatilities vs and Parametric volatilities')
    fig = ax.get_figure()
    ax.set_xticks(myRange);
    ax.set_xticklabels(xticks, rotation=45)
    plt.savefig('ParametricCorrCapletVolatiliy')
    plt.show()        

def plotParametric1vs2(df):
    """
    Plot Market Data Stripped Caplet volatilities vs and Parametric volatilities
    """
    ParametricCapletVolatiliy = df[['ParametricCapletVolatiliy', 'ParametricCorrCapletVolatiliy']][8:46] # start  from  T6M-10Y
    myRange = np.array([8, 10, 15, 20, 25, 30, 35, 40, 42, 46])
    xticks = list(df['TenorTi'].iloc[myRange])        
    ax = ParametricCapletVolatiliy.plot(title='Market Data Stripped Caplet volatilities vs and Parametric volatilities')
    fig = ax.get_figure()
    ax.set_xticks(myRange);
    ax.set_xticklabels(xticks, rotation=45)
    plt.savefig('ParametricVSCorrCapletVolatiliy')
    plt.show()        

def f(t, Ti, v):
    v1 = v[0]
    v2 = v[1]
    v3 = v[2]
    v4 = v[3]
    return v1 + (v2 + v3*(Ti-t))*math.exp(-v4*(Ti-t))

def I2(Ti, v):
    intgrl, abserr = quad(lambda t: f(t, Ti, v)**2, 0, Ti) 
    return intgrl


def epsilon(Ti, g):
    g1 = g[0]
    g2 = g[1]
    g3 = g[2]
    return g1 + g2*math.cos(g3*Ti)

def corr(Ti, g, v):
        """
        v is the abcd parameters estmated
        with the abcdObjectiveFunc function
        v = [-0.10523557, 0.42152995, -1.03073371, 1.23969396]
        """
        return (1 + epsilon(Ti, g))*I2(Ti, v)

class g1g2g2FitTests(TestCase):
    def testParamsg1g2g2Fit(self):
        """
        Paramaters g1, g2, g2 estimation test
        """
        df = pd.read_excel('lmmData.xlsx', sheetname='lmmCalibration')
        
        T6M = df[df['TenorTi']=='T6M']
        T10Y = df[df['TenorTi']=='T10Y']
        start_idx = T6M.index[0]
        end_idx = T10Y.index[0]

        myDeltaT0Ti	= df['DeltaT0Ti'][start_idx:1+end_idx]
        mySigmaCaplet2DeltaT0Ti = df['SigmaCaplet2*TimeToMaturity'][start_idx:1+end_idx]       

        def g1g2g3ObjectiveFunc(g, v):
            fmin = 0.0
            #print('g = {0}'.format(g))
            length = len(myDeltaT0Ti)
            for idx in range(0, length):
                fmin = fmin + (mySigmaCaplet2DeltaT0Ti.iloc[idx] - corr(myDeltaT0Ti.iloc[idx], g, v))**2

            return math.sqrt(fmin)
        
        abcd = [-0.10523557, 0.42152995, -1.03073371, 1.23969396]
        g0 = [0.1, 0.1, 0.1]
        solution = optimize.minimize(g1g2g3ObjectiveFunc, g0, args=(abcd),  method='SLSQP')
        print(solution)
        print('----------------------------')
        print('g1g2g3: {0}'.format(solution.x))
        print('\n')
   

    def testParametricCorrInstVol(self):
        """
        Starting from the parameters v1v2v3v4(abcd) and (g1g2g3) estimated
        in the previous test calculate the corrected instantaneous volatility
        for maturities from T6M to T10Y
        """
        v = np.array([-0.10522141, 0.4215886, -1.03067277, 1.23953503])
        g = np.array([-0.0008404, -0.00193876, 0.30603882])

        df = pd.read_excel('lmmData.xlsx', sheetname='lmmCalibration')
        
        start_idx = df[df['TenorTi']=='T6M'].index[0]
        end_idx = df[df['TenorTi']=='T10Y'].index[0]

        deltaT0Ti	= df['DeltaT0Ti'][start_idx:1+end_idx]
        sigmaCaplet2DeltaT0Ti = df['SigmaCaplet2*TimeToMaturity'][start_idx:1+end_idx]

        length = len(deltaT0Ti)
        df['ParametricCorrCapletVolatiliy'] = np.NaN
        for idx in range(length):
            df['ParametricCorrCapletVolatiliy'].iloc[idx+8] = corr(deltaT0Ti.iloc[idx], g, v)


        df['DeltaTi'] = np.NaN
        for idx in range(length):
            df['DeltaTi'].iloc[idx+8] = (df['SigmaCaplet2*TimeToMaturity'].iloc[idx+8]/df['ParametricCorrCapletVolatiliy'].iloc[idx+8])-1

        LiteLibrary.writeDataFrame(df, 'lmmCalibration')
        plotParametricCorrCapletVolatiliy(df)
        plotParametric1vs2(df)