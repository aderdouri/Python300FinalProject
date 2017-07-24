"""
Date: Monday, 24 July 2017
File name: 
Version: 1 
Author: Abderrazak DERDOURI
Subject: CQF Final Project

Description: Solve for the v1v2v3v4 (or abcd) parameters as described in paragraph 3.5.1

Notes: AbderrazakDerdouriCQFFinalProject.pdf

Run: python -m unittest abcdInstantaneousVolFit.InstVolFitTests.testabcdFit
     python -m unittest abcdInstantaneousVolFit.InstVolFitTests.testParametricInstVol 

requirement : must be run after  
              python -m unittest capletVolatilityStripping.CplVolStripTests.testCplVolStrip
              to have lmmCalibration sheet already prepared

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


def f(t, Ti, v):
    """
    See paragraph 3.5.1 from the notes for specification of this function
    """
    v1 = v[0]
    v2 = v[1]
    v3 = v[2]
    v4 = v[3]
    return abs(v1 + (v2 + v3*(Ti-t))*math.exp(-v4*(Ti-t)))

def I2(Ti, v):
    """
    See paragraph 3.5.1 from the notes for specification of this function
    """
    intgrl, abserr = quad(lambda t: f(t, Ti, v)**2, 0, Ti) 
    return intgrl

def plotParametricCapletVolatiliy(df):
    """
    Plot Market Data Stripped Caplet volatilities vs and Parametric volatilities
    """
    ParametricCapletVolatiliy = df[['SigmaCaplet2*TimeToMaturity', 'ParametricCapletVolatiliy']][8:46] # start  from  T6M-10Y
    myRange = np.array([8, 10, 15, 20, 25, 30, 35, 40, 42, 46])
    xticks = list(df['TenorTi'].iloc[myRange])        
    ax = ParametricCapletVolatiliy.plot(title='Market Data Stripped Caplet volatilities vs and Parametric volatilities')
    fig = ax.get_figure()
    ax.set_xticks(myRange);
    ax.set_xticklabels(xticks, rotation=45)
    plt.savefig('ParametricCapletVolatiliy')
    plt.show()        


class InstVolFitTests(TestCase):
    def testabcdFit(self):
        """
        Run a test solving for v1v2v3v4(abcd) parameters
        """
        df = pd.read_excel('lmmData.xlsx', sheetname='lmmCalibration')
        
        start_idx = df[df['TenorTi']=='T6M'].index[0]
        end_idx = df[df['TenorTi']=='T10Y'].index[0]

        deltaT0Ti	= df['DeltaT0Ti'][start_idx:1+end_idx]
        deltaT0TiIndex	= deltaT0Ti.index
        sigmaCaplet2DeltaT0Ti = df['SigmaCaplet2*TimeToMaturity'][start_idx:1+end_idx]
        
        def abcdObjectiveFunc(v):
            v1 = v[0]
            v2 = v[1]
            v3 = v[2]
            v4 = v[3]

            fmin = 0.0
            length = len(deltaT0Ti)
            for idx in range(0, length):
                fmin = fmin + (sigmaCaplet2DeltaT0Ti.iloc[idx] - I2(deltaT0Ti.iloc[idx], v))**2

            return math.sqrt(fmin)

        
        v0 = [0.1, 0.1, 0.1, 0.1]
        solution = optimize.minimize(abcdObjectiveFunc, v0, method='BFGS')
        print(solution)
        df['abcd'] = np.NaN

        for idx in deltaT0TiIndex:
            df['abcd'].iloc[idx] = f(0, deltaT0Ti.iloc[idx-start_idx], solution.x)

        
        abcdFunction = df[['abcd']][start_idx:end_idx]
        ax = abcdFunction.plot(title='Instantaneous volatility function')
        fig = ax.get_figure()
        ax.legend(labels=['abcd=' + str(solution.x)])
        plt.savefig('InstantaneousVolatilityFunctions')
        plt.show()        

    def testParametricInstVol(self):
        """
        Starting from the parameters v1v2v3v4(abcd) estimated
        in the previous test calculate the instantaneous volatility
        for maturities from T6M to T10Y
        """
        v = np.array([-0.10522141, 0.4215886, -1.03067277, 1.23953503])

        df = pd.read_excel('lmmData.xlsx', sheetname='lmmCalibration')
        
        start_idx = df[df['TenorTi']=='T6M'].index[0]
        end_idx = df[df['TenorTi']=='T10Y'].index[0]

        deltaT0Ti	= df['DeltaT0Ti'][start_idx:1+end_idx]
        sigmaCaplet2DeltaT0Ti = df['SigmaCaplet2*TimeToMaturity'][start_idx:1+end_idx]
        volatility = df['CapletVolatility'][start_idx:1+end_idx]

        length = len(deltaT0Ti)
        df['ParametricCapletVolatiliy'] = np.NaN
        for idx in range(length):
            df['ParametricCapletVolatiliy'].iloc[idx+8] = I2(deltaT0Ti.iloc[idx], v)

        LiteLibrary.writeDataFrame(df, 'lmmCalibration')
        plotParametricCapletVolatiliy(df)