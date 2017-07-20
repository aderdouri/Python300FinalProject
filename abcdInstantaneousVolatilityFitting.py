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


from unittest import TestCase
import sys
import math
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from datetime import date

from scipy.optimize import root, fsolve
import scipy.optimize as optimize
from scipy.integrate import quad
from scipy import integrate

import marketDataPreprocessing
import pandas as pd
import numpy as np


class Optimization_tests(TestCase):
    def test_optimization(self):
        df = pd.read_excel('PreProcessedMarketData.xlsx'
                                           , sheetname='PreProcessedMarketData')
        

        T6M = df[df['TenorTi']=='T6M']
        T10Y = df[df['TenorTi']=='T10Y']
        start_idx = T6M.index[0]
        end_idx = T10Y.index[0]

        myDeltaT0Ti	= df['DeltaT0Ti'][start_idx:1+end_idx]
        mySigmaCaplet2DeltaT0Ti = df['SigmaCaplet^2*TimeToMaturity'][start_idx:1+end_idx]
        
        def f(t, Ti, v):
            v1 = v[0]
            v2 = v[1]
            v3 = v[2]
            v4 = v[3]
            return v1 + (v2 + v3*(Ti-t))*math.exp(-v4*(Ti-t))

        # In fact I2 is sqaure (I^2)
        def I2(Ti, v):
            intgrl, abserr = quad(lambda t: f(t, Ti, v)**2, 0, Ti) 
            return intgrl


        def epsilon(Ti, g):
            g1 = g[0]
            g2 = g[1]
            g3 = g[2]
            return g1 + g2*math.cos(g3*Ti)

        def corr(Ti, g, v):
            # v is the abcd parameters estmated
            # with the abcdObjectiveFunc function
            # v = [-0.10523557, 0.42152995, -1.03073371, 1.23969396]
             return (1 + epsilon(Ti, g))*I2(Ti, v)

        def abcdObjectiveFunc(v):
            v1 = v[0]
            v2 = v[1]
            v3 = v[2]
            v4 = v[3]

            fmin = 0.0
            #print('v = {0}'.format(v))
            length = len(myDeltaT0Ti)
            for idx in range(0, length):
                fmin = fmin + (mySigmaCaplet2DeltaT0Ti.iloc[idx] - I2(myDeltaT0Ti.iloc[idx], v))**2

            return math.sqrt(fmin)


        def g1g2g3ObjectiveFunc(g, v):
            fmin = 0.0
            #print('g = {0}'.format(g))
            length = len(myDeltaT0Ti)
            for idx in range(0, length):
                fmin = fmin + (mySigmaCaplet2DeltaT0Ti.iloc[idx] - corr(myDeltaT0Ti.iloc[idx], g, v))**2

            return math.sqrt(fmin)


        
        v0 = [0.1, 0.1, 0.1, 0.1]
        solution = optimize.minimize(abcdObjectiveFunc, v0, method='BFGS')
        print(solution)
        print('abcd: {0}'.format(solution.x))
        print('\n')

        '''
        data = solution.x
        g0 = [0.1, 0.1, 0.1]
        solution = optimize.minimize(g1g2g3ObjectiveFunc, g0, args=(data),  method='SLSQP')
        print(solution)
        print('----------------------------')
        print('g1g2g3: {0}'.format(solution.x))
        print('\n')
        '''
   

        v = [-0.10523557, 0.42152995, -1.03073371, 1.23969396]
        g = [-0.0008404, -0.00193876, 0.30603881]

        '''
        for idx in range(start_idx, 1+end_idx):
            str = df['TenorTi'][idx]
            str = str + ': {0}'
            print(str.format(abs(f(0.0, df['DeltaT0Ti'].ix[idx], v))))
        '''
        
        '''
        for idx in range(start_idx, 1+end_idx):
            str = df['TenorTi'][idx]
            str = str + ': {0}'
            print(str.format(I2(df['DeltaT0Ti'].ix[idx], v)))
        '''

        '''
        df['DELTA_i'] = df.apply(lambda row: 
                                 (row['SigmaCaplet^2*TimeToMaturity']/corr(row['DeltaT0Ti'] , g, v) - 1)
                                 , axis = 1)        
        #df['DELTA_i'] = df.apply(lambda row: row.index.values ,axis = 1)
       
        for idx in range(start_idx, 1+end_idx):
            df['DELTA_i'].ix[idx] = (df['SigmaCaplet^2*TimeToMaturity'].ix[idx]/corr(df['DeltaT0Ti'].ix[idx], g, v) - 1)


        for idx in range(start_idx, 1+end_idx):
            print(idx)
        
        #print(df['DELTA_i'])
        #df['CapletVolatility'].ix[idx] = x[0]
        #writeDataFrame(df)
        '''
