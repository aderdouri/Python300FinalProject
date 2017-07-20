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
        df = pd.read_excel('PreProcessedMarketData.xlsx', sheetname='PreProcessedMarketData')
        
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
   