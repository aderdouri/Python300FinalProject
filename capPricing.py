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
from unittest import TestCase
import pandas as pd
from scipy.optimize import root, fsolve
import numpy as np
import matplotlib.pyplot as plt


class CapPricingTests(TestCase):
    def test_capletPricing(self):
        return
        print('test_capletPricing')
        dateT0 = pd.to_datetime('25-01-2005')
        dateT1Y = pd.to_datetime('25-01-2006')
        dateT1Y_3M = pd.to_datetime('25-04-2006')
        deltaT0_TY1 = LiteLibrary.year_fraction(dateT0, dateT1Y)
        deltaTY1_T1Y3M = LiteLibrary.year_fraction(dateT1Y, dateT1Y_3M)
    
        B_T0_T1Y = 0.9774658
        B_T0_T1Y3M = 0.9712884
        Tau_i = deltaTY1_T1Y3M

        Strike = 0.02361
        F_T0_T1Y_T1Y3M = LiteLibrary.liborRate(B_T0_T1Y, B_T0_T1Y3M, deltaTY1_T1Y3M)
        
        print('F_T0_T1Y_T1Y3M: {0}'.format(F_T0_T1Y_T1Y3M))

        Ti_1 = deltaT0_TY1
        r = 0.0
        sigma_caplet = 0.2015
        capletPrice  = B_T0_T1Y3M*Tau_i*LiteLibrary.bsm_call_value(F_T0_T1Y_T1Y3M, Strike, Ti_1, r, sigma_caplet)

        print('capletPrice: {0}'.format(capletPrice))

    def test_capPricing(self):
        return
        print('test_capPricing')
        F_T0_T3M_T6M = 0.021944643
        F_T0_T3M_T9M = 0.0229441
        F_T0_T3M_T1Y = 0.024150014
        F_T0_T3M_T1Y3M = 0.027187013

        Notional = 1000000
        r = 0.0
        Strike = 0.0361
        tau = 0.25
        Fwd = [0.021944643, 0.0229441, 0.024150014 ,0.027187013]        
        sigma = [0.1641, 0.1641, 0.1641, 0.206419203]
        T = [0.5, 0.75, 1, 1.25]
        DF = [0.989265115 ,0.98349838, 0.977465783, 0.97086704]

        capletPrice = []
        length = len(Fwd)
        for i in range(length):
            capletPrice.append(Notional*DF[i]*tau*LiteLibrary.bsm_call_value(Fwd[i], Strike, T[i], r, sigma[i]))
        
        print('capletPrice Price: {0}'.format(sum(capletPrice)))

        capPrice = []
        sigma_cap = 0.1641
        length = len(Fwd)
        for i in range(length):
            capPrice.append(Notional*DF[i]*tau*LiteLibrary.bsm_call_value(Fwd[i], Strike, T[i], r, sigma_cap))

        print('Cap Price: {0}'.format(sum(capPrice)))

        

def capPrice(df):
    DF_T6M_idx = df[df['TenorTi']=='T6M'].index[0]    
    start_idx = DF_T6M_idx
    max_idx = len(df.index)
    print('start_idx: {0}'.format(start_idx))

    row_idx = df[df['TenorTi']=='T1Y9M'].index[0] # 1Y6M
    print('row_idx: {0}'.format(row_idx))

    capValue = 0.0
    capletValue  = 0.0

    for idx in range(start_idx, row_idx+1):
        DF_T0_Ti_1 = df['DF'].iloc[idx-1] # B(T0, Ti-1)
        DF_T0_Ti = df['DF'].iloc[idx]     # B(T0, Ti)
        Tau_i =  df['Delta'].iloc[idx]    # time(T3M, Ti)
        print('Tau_i: {0}'.format(Tau_i))

        Strike = df['ForwardSwapRate'].iloc[idx] # S(T0, T3M, Ti)
        L_T0_Ti_1_Ti = LiteLibrary.liborRate(DF_T0_Ti_1, DF_T0_Ti, 0.25) 

        Ti_1 = df['DeltaT0Ti'].iloc[idx]
        r = 0.0
        sigma_cap = df['CapVolatility'].iloc[row_idx]
        capValue = capValue + DF_T0_Ti*Tau_i*LiteLibrary.bsm_call_value(L_T0_Ti_1_Ti, Strike, Ti_1, r, sigma_cap)


    return capValue        

if __name__ == '__main__':  
    df = pd.read_excel('PreProcessedMarketData.xlsx', sheetname='PreProcessedMarketData')      
    print('Cap Price: {0}'.format(capPrice(df)))