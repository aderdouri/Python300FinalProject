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
from scipy.optimize import root, fsolve
from unittest import TestCase
import numpy as np
import marketDataPreprocessing
import matplotlib.pyplot as plt

def resolveForCapletVolatility(df):
    DF_T6M_idx = df[df['TenorTi']=='T6M'].index[0]    
    start_idx = DF_T6M_idx
    max_idx = len(df.index)

    for row_idx in range(start_idx +1, max_idx):
        capValue = 0.0
        capletValue  = 0.0
        for idx in range(start_idx, row_idx+1):
            DF_T0_Ti_1 = df['DF'].iloc[idx-1] # B(T0, Ti-1)
            DF_T0_Ti = df['DF'].iloc[idx]     # B(T0, Ti)
            Tau_i =  df['Delta'].iloc[idx]    # time(T3M, Ti)

            Strike = df['ForwardSwapRate'].iloc[idx] # S(T0, T3M, Ti)
            L_T0_Ti_1_Ti = LiteLibrary.liborRate(DF_T0_Ti_1, DF_T0_Ti, Tau_i) # L(Ti-1, Ti-1, Ti) = F(T0, Ti-1, Ti)
            Ti_1 = df['DeltaT0Ti'].iloc[idx]
            r = 0.0
            sigma_cap = df['CapVolatility'].iloc[row_idx]

            capValue = capValue + DF_T0_Ti*Tau_i*LiteLibrary.bsm_call_value(L_T0_Ti_1_Ti, Strike, Ti_1, r, sigma_cap)

            if (idx==row_idx):
                def func(x):
                    return ( capValue 
                            - capletValue
                            - DF_T0_Ti*Tau_i*LiteLibrary.bsm_call_value(L_T0_Ti_1_Ti, Strike, Ti_1, r, x)
                        )

                sigma_cpl0 = 1.0
                sigma_cpl = fsolve (func, sigma_cpl0)
                #print ('sigma_cpl={0}'.format(sigma_cpl))
                df['CapletVolatility'].iloc[idx] = sigma_cpl[0]
            else:
                sigma_caplet = df['CapletVolatility'].iloc[idx]
                capletValue = capletValue + DF_T0_Ti*Tau_i*LiteLibrary.bsm_call_value(L_T0_Ti_1_Ti, Strike, Ti_1, r, sigma_caplet)
        
    return df

class CapletVolatilityStrippingTests(TestCase):
    def test_capletVolatilityStripping(self):
        print('capletVolatilityStripping')

        df = marketDataPreprocessing.preProcessMarketData()
        LiteLibrary.writeDataFrame(df)

        df = resolveForCapletVolatility(df)
        df['SigmaCaplet^2*TimeToMaturity'] = df.apply(lambda row: np.power(row['CapletVolatility'], 2)*row['DeltaT0Ti'] , axis=1)

        CapVolatility    = df['CapVolatility']
        CapletVolatility    = df['CapletVolatility']
        plt.plot(CapVolatility)
        plt.plot(CapletVolatility)
        plt.show()


