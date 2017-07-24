"""
Date: Monday, 24 July 2017
File name: monteCarloCapPricing.py
Version: 1 
Author: Abderrazak DERDOURI
Subject: CQF Final Project

Description: Price a cap using MonteCarlo simulation 
             Plot MonteCarlo simulated price vs Analytical price

Notes: AbderrazakDerdouriCQFFinalProject.pdf

Run: python -m unittest monteCarloCapPricing.MCCapPricingTests.testMCCapPricing

Revision History:
"""

import pandas as pd
from unittest import TestCase
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import LiteLibrary


def monteCarloSimu(Notional, Strike, SpoteRatesList, tau, Expiry, sigmaCapletsList, timeSteps, NBSIMUS):
    """
    Monte Carlo simulations as described in paragraph 3.5.1
    """
    # STEP 1: READ INPUT CONTRACT
    K = Strike
    sigma = sigmaCapletsList
    N = Expiry # number forward rates
    dT = timeSteps

    L = np.zeros( (N, N) ) # forward rates
    D = np.zeros( (N, N) ) # discount factors
    dW = np.zeros( N ) # brownien motions
    V = np.zeros( N ) # future value payment
    FVprime = np.zeros( N ) # numeraire-rebased FV payment
    VCap = np.zeros( NBSIMUS ) # simulation payoff
        
    df_prod = 1.0
    drift_sum = 0.0        
   
	# STEP 2: INITIALISE SPOT RATES
    for i in range(N):
        L[i][0] =  SpoteRatesList[i]
        
    # start main MC loop
    for nsim in range(0, NBSIMUS):
        # STEP 3: BROWNIAN MOTION INCREMENTS
        for i in range(0, N):
            dW[i]=math.sqrt(dT)*np.random.normal()

        # STEP 4: COMPUTE FORWARD RATES TABLEAU
        for n in range(0, N-1):
            for i in range(n+1, N):
                drift_sum = 0.0
                for k in range(i+1, N):
                    drift_sum = drift_sum + (tau*sigma[k]*L[k][n])/(1+tau*L[k][n])
                L[i][n+1] = L[i][n]*math.exp((-drift_sum*sigma[n] - 0.5*sigma[n]*sigma[n])*dT + sigma[n]*dW[n+1])
            
        # STEP 5: COMPUTE DISCOUNT RATES TABLEAU
        for n in range(0, N):
            for i in range(n+1, N+1):
                df_prod = 1.0
                for k in range(n, i):
                    df_prod = df_prod * 1/(1+tau*L[k][n])
                D[i-1][n] = df_prod

        # STEP 6: COMPUTE EFFECTIVE FV CAPLET PAYMENTS
        for i in range(0, N):
            V[i] = Notional*tau*(max(L[i][i]-K, 0))

        # STEP 7: COMPUTE NUMERAIRE-REBASED PAYMENT
        for i in range(0, N):
            FVprime[i]=V[i]*D[i][i]/D[N-1][i]
            
        # STEP 8: COMPUTE IRS NPV
        VCap[nsim] = sum(FVprime)

    # STEP 9: COMPUTE DISCOUNTED EXPECTED PAYOFF
    payoff = D[N-1][0] * (sum(VCap) / NBSIMUS)
    return payoff


class MCCapPricingTests(TestCase):       
    def testMCCapPricing(self):        
        """
        Let's consider a range of caps each one is a set of 5 caplets
        and price them analytically and using Monte Carlo simulation
        Plot MonteCarlo simulated price vs Analytical price
        capStart/capMat = ['T3Y6M/T4Y9M', 'T3Y9M/T5Y', 'T4Y/T5Y3M', 'T4Y3M/T5Y6M', 'T4Y6M/T5Y9M'
        , 'T4Y9M/T6Y', 'T5Y/T6Y3M', 'T5Y3M/T6Y6M', 'T5Y6M/T6Y9M', 'T5Y9M/T7Y']
        """
        df = pd.read_excel('lmmData.xlsx', sheetname='lmmCalibration')      
        start_idx = df[df['TenorTi']=='T1Y'].index[0]
        maturity_idx = df[df['TenorTi']=='T2Y3M'].index[0]
        
        Notional = 1.0
        Strike = 0.0236
        tau = 0.25
        Expiry = 5
        timeSteps = 0.25
        NBSIMUS = 1000 # number of simulations

        analyticalCapletValueList = []
        sigmaCapletsList = []
        SpoteRatesList = []
        capMaturities = []

        df['analyticalCapValue'] = np.NaN
        df['monteCarloCapValue'] = np.NaN
        df['diffAnaliticalMC'] = np.NaN

        for t in range(10, 20):
            capStart = df['TenorTi'].iloc[start_idx+t]
            capMat = df['TenorTi'].iloc[maturity_idx+t]
            capMaturities.append(capStart+'/'+capMat)

            analyticalCapletValueList = []
            sigmaCapletsList = []
            SpoteRatesList = []
            for idx in range(start_idx+t, maturity_idx+t):
                DF_T0_Ti_1 = df['DF'].iloc[idx-1] # D(T0, Ti-1)
                DF_T0_Ti = df['DF'].iloc[idx]     # D(T0, Ti)
                Tau_i = 0.25

                sigmaCaplet = df['CapletVolatility'].iloc[idx]
                sigmaCapletsList.append(sigmaCaplet)

                L_T0_Ti_1_Ti = df['LT0Ti-3MTi'].iloc[idx]
                SpoteRatesList.append(L_T0_Ti_1_Ti)

                Ti_1 = df['DeltaT0Ti'].iloc[idx]

                capletValue = LiteLibrary.blackCapletValue(DF_T0_Ti, Tau_i, L_T0_Ti_1_Ti, Strike, Ti_1, sigmaCaplet)
                analyticalCapletValueList.append(capletValue)
        
            df['analyticalCapValue'].iloc[t] = sum(analyticalCapletValueList)

            payoff = monteCarloSimu(Notional, Strike, SpoteRatesList, tau, Expiry, sigmaCapletsList, timeSteps, NBSIMUS)
            df['monteCarloCapValue'].iloc[t] = payoff
            df['diffAnaliticalMC'].iloc[t] = df['monteCarloCapValue'].iloc[t] - df['analyticalCapValue'].iloc[t]
        
        AnalyticalVsMCCap = df[['monteCarloCapValue', 'analyticalCapValue', 'diffAnaliticalMC']][10:20]
        print(AnalyticalVsMCCap)
        xticks = capMaturities     
        ax = AnalyticalVsMCCap.plot(title='Analytical vs monteCarlo Cap pricing')
        fig = ax.get_figure()
        ax.set_xticklabels(xticks, rotation=20)
        plt.savefig('MonteCarloVSAnalyticCapPrice')
        plt.show()        