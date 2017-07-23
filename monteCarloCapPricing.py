"""
Date: Monday, 24 July 2017
File name: monteCarloCapPricing.py
Version: 1 
Author: Abderrazak DERDOURI
Subject: CQF Final Project

Description: Price a cap using MonteCarlo simulation 
             Plot MonteCarlo simulated price vs Analytical price

Notes: AbderrazakDerdouriCQFFinalProject.pdf

Run: python -m unittest monteCarloCapPricing.MonteCarloCapPricingTests.testMonteCarloCapPricing

Revision History:
"""

import pandas as pd
from unittest import TestCase
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import LiteLibrary


def load_data(sheetname):    
    cvaInputData = pd.read_excel('capData.xls', sheetname=sheetname, skiprows=1)
    cvaInputData.columns = ['FixedRate', 'SpotRate', 'Tau', 'Maturity', 'Notional', 'Lambda', 'RecoveryRate']
    return cvaInputData

def date_diff(row):
    return 1

def year_fraction(start_date, end_date):
    """Returns fraction in years between start_date and end_date, using Actual/360 convention"""
    return day_count(start_date, end_date) / 360.0

def calculateMtM(df):
    for index, row in df.iterrows():
        df.loc[index+1, 'MtM'] = df[1+index:5]['Payment'].sum()
    return df

def calculateExposureAverage(df):
    nbElement = len(df.index)
    for index in range(nbElement-1):
        df.loc[index+1, 'ExposureAvg'] = (df.loc[index, 'Exposure'] + df.loc[1+index, 'Exposure'])/2.0
        df.loc[index+2, 'DFAvg'] = (df.loc[index, 'DF'] + df.loc[1+index, 'DF'])/2.0
        df.loc[index+1, 'PDAvg'] = df.loc[index, 'PD']

    return df

def monteCarloSimu(Notional, Strike, SpoteRatesList, tau, Expiry, sigmaCapletsList, timeSteps, NBSIMUS):
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
            
        #print('L: {0}'.format(L))
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
            #print('L[i][i]: {0}'.format(L[i][i]))
            #V[i] = max(L[i][i]-K, 0.0)

        # STEP 7: COMPUTE NUMERAIRE-REBASED PAYMENT
        for i in range(0, N):
            FVprime[i]=V[i]*D[i][i]/D[N-1][i]
            #FVprime[i] = V[i]*DF[i]

            
        # STEP 8: COMPUTE IRS NPV
        VCap[nsim] = sum(FVprime)

    # STEP 9: COMPUTE DISCOUNTED EXPECTED PAYOFF
    sumCap = 0.0
    for nsim in range(0, NBSIMUS):
        sumCap = sumCap + VCap[nsim]
    payoff = D[N-1][0] * sumCap / NBSIMUS
    return payoff


class MonteCarloCapPricingTests(TestCase):    
    def testMonteCarloCapPricing(self):        
        """
        Test MonteCarlo simulation Cap Pricing
        Plot MonteCarlo simulated price vs Analytical price
        """
        df = pd.read_excel('marketData.xlsx', sheetname='preProcessedMarketData')      

        DF_T6M_idx = df[df['TenorTi']=='T6M'].index[0]    
        start_idx = DF_T6M_idx
        max_idx = len(df.index)
        print('max_idx: {0}'.format(max_idx))
        
        analyticCapValueList  = []
        monteCarloCapValueList  = []
        for row_idx in range(start_idx +1, start_idx+20):
            sigmaCapletsList = []
            SpoteRatesList = []
            capValue = 0.0

            for idx in range(start_idx, row_idx+1):
                DF_T0_Ti_1 = df['DF'].iloc[idx-1] # B(T0, Ti-1)
                DF_T0_Ti = df['DF'].iloc[idx]     # B(T0, Ti)
                Tau_i =  df['Delta'].iloc[idx]    # time(T3M, Ti)
                #print('Tau_i: {0}'.format(Tau_i))

                sigmaCapletsList.append(df['CapletVolatility'].iloc[idx])
                L_T0_Ti_1_Ti = LiteLibrary.liborRate(DF_T0_Ti_1, DF_T0_Ti, Tau_i) 
                SpoteRatesList.append(L_T0_Ti_1_Ti)


                Strike = df['ForwardSwapRate'].iloc[idx] # S(T0, T3M, Ti)
                Ti_1 = df['DeltaT0Ti'].iloc[idx]
                r = 0.0
                sigma_cap = df['CapVolatility'].iloc[row_idx]
                capValue = capValue + DF_T0_Ti*Tau_i*LiteLibrary.bsm_call_value(L_T0_Ti_1_Ti, Strike, Ti_1, r, sigma_cap)

            analyticCapValueList.append(capValue)

            Notional = 1.0 #df['Notional'][0]
            K = 0.02361    #df['FixedRate'][0] # fixed rate IRS
            tau = 0.25     # df['Tau'][0] # daycount factor
            Expiry = (row_idx-start_idx+1)
            #print('Expiry: {0}'.format(Expiry)) 
            timeSteps = 0.25
            NBSIMUS = 1000 # number of simulations

       
            payoff = monteCarloSimu(Notional, K, SpoteRatesList, tau, Expiry, sigmaCapletsList, timeSteps, NBSIMUS)
            monteCarloCapValueList.append(payoff)
            print('payoff: {0}'.format(payoff))        
        
        diff = np.array(monteCarloCapValueList) - np.array(analyticCapValueList)
        print('monteCarloCapValueList: {0}'.format(monteCarloCapValueList))        
        print('analyticCapValueList: {0}'.format(analyticCapValueList))        

        plt.figure()
        plt.plot(monteCarloCapValueList, label='MonteCarlo')
        plt.plot(analyticCapValueList, label='Analytic')
        plt.plot(diff, label='Differences')
        plt.legend()
        plt.title('Payer SwaptionPrice StrikeLevel')
        plt.xlabel('Strike %')
        plt.ylabel('Cap Price')
        plt.savefig('MoneCarloVSAnalyticCapPrice')
        plt.show()       