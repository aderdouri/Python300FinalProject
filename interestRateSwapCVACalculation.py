"""
Date: Monday, 24 July 2017
File name: interestRateSwapCVACalculation.py
Version: 1 
Author: Abderrazak DERDOURI
Subject: CQF Final Project

Description: Implement Credit Value Adjustement for an Interest Rate Swap

Notes: AbderrazakDerdouriCQFFinalProject.pdf

Run: python -m unittest interestRateSwapCVACalculation.IRSCVATests.test01IRSCVA
     python -m unittest interestRateSwapCVACalculation.IRSCVATests.test02IRSCVA

Revision History:
"""

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest import TestCase


def load_data(sheetname):    
    cvaInputData = pd.read_excel('cvaInputData.xlsx', sheetname=sheetname, skiprows=1)
    cvaInputData.columns = ['FixedRate', 'SpotRate', 'Tau', 'Maturity', 'Notional', 'Lambda', 'RecoveryRate']
    return cvaInputData


def MonteCarloSimu(sheetName):
    """
    Monte Carlo simulation as described in Algorithm 2.4.3 (see notes)
    """
    # STEP 1: READ INPUT CONTRAT
    df = load_data(sheetName)
    Notional = df['Notional'][0]
    K = df['FixedRate'][0] # fixed rate IRS
    tau = df['Tau'][0] # daycount factor
    RR = df['RecoveryRate'][0]# Recovery Rate
    Lambda = df['Lambda'][0]# Recovery Rate
    Maturity = df['Maturity'][0]# Swap Maturity

    sigma = 0.12 # fwd rates volatility
    N = int(Maturity/tau) # number forward rates
    dT = 0.25
    NBSIMUS = 10000 # number of simulations

    L = np.zeros( (N, N) ) # forward rates
    D = np.zeros( (N, N) ) # discount factors
    dW = np.zeros( N ) # brownien motions
    FV = np.zeros( N ) # future value payment
    FVprime = np.zeros( N ) # numeraire-rebased FV payment
    MtM = np.zeros( N+1 )        
    V = np.zeros( NBSIMUS ) # simulation payoff
    CVA_PERIOD = np.zeros( N+1 ) 

        
    df['Ti'] = tau
    idx = df['Ti'].index
    for i in idx:
        df['Ti'].iloc[i] = tau*(i + 1)
    df['Ti_1'] = df['Ti'].shift(1)


    df['SpotRate_i_1'] = df['SpotRate'].shift(1)
    df['forwardRate'] = df['SpotRate']
    forwardRate0 = df['forwardRate'].iloc[0]

    df['forwardRate'] = df.apply(lambda row: row['forwardRate'] if (forwardRate0==row['forwardRate']) 
                                    else (row['Ti']*row['SpotRate'] - row['Ti_1']*row['SpotRate_i_1'])
                                    /(row['Ti']-row['Ti_1']), axis=1)

    df['PD'] = df.apply(lambda row: 1-math.exp(-Lambda*row['Ti']) if (forwardRate0==row['forwardRate']) 
                        else math.exp(-Lambda*row['Ti_1']) - math.exp(-Lambda*row['Ti'])
                        , axis=1)


    df_prod = 1.0
    drift_sum = 0.0        

    Exposures = []
	# STEP 2: INITIALISE SPOT RATES
    for i in range(0, N):
        L[i][0] = df['forwardRate'].iloc[i]
        
    # Start main MC loop
    for nsim in range(0, NBSIMUS):
        # STEP 3: BROWNIAN MOTION INCREMENTS
        for i in range(0, N):
            dW[i]=math.sqrt(dT)*np.random.normal()

        # STEP 4: COMPUTE FORWARD RATES TABLEAU
        for n in range(0, N-1):
            for i in range(n+1, N):
                drift_sum = 0.0
                for k in range(i+1, N):
                    drift_sum = drift_sum + (tau*sigma*L[k][n])/(1+tau*L[k][n])
                L[i][n+1] = L[i][n]*math.exp((-drift_sum*sigma-0.5*sigma*sigma)*dT + sigma*dW[n+1])
            

        # STEP 5: COMPUTE DISCOUNT RATES TABLEAU
        for n in range(0, N):
            for i in range(n+1, N+1):
                df_prod = 1.0
                for k in range(n, i):
                    df_prod = df_prod * 1/(1+tau*L[k][n])
                D[i-1][n] = df_prod

        # STEP 6: COMPUTE EFFECTIVE FV PAYMENTS
        for i in range(0, N):
            FV[i] = Notional*tau*(L[i][i]-K)

        # STEP 7: COMPUTE NUMERAIRE-REBASED PAYMENT
        for i in range(0, N):
            FVprime[i] = FV[i]*D[i][i]/D[N-1][i]
            
        # STEP 8: COMPUTE IRS NPV
        V[nsim] = sum(FVprime*D[:,0])

        for i in range(0, N):
            MtM[i] = sum(FVprime[i:]*D[i:,0])
        MtM[N] = 0.0

        Exposure = np.zeros( N+1 )
        ExposureAvg = np.zeros( N+1 )

        for i in range(0, N+1):
            Exposure[i] = max(0, MtM[i])

        for i in range(0, N+1):
            if (i==N):
                ExposureAvg[N]
            else:
                ExposureAvg[i] = (Exposure[i] + Exposure[i+1])/2.0

        Exposures.append(ExposureAvg)
        #End main MC loop

    np_Exposures = np.array(Exposures)        
    ExpectedExposure = np.average(np_Exposures, axis=0)
    myPD = df['PD']

    for i in range(0, N):
        CVA_PERIOD[i] = (1-RR)*myPD[i]*ExpectedExposure[i]

    print('CVA_PERIOD: {0}'.format(CVA_PERIOD))
    print(CVA_PERIOD.sum())

    return df, ExpectedExposure, CVA_PERIOD
    
def plotCVA(data, figureName, title):
    """
    Plot a data list 
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    N = len(data)
    ind = np.arange(N)/2.0 # the x locations for the groups
    indYears = tuple ( (str(e)+'Y' for e in ind))
    width = 0.35       # the width of the bars
    ax  = plt.gca()
    #ax.set_xlim([0.0, .0])
    print(ind)

    rects = ax.bar(ind, data, width, color='orange')	
    plt.xticks(ind, indYears)
    plt.plot(ind, data)
    plt.title(title)

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%d' % int(height), ha='center', va='bottom')

    autolabel(rects)
    plt.savefig(figureName)
    plt.show()

def plotTermStructure(df, figureName, title):
    ax = df[['SpotRate', 'forwardRate']].plot(title=title)
    xticks = ['6M', '1Y', '1.5Y', '2Y', '2.5Y', '3Y', '3.5Y', '4Y', '4.5Y', '5Y']
    ax.set_xticklabels(xticks, rotation=45)
    plt.savefig(figureName)
    plt.show()


class IRSCVATests(TestCase):    
    def test01IRSCVA(self):        
        df, ExpectedExposure, CVA_PERIOD = MonteCarloSimu('cvaIncreasing')

        plotTermStructure(df, 'IncreasingTermStructure', 'Increasing Term structure')

        plotCVA(ExpectedExposure, 'IncreasingTermStructureExpectedExposure', 'IRS Expected Exposure')
        plotCVA(CVA_PERIOD, 'IncreasingTermStructureCVAPeriod', 'CVA Period')
        print('CVA_PERIOD: {0}'.format(sum(CVA_PERIOD)))


    def test02IRSCVA(self):        
        df, ExpectedExposure, CVA_PERIOD = MonteCarloSimu('cvaDecreasing')

        plotTermStructure(df, 'DecreasingTermStructure', 'Decreasing Term structure')

        plotCVA(ExpectedExposure, 'DecreasingTermStructureExpectedExposure', 'IRS Expected Exposure')
        plotCVA(CVA_PERIOD, 'DecreasingTermStructureCVAPeriod', 'CVA Period')
        print('CVA_PERIOD: {0}'.format(sum(CVA_PERIOD)))
