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

import pandas as pd
from unittest import TestCase
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import LiteLibrary
from math import log, sqrt, exp
from scipy import stats


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

def monteCarloSimu(initialForwardVec, initialSigmaVec, corrMatrix, Notional, Strike, tau, Expiry, timeSteps, NBSIMUS, direction):
    K = Strike
    sigma = initialSigmaVec
    N = Expiry # number forward rates
    dT = timeSteps

    L = np.zeros( (N, N) ) # forward rates
    D = np.zeros( (N, N) ) # discount factors
    dW = np.zeros( N ) # brownien motions
    V = np.zeros( N ) # future value payment
    FVprime = np.zeros( N ) # numeraire-rebased FV payment
    VSwaption = np.zeros( NBSIMUS ) # simulation payoff
        
    df_prod = 1.0
    drift_sum = 0.0        
   
	# STEP 2: INITIALISE SPOT RATES
    for i in range(N):
        L[i][0] =  initialForwardVec[i]
        
    mean = N*[0]

    #print('corrMatrix: {0}'.format(corrMatrix))
    
    
    #NBSIMUS=10
    # start main MC loop
    for nsim in range(0, NBSIMUS):
        # STEP 3: BROWNIAN MOTION INCREMENTS
        #for i in range(0, N):
            #dW[i]=math.sqrt(dT)*np.random.normal()

        # Draws a new random vector
        dW = np.random.multivariate_normal(mean, corrMatrix)
        #print('dW: {0}'.format(dW))

        # STEP 4: COMPUTE FORWARD RATES TABLEAU
        for n in range(0, N-1):
            for i in range(n+1, N):
                drift_sum = 0.0
                for k in range(i+1, N):
                    #print('(n, k): ({0}, {1})'.format(n, k))
                    drift_sum = drift_sum + (tau*corrMatrix[n, k]*sigma[k]*L[k][n])/(1+tau*L[k][n])
                L[i][n+1] = L[i][n]*math.exp((-drift_sum*sigma[n] - 0.5*sigma[n]*sigma[n])*dT + sigma[n]*dW[n+1])
            
        #print('L: {0}'.format(L))
        # STEP 5: COMPUTE DISCOUNT RATES TABLEAU
        for n in range(0, N):
            for i in range(n+1, N+1):
                df_prod = 1.0
                for k in range(n, i):
                    df_prod = df_prod * 1/(1+tau*L[k][n])
                D[i-1][n] = df_prod

        
        #Calculate the forward Swap rate
        numerator =  1 - np.prod( 1/(1 + tau * L.diagonal()) )
        #print('numerator: {0}'.format(numerator))
        
        denominator = 0.0
        for j in range(0, N):
            denominator = denominator + tau * np.prod(1/(1+L.diagonal()[0:j+1]))

        fwdSwapRate = numerator / denominator
        #print('fwdSwapRate: {0}'.format(fwdSwapRate))

        PVBP = sum ( tau / ( 1 + 1 + L.diagonal() * tau ) )
        PVBP = D[N-1][0] * PVBP         
        swaptionPrice = max( direction*(fwdSwapRate - K), 0) * PVBP
        
        # STEP 7: COMPUTE NUMERAIRE-REBASED PAYMENT
        #for i in range(0, N):
            #FVprime[i]=V[i]*D[i][i]/D[N-1][i]
            #FVprime[i] = V[i]*DF[i]

        #print('swaptionPrice: {0}'.format(swaptionPrice))

        # STEP 8: COMPUTE IRS NPV
        VSwaption[nsim] = swaptionPrice
        #print('VCap: {0}'.format(VCap))
        #print('FVprime: {0}'.format(FVprime[1]))
        # end main MC loop

    #print('D: {0}'.format(D))

    # STEP 9: COMPUTE DISCOUNTED EXPECTED PAYOFF
    sumSwaption = 0.0
    for nsim in range(0, NBSIMUS):
        sumSwaption = sumSwaption + VSwaption[nsim]
    payoff = sumSwaption / NBSIMUS
    return payoff


def MCSimu(initialForwardVec, initialSigmaVec, corrMatrix, Notional, Strike, tau, Expiry, timeSteps, NBSIMUS):
    K = Strike
    sigma = initialSigmaVec
    N = Expiry # number forward rates
    dT = timeSteps

    L = np.zeros( (N, N) ) # forward rates
    D = np.zeros( (N, N) ) # discount factors
    dW = np.zeros( N ) # brownien motions
    V = np.zeros( N ) # future value payment
    FVprime = np.zeros( N ) # numeraire-rebased FV payment
    VSwaption = np.zeros( (NBSIMUS, 2) ) # simulation payoff
        
    df_prod = 1.0
    drift_sum = 0.0        
   
	# STEP 2: INITIALISE SPOT RATES
    for i in range(N):
        L[i][0] =  initialForwardVec[i]
        
    mean = N*[0]  
    
    #NBSIMUS=10
    # start main MC loop
    for nsim in range(0, NBSIMUS):
        # STEP 3: BROWNIAN MOTION INCREMENTS
        #for i in range(0, N):
            #dW[i]=math.sqrt(dT)*np.random.normal()

        # Draws a new random vector
        dW = np.random.multivariate_normal(mean, corrMatrix)
        #print('dW: {0}'.format(dW))

        # STEP 4: COMPUTE FORWARD RATES TABLEAU
        for n in range(0, N-1):
            for i in range(n+1, N):
                drift_sum = 0.0
                for k in range(i+1, N):
                    #print('(n, k): ({0}, {1})'.format(n, k))
                    drift_sum = drift_sum + (tau*corrMatrix[n, k]*sigma[k]*L[k][n])/(1+tau*L[k][n])
                L[i][n+1] = L[i][n]*math.exp((-drift_sum*sigma[n] - 0.5*sigma[n]*sigma[n])*dT + sigma[n]*dW[n+1])
            
        #print('L: {0}'.format(L))
        # STEP 5: COMPUTE DISCOUNT RATES TABLEAU
        for n in range(0, N):
            for i in range(n+1, N+1):
                df_prod = 1.0
                for k in range(n, i):
                    df_prod = df_prod * 1/(1+tau*L[k][n])
                D[i-1][n] = df_prod

        
        #Calculate the forward Swap rate
        numerator =  1 - np.prod( 1/(1 + tau * L.diagonal()) )
        #print('numerator: {0}'.format(numerator))
        
        denominator = 0.0
        for j in range(0, N):
            denominator = denominator + tau * np.prod(1/(1+L.diagonal()[0:j+1]))

        fwdSwapRate = numerator / denominator
        #print('fwdSwapRate: {0}'.format(fwdSwapRate))

        PVBP = sum ( tau / ( 1 + 1 + L.diagonal() * tau ) )
        PVBP = D[N-1][0] * PVBP         
        payerSwaptionPrice = max( (fwdSwapRate - K), 0) * PVBP
        receiverSwaptionPrice = max( (K - fwdSwapRate), 0) * PVBP
        
        #print('payerSwaptionPrice: {0}'.format(payerSwaptionPrice))
        #print('receiverSwaptionPrice: {0}'.format(receiverSwaptionPrice))

        # STEP 7: COMPUTE NUMERAIRE-REBASED PAYMENT
        #for i in range(0, N):
            #FVprime[i]=V[i]*D[i][i]/D[N-1][i]
            #FVprime[i] = V[i]*DF[i]

        #print('swaptionPrice: {0}'.format(swaptionPrice))

        # STEP 8: COMPUTE IRS NPV
        VSwaption[nsim, ] = (payerSwaptionPrice, receiverSwaptionPrice)
        # end main MC loop

    # STEP 9: COMPUTE DISCOUNTED EXPECTED PAYOFF
    return sum(VSwaption) / NBSIMUS
    
    """
    sumSwaption = 0.0
    for nsim in range(0, NBSIMUS):
        sumSwaption = sumSwaption + VSwaption[nsim]
    payoff = sumSwaption / NBSIMUS
    return payoff
    """

def swaption(K, inForwardVec, corrMatrix, sigmaVec, dim):
    """
    Initializing
    Generates a new object to simulate from the
    multivariate normal distribution
    """
    # No of simulations
    NRUNS = 1000
    # No of steps per simulation
    STEPS = 250
    # Time step 
    DELTA = 1.0 / STEPS
    # P0 : Zero coupon bond for the desired time period
    P0 = 0.9473
    sum = 0

    # Looping through NRUNS MC simulations
    for run in range(0, NRUNS):
        #print("run: {0}".format(run))
        # Resets the forward vector
        forwardVec = np.copy(inForwardVec)

		# Looping through the time steps
        for i in range(0, STEPS):
            # Draws a new random vector
            mean = [0, 0, 0, 0, 0]
            sim = np.random.multivariate_normal(mean, corrMatrix, 1)[0]
			# Simulating each rate under the Q^alpha forward measure
            for j in range(0, dim):
                forward = forwardVec[j]
                sigma = sigmaVec[j]
                temp = math.log(forward)
                temp = temp - 0.5*sigma*sigma*DELTA
                temp = temp + sigma*sim[j]*math.sqrt(DELTA)
                temp2 = 0.0
                for k in range(0, j+1):
                    temp2 = temp2 + (corrMatrix[j, k]*sigmaVec[k]*forwardVec[k]) / (1 + forwardVec[k])

                temp2 = DELTA*sigmaVec[j]
                temp = temp + temp2
                forwardVec[j] = math.exp(temp)

		# Calculating the swap rate
        #print(forwardVec)
        temp = 0.0
        swap = 0.0
        temp2 = 1.0
        for i in range(0, dim):
            temp2 = 1.0
            for j in range(0, i+1):
                temp2 = 1.0 / (1.0 + forwardVec[j])
            temp = temp + temp2
        swap = (1.0 - temp2) / temp
        """
		Calculating the corresponding zero coupons
		The first forward rate is "dead" and can be
		considered as the LIBOR interest rate .
		The first zero coupons must therefore be treated individually
        """
        zero = 1 / (1 + forwardVec[0])
        zeroSum = zero
        for i in range(1, dim):
            zero = zero / (1 + forwardVec[i])
            zeroSum = zeroSum + zero

        swaption = 0
        if (swap-K > 0):
           swaption = (swap - K)*zeroSum
        else:
            swaption = 0
        sum = sum + swaption

    return P0*(sum / NRUNS)


class Swaption_tests(TestCase):    
    def test_01_swaption(self): 
        return
        print('swaption 1Y/1Y3M')
        """
        T0--->1Y--->2Y3M
        """
	    #Manually defines the correlation matrix
        corrMatrix = np.matrix([
            [1.0000000, 0.6036069, 0.4837154, 0.3906583, 0.2847411],
            [0.6036069, 1.0000000, 0.5462708, 0.4847784, 0.3399323],
            [0.4837154, 0.5462708, 1.0000000, 0.4631405, 0.2109093],
            [0.3906583, 0.4847784, 0.4631405, 1.0000000, 0.2191104],
            [0.2847411, 0.3399323, 0.2109093, 0.2191104, 1.0000000]
            ])

        df = pd.read_excel('PreProcessedMarketData.xlsx', sheetname='PreProcessedMarketData')      
        start_idx = df[df['TenorTi']=='T1Y'].index[0]    
        end_idx = df[df['TenorTi']=='T2Y3M'].index[0]    
        
        initialForwardVec = np.array(df['LT0Ti-3MTi'][start_idx+1:end_idx+1])
        initialSigmaVec = np.array(df['CapletVolatility'][start_idx+1:end_idx+1])

        Notional = 1.0
        Strike = 0.0236
        tau = 0.25
        Expiry = end_idx - start_idx
        timeSteps = 0.25
        NBSIMUS = 1000 # number of simulations 

        direction = -1.0
        Strikes = np.linspace(0.01, 0.05, 15)
        payoffList = []        
        for Strike in Strikes:
            payoff = monteCarloSimu(initialForwardVec, initialSigmaVec, corrMatrix, Notional, Strike, tau, Expiry, timeSteps, NBSIMUS, direction)
            payoffList.append(payoff)

        StrikeLevel = np.linspace(-0.011, 0.011, 15)
        plt.figure()
        plt.plot(Strikes, payoffList, label='Swaption Price')
        ax  = plt.gca()
        ax.set_ylim([-0.012, 0.012])
        plt.plot(Strikes, StrikeLevel, label='Strike Level')
        plt.legend()
        plt.show()            


    def test_02_swaption(self):
        return
        print('swaption 1Y/1Y3M')
        """
        T0--->1Y--->2Y3M
        """
	    #Manually defines the correlation matrix
        corrMatrix = np.matrix([
            [1.0000000, 0.6036069, 0.4837154, 0.3906583, 0.2847411],
            [0.6036069, 1.0000000, 0.5462708, 0.4847784, 0.3399323],
            [0.4837154, 0.5462708, 1.0000000, 0.4631405, 0.2109093],
            [0.3906583, 0.4847784, 0.4631405, 1.0000000, 0.2191104],
            [0.2847411, 0.3399323, 0.2109093, 0.2191104, 1.0000000]
            ])

        df = pd.read_excel('PreProcessedMarketData.xlsx', sheetname='PreProcessedMarketData')      
        start_idx = df[df['TenorTi']=='T1Y'].index[0]    
        end_idx = df[df['TenorTi']=='T2Y3M'].index[0]    
        
        initialForwardVec = np.array(df['LT0Ti-3MTi'][start_idx+1:end_idx+1])
        initialSigmaVec = np.array(df['CapletVolatility'][start_idx+1:end_idx+1])

        Notional = 1.0
        Strike = 0.0236
        tau = 0.25
        Expiry = end_idx - start_idx
        timeSteps = 0.25
        NBSIMUS = 1000 # number of simulations 

        direction = 1.0
        Strikes = np.linspace(0.01, 0.05, 15)
        payoffList = []        
        for Strike in Strikes:
            payoff = monteCarloSimu(initialForwardVec, initialSigmaVec, corrMatrix, Notional, Strike, tau, Expiry, timeSteps, NBSIMUS, direction)
            payoffList.append(payoff)

        StrikeLevel = np.linspace(0.0145, -0.0145, 15)
        plt.figure()
        plt.plot(Strikes, payoffList, label='Swaption Price')
        ax  = plt.gca()
        ax.set_ylim([-0.016, 0.016])
        plt.plot(Strikes, StrikeLevel, label='Strike Level')
        plt.legend()
        plt.show()            


    def test_03_swaption(self):        
        print('swaption 1Y/1Y3M')
        """
        T0--->1Y--->2Y3M
        """
	    #Manually defines the correlation matrix
        corrMatrix = np.matrix([
            [1.0000000, 0.6036069, 0.4837154, 0.3906583, 0.2847411],
            [0.6036069, 1.0000000, 0.5462708, 0.4847784, 0.3399323],
            [0.4837154, 0.5462708, 1.0000000, 0.4631405, 0.2109093],
            [0.3906583, 0.4847784, 0.4631405, 1.0000000, 0.2191104],
            [0.2847411, 0.3399323, 0.2109093, 0.2191104, 1.0000000]
            ])

        df = pd.read_excel('PreProcessedMarketData.xlsx', sheetname='PreProcessedMarketData')      
        start_idx = df[df['TenorTi']=='T1Y'].index[0]    
        end_idx = df[df['TenorTi']=='T2Y3M'].index[0]    
        
        initialForwardVec = np.array(df['LT0Ti-3MTi'][start_idx+1:end_idx+1])
        initialSigmaVec = np.array(df['CapletVolatility'][start_idx+1:end_idx+1])

        Notional = 1.0
        Strike = 0.0236
        tau = 0.25
        Expiry = end_idx - start_idx
        timeSteps = 0.25
        NBSIMUS = 1000 # number of simulations 

        Strikes = np.linspace(0.01, 0.05, 15)
        payoffPayerSwaptionList = []        
        payoffReceiverSwaptionList = []
        
        for Strike in Strikes:
            payer, receiver = MCSimu(initialForwardVec, initialSigmaVec, corrMatrix, Notional, Strike, tau, Expiry, timeSteps, NBSIMUS)
            payoffPayerSwaptionList.append(payer)
            payoffReceiverSwaptionList.append(receiver)

        plt.figure()
        plt.plot(Strikes, payoffPayerSwaptionList, label='Payer Swaption Payoff')
        plt.plot(Strikes, payoffReceiverSwaptionList, label='Receievr Swaption Payoff')

        #ax  = plt.gca()
        #ax.set_ylim([-0.012, 0.012])
        #plt.plot(Strikes, StrikeLevel, label='Strike Level')
        plt.legend()
        plt.show()       
        
        """
        Strikes = np.linspace(0.01, 0.05, 15)
        payoffList = []        
        for Strike in Strikes:
            payoff = MCSimu(initialForwardVec, initialSigmaVec, corrMatrix, Notional, Strike, tau, Expiry, timeSteps, NBSIMUS)
            payoffPayerSwaptionList.append(payoff)
        """ 

        """
        StrikeLevel = np.linspace(0.0145, -0.0145, 15)
        plt.figure()
        plt.plot(Strikes, payoffPayerSwaptionList, label='Swaption Price')
        ax  = plt.gca()
        ax.set_ylim([-0.016, 0.016])
        plt.plot(Strikes, StrikeLevel, label='Strike Level')
        plt.legend()
        plt.show()            
        """



    def test_04_swaption(self):
        return
        print("Swaption-Analytic Price")
        """
        Example
        Consider a 2-year payer swaption on a 4-year swap with semi-annual compounding. The
        forward swap rate of 7% starts 2 years from now and ends 6 years from now. The strike
        is 7.5%; the risk-free interest rate is 6%; the volatility of the forward starting swap rate is
        20% p.a.
        """
        fwdSwapRate = 0.07
        K = 0.075
        T = 2.0
        sigma = 0.2
        r = 0.06

        F = np.array(8*[fwdSwapRate])
        tau = 0.5
        P0 = exp(-r*T)
        PVBP = sum(tau/(1 + tau * F))
        
        black = LiteLibrary.bsm_call_value(fwdSwapRate, K, T, 0.0, sigma)
        print('black: {0}'.format(black))
        
        swaptionPrice  = P0*PVBP*LiteLibrary.bsm_call_value(fwdSwapRate, K, T, 0.0, sigma)

        print('P0: {0}'.format(P0))
        print('PVBP: {0}'.format(PVBP))

        S0 = float(fwdSwapRate)
      
        d1 = (log(S0 / K) + (0.0 + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        d2 = (log(S0 / K) + (0.0 - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))

        Nd1 = stats.norm.cdf(d1, 0.0, 1.0)
        Nd2 = stats.norm.cdf(d2, 0.0, 1.0)

        print('d1: {0}'.format(d1))        
        print('d2: {0}'.format(d2))        
        print('Nd1: {0}'.format(Nd1))        
        print('Nd2: {0}'.format(Nd2))        
        
        print('swaptionPrice: {0}'.format(swaptionPrice))        


