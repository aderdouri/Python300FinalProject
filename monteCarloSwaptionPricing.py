"""
Date: Monday, 24 July 2017
File name: monteCarloSwaptionPricing.py
Version: 1 
Author: Abderrazak DERDOURI
Subject: CQF Final Project

Description: Price a cap using MonteCarlo simulation 
             Plot MonteCarlo simulated price vs Analytical price

Notes: AbderrazakDerdouriCQFFinalProject.pdf

Run: python -m unittest monteCarloSwaptionPricing.MCSwaptionPricingTests.testMCSwaptionPricing

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
import volCorrelationGen 


def MCSwaptionSimu(initialForwardVec, initialSigmaVec, corrMatrix, Notional, Strike, tau, Expiry, timeSteps, NBSIMUS):
    """
    Monte Carlo simulations as described in paragraph 3.5.1
    """
    # STEP 1: INITIALISE INPUTS
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
   
	# STEP 2: INITIALISE INITIAL FORWARD RATE
    for i in range(N):
        L[i][0] =  initialForwardVec[i]
        
    mean = N*[0]      
    # start main MC loop
    for nsim in range(0, NBSIMUS):
        # Draws a new random vector
        dW = np.random.multivariate_normal(mean, corrMatrix)

        # STEP 3: COMPUTE FORWARD RATES TABLEAU
        for n in range(0, N-1):
            for i in range(n+1, N):
                drift_sum = 0.0
                for k in range(i+1, N):
                    drift_sum = drift_sum + (tau*corrMatrix[n, k]*sigma[k]*L[k][n])/(1+tau*L[k][n])
                L[i][n+1] = L[i][n]*math.exp((-drift_sum*sigma[n] - 0.5*sigma[n]*sigma[n])*dT + sigma[n]*dW[n+1])
            
        # STEP 4: COMPUTE DISCOUNT RATES TABLEAU
        for n in range(0, N):
            for i in range(n+1, N+1):
                df_prod = 1.0
                for k in range(n, i):
                    df_prod = df_prod * 1/(1+tau*L[k][n])
                D[i-1][n] = df_prod

        
        #Calculate the forward Swap rate
        numerator =  1 - np.prod( 1/(1 + tau * L.diagonal()) )
        
        denominator = 0.0
        for j in range(0, N):
            denominator = denominator + tau * np.prod(1/(1+L.diagonal()[0:j+1]))

        fwdSwapRate = numerator / denominator

        PVBP = sum ( tau / ( 1 + 1 + L.diagonal() * tau ) )
        PVBP = D[N-1][0] * PVBP         
        payerSwaptionPrice = max( (fwdSwapRate - K), 0) * PVBP
        receiverSwaptionPrice = max( (K - fwdSwapRate), 0) * PVBP
        
        # STEP 5: COMPUTE Swaption NPV
        VSwaption[nsim, ] = (payerSwaptionPrice, receiverSwaptionPrice)
        # end main MC loop

    # STEP 6: EXPECTED PAYOFF
    return sum(VSwaption) / NBSIMUS
    
class MCSwaptionPricingTests(TestCase):    
    def testMCSwaptionPricing(self):        
        """
        Pricing of Swaption using Monte Carlo and volatilities from a calibrated LMM
        T0--->1Y--->2Y3M
        """

        maturity_grid = [1, 1.25, 1.5, 2, 2.25]
        # Make data.
        beta = 0.05
        #maturity_grid = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
        #maturity_grid = np.linspace(1, 30, 30) #.reshape(-1, 1)
        corrMatrix = volCorrelationGen.genExponentialCorr(beta, maturity_grid)
        print(corrMatrix)

        beta, rho_inf = 0.1, 0.4
        corr_matrix = volCorrelationGen.genParametricCorr(beta, rho_inf, maturity_grid)
        print(corr_matrix)

        df = pd.read_excel('lmmData.xlsx', sheetname='lmmCalibration')      
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
            payer, receiver = MCSwaptionSimu(initialForwardVec, initialSigmaVec, corrMatrix, Notional, Strike, tau, Expiry, timeSteps, NBSIMUS)
            payoffPayerSwaptionList.append(payer)
            payoffReceiverSwaptionList.append(receiver)

               
        # Plot receiver
        StrikeLevel = np.linspace(-0.011, 0.011, 15)
        plt.figure()
        plt.plot(Strikes, payoffReceiverSwaptionList, label='Swaption Price')
        ax  = plt.gca()
        ax.set_ylim([-0.012, 0.012])
        plt.plot(Strikes, StrikeLevel, label='Strike Level')
        plt.legend()
        plt.title('Receiver SwaptionPrice StrikeLevel')
        plt.xlabel('Strike %')
        plt.ylabel('Swaption Price')
        plt.savefig('ReceiverSwaptionPriceStrikeLevel')
        plt.show()      

        
        # Plot Payer
        StrikeLevel = np.linspace(0.0145, -0.0145, 15)
        plt.figure()
        plt.plot(Strikes, payoffPayerSwaptionList, label='Swaption Price')
        ax  = plt.gca()
        ax.set_ylim([-0.016, 0.016])
        plt.plot(Strikes, StrikeLevel, label='Strike Level')
        plt.legend()
        plt.title('Payer SwaptionPrice StrikeLevel')
        plt.xlabel('Strike %')
        plt.ylabel('Swaption Price')
        plt.savefig('PayerSwaptionPriceStrikeLevel')
        plt.show()                   

        # Plot both Payer and Receiver
        plt.figure()
        plt.plot(Strikes, payoffPayerSwaptionList, label='Payer Swaption Payoff')
        plt.plot(Strikes, payoffReceiverSwaptionList, label='Receievr Swaption Payoff')
        plt.legend()
        plt.title('Payer SwaptionPrice StrikeLevel')
        plt.xlabel('Strike %')
        plt.ylabel('Swaption Price')


        plt.savefig('PayerSwaptionAndReceiverSwaption')
        plt.show()       
        