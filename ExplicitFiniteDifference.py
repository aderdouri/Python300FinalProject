"""
Date: Summer 2016
File name: ExplicitFiniteDifference.py
Version: 1 
Author: Abderrazak DERDOURI
Course: PYTHON 300: System Development with Python

Description: Price a binary call option using the Explicit Finite Difference Method

Notes: Abderrazak_DERDOURI_Python300Final.pdf

Revision History:
"""

import numpy as np

class ExplicitFiniteDifference(object):
    def __init__(self, Asset, Strike, Expiration, InterestRate, dividend, Volatility, NAS=1000):
        """
        Init parameters for a binary call option
        """
        self.Asset = Asset 
        self.Strike = Strike 
        self.Expiration = Expiration
        self.InterestRate = InterestRate
        self.dividend  = dividend 
        self.Volatility = Volatility 
        self.NAS = NAS
        self.Price = 0.0

    def alpha(self, n, dt):
        """
        Alpha in the backward schema from the Finite Difference Method (pdf page 2)
        """
        return (0.5 * (self.Volatility*self.Volatility) * (n*n) * dt - 0.5 * n * self.InterestRate * dt)

    def beta(self, n, dt):
        """
        Beta in the backward schema from the Finite Difference Method (pdf page 2)
        """
        return (1 - (self.Volatility*self.Volatility) * (n*n) * dt - self.InterestRate * dt)

    def gamma(self, n, dt):
        """
        Gamma in the backward schema from the Finite Difference Method (pdf page 2)
        """
        return (0.5 * (self.Volatility*self.Volatility) * (n*n) * dt + 0.5 * n * self.InterestRate  * dt)

    def price(self):
        """
	    Price a binary call option using the Finite Difference Method
	    """
        NAS = self.NAS               # Number of asset steps
        Asset = self.Asset           # Asset initial value
        Strike = self.Strike         # Strike 
        Expiration = self.Expiration # Experiy date
        Vol = self.Volatility        # Volatility
        Int_Rate = self.InterestRate # Interest Rate

        dS = Asset / NAS
        NAS = int(np.floor(Strike / dS) * 2)
        SGridPt = int(Asset / dS)
        dt = dS*dS / (Vol*Vol * 4 * Strike*Strike)
        NTS = int(np.floor(Expiration / dt) + 1)
        dt = Expiration / NTS
        
        V = np.zeros(shape=(NAS+1, NTS+1))
        
        for n in range(1, 1+NAS):
            V[n, NTS] = ( 1.0 if (n * dS > Strike) else 0.0 )

        """
        At S = 0
	    V(m-1, 0) = (1 - r dt) * V(m,0)
        """
        for  m in range(NTS, 1, -1):
            V[1, m-1] = (1 - Int_Rate * dt) * V[1, m]

        for m in range(NTS, m >= 1, -1):
            for n in range(2, NAS):
                V[n, m-1] = (self.alpha(n, dt) * V[n-1, m] + self.beta(n, dt) * V[n, m] + self.gamma(n, dt) * V[n + 1, m])
                """
                Satisfy boundary conditions
                """
            V[NAS, m-1] = (self.alpha(NAS, dt) - self.gamma(NAS, dt)) * V[NAS - 1, m] + (self.beta(NAS, dt) + 2 * self.gamma(NAS, dt)) * V[NAS, m]
            
        self.Price = V[SGridPt, 1]

    def getPrice(self):
        """
        Price getter
        """
        return self.Price