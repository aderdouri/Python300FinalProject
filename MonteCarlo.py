"""
Date: Summer 2016
File name: MonteCarlo.py
Version: 1 
Author: Abderrazak DERDOURI
Course: PYTHON 300: System Development with Python

Description: Price a binary call option using the Monte Carlo method

Notes:

Revision History:
"""

import random
import statistics as st
import numpy as np

class MonteCarlo(object):
    """
    Init a standard option parameters and Monte Carlo defaulted paramaters
    """
    def __init__(self, Asset, Strike, InterestRate, Volatility, Expiration, tStep=0.01, NAS=1000, NbSimu=10000):
        self.Asset = Asset
        self.Strike = Strike
        self.InterestRate = InterestRate
        self.Volatility = Volatility 
        self.Expiration = Expiration
        self.tStep = tStep
        self.NAS = NAS
        self.NbSimu = NbSimu
        self.Price = 0.0

    def price(self):
        """
        give the price of a standard option usign Monte Carlo algorithm and simulation
        """
        PathMatrix = np.zeros(shape=(self.NAS, self.NbSimu))
        AssetPriceAtExpiration = [] # Size of NbSimu
        for nbSimu in range(0, self.NbSimu):
            CurrentAssetPrice = self.Asset
            for tStepCounter in range(0, self.NAS):
                phi = self.rnorm()
                CurrentAssetPrice *= (1 + self.InterestRate*self.tStep + self.Volatility*phi*np.math.sqrt(self.tStep) + 0.5*self.Volatility*self.Volatility* (phi*phi - 1)*self.tStep)
                PathMatrix[tStepCounter, nbSimu] = CurrentAssetPrice
        """
        Calculate a Binary Call price
        """
        PayOffVect = []
        for nbSimu in range(0, self.NbSimu):
            PayOffVect.append(1.0 if PathMatrix[self.NAS - 1, nbSimu] > self.Strike else 0.0)
        self.Price = self.discountFactor()*st.mean(PayOffVect);

    def rnorm(self, mu=0.0, sigma=1.0):
        """
        generate a gaussian random variable
        """
        return random.gauss(mu, sigma)

    def mean(self, alist):
        """
        excpected value or mean of a vector of variables
        """
        return (st.mean(aList))
        
    def discountFactor(self):
        """
        Continuous discount factor
        """
        return np.math.exp(-self.InterestRate*self.Expiration)

    def getPrice(self):
        """
        Price getter
        """
        return self.Price