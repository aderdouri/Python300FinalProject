"""
Date: Summer 2016
File name: BlackScholes.py
Version: 1 
Author: Abderrazak DERDOURI
Course: PYTHON 300: System Development with Python

Description: Price a standard option using Black-Scholes formula

Notes: Abderrazak_DERDOURI_Python300Final.pdf

Revision History:
"""

import math

#Use stats.norm.cdf

def normcdf(x):
    """
    Calculate cumulative normal distribution
    """
    k = 1.0 / (1.0 + 0.2316419*x)
    k_sum = k*(0.319381530 + k*(-0.356563782 + k*(1.781477937 + k*(-1.821255978 + 1.330274429*k))))
    if (x >= 0.0): 
        return (1.0 - (1.0 / (math.pow(2 * math.pi, 0.5)))*math.exp(-0.5*x*x) * k_sum)
    else:
        return 1.0 - normcdf(-x)

class BlackScholes(object):
    """
    Init a standard option parameters in the Black-Scholes formula
    """
    def __init__(self, Asset, Strike, InterestRate, Volatility, Expiration, ValueDate=0.0):
        self.Asset = Asset 
        self.Strike = Strike 
        self.InterestRate = InterestRate
        self.Volatility = Volatility 
        self.Expiration = Expiration
        self.ValueDate = ValueDate
        self.Price = 0.0

    def price(self):
        """
        give the price of a standard option usign Black-Scholes formula
        """
        self.Price = self.discountFactor()*normcdf(self.d2())

    def d1(self):
        """
        see page 1 from the pdf notes for d1 and d2 definition
        """
        return ( (math.log(self.Asset / self.Strike) 
                  + (self.InterestRate + 0.5 * self.Volatility*self.Volatility) 
                  * (self.Expiration - self.ValueDate)) 
                / (self.Volatility * math.sqrt(self.Expiration - self.ValueDate)) )

    def d2(self):
        """
        see page 1 from the pdf notes for d1 and d2 calculation
        """
        return ( self.d1() - math.sqrt(self.Expiration - self.ValueDate)*self.Volatility )

    def discountFactor(self):
        """
        Continuous discount factor
        """
        return math.exp(-self.InterestRate*self.Expiration)

    def getPrice(self):
        """
        Price getter
        """
        return self.Price