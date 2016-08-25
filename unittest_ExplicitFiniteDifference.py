"""
Date: Summer 2016
File name: ExplicitFiniteDifference.py
Version: 1 
Author: Abderrazak DERDOURI
Course: PYTHON 300: System Development with Python

Description: ExplicitFiniteDifference tests

Notes: Abderrazak_DERDOURI_Python300Final.pdf

Revision History:
"""

from unittest import TestCase
from ExplicitFiniteDifference import ExplicitFiniteDifference
from BlackScholes import BlackScholes
import random
import math
import numpy as np
import csv


def printHeaderBinaryOptionPriceFile(fileName):
    with open(fileName, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=';',
                               quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['Asset', 'Strike', 'IntRate', 'Vol', 'Expiration', 'NAS', 'BS Price', 'FDM Price', 'ErrWithBS', 'ErrWithBSPer'])
        
def printBinaryOptionPriceFile(fileName, Asset, Strike, InterestRate, Volatility, Expiration
                               , NumberAssetStep, BSPrice, FDMPrice, ErrorWithBS, ErrorWithBSPercentage):    		
    with open(fileName, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow([Asset, Strike, InterestRate, Volatility, Expiration, 
                            NumberAssetStep, BSPrice, FDMPrice, ErrorWithBS, ErrorWithBSPercentage])

class ExplicitFiniteDifferenceTests(TestCase):

    def test_01_ExplicitFiniteDifference(self):
        """
        Print prices using both BS and FDM
        Examples are taken from verified literature
        """
        Asset = 100
        Strike = 100.0
        InterestRate = 0.05
        Volatility = 0.2
        Expiration = 1.0
        Dividend = 0.0
        
        NumberAssetStep = 80
        
        """
        Price with Black and Scholes 
        """	    
        BS = BlackScholes(Asset, Strike, InterestRate, Volatility, Expiration)
        BS.price()
        self.assertEqual(round(BS.getPrice(), 2), 0.53)
        
        """
        Price with a Backward Scheme Explicit Finite Difference 
        """        
        FDM = ExplicitFiniteDifference(Asset, Strike, Expiration, InterestRate, Dividend, Volatility, NumberAssetStep)
        FDM.price()
        self.assertEqual(round(FDM.getPrice(), 2), 0.52)
        
    def test_02_ExplicitFiniteDifference(self):
        """
        Varying the nummber of asset step and Expiration
        """
        Asset = 110
        Strike = 100.
        InterestRate = 0.05
        Volatility = 0.2
        Expiration = 1.0
        Dividend = 0.0
        NumberAssetStep = 80
        
        printHeaderBinaryOptionPriceFile("binaryCallFDMPrice.csv")

        Asset = 60
        while Asset <= 110:
            Expiration = 0.2
            while Expiration <= 1.0:			    
                """
                Price with Black and Scholes
                """
                BS = BlackScholes(Asset, Strike, InterestRate, Volatility, Expiration)
                BS.price()

                """
                Price with a Backward Scheme Explicit Finite Difference
                """
                FDM = ExplicitFiniteDifference(Asset, Strike, Expiration, InterestRate, Dividend, Volatility, NumberAssetStep)
                FDM.price()
                
                err = BS.getPrice() - FDM.getPrice()
                self.assertGreater(BS.getPrice(), 0.0)
                printBinaryOptionPriceFile("binaryCallFDMPrice.csv", Asset, Strike, InterestRate
                                           , Volatility, Expiration, NumberAssetStep, BS.getPrice(), FDM.getPrice()
                                           , err, math.fabs(err / BS.getPrice()))
                Expiration += 0.2
            Asset += 10
        