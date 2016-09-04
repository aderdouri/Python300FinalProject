"""
Date: Summer 2016
File name: unittest_MonteCarlo.py
Version: 1 
Author: Abderrazak DERDOURI
Course: PYTHON 300: System Development with Python

Description: MonteCarlo tests

Notes: Abderrazak_DERDOURI_Python300Final.pdf

Revision History:
"""

from unittest import TestCase
from BlackScholes import BlackScholes
from MonteCarlo import MonteCarlo
from FilesProcessor import writeToFile
import math

class MonteCarloTests(TestCase):

    def test_01_MonteCarlo(self):
        """
        Varying the number of simulations
        """
        Asset = 100.0
        Strike = 100.0
        InterestRate = 0.05
        Volatility = 0.2
        Expiration = 1.0
        NumberAssetStep = 100
        TimeStep = Expiration / NumberAssetStep
        NumbreOfSimulation = 1000
        
        """
        Examples are taken from verified literature
        Price with Black and Scholes and Monte Carlo and print the differences
        """ 

        listOfValuesList = []
        for i in range(0, 10):
            BS = BlackScholes(Asset, Strike, InterestRate, Volatility, Expiration)
            BS.price()
            
            MC = MonteCarlo(Asset, Strike, InterestRate, Volatility, Expiration, TimeStep, NumberAssetStep, NumbreOfSimulation)
            MC.price()
            
            err = BS.getPrice() - MC.getPrice()            
            self.assertGreater(BS.getPrice(), 0.0)

            valuesList = [Asset, Strike, InterestRate, Volatility, Expiration, TimeStep, NumberAssetStep
                          , NumbreOfSimulation, BS.getPrice(), MC.getPrice(), err, math.fabs(err/BS.getPrice())]
            
            listOfValuesList.append(valuesList)

            NumbreOfSimulation += 1000

        headerList = ['Asset', 'Strike', 'IntRate', 'Vol', 'Expiration', 'TimeStep', 'NumberAssetStep'
                      , 'NumbreOfSimulation', 'BSPrice', 'MCPrice', 'ErrorWithBS', 'ErrWithBSPer']

        writeToFile("binaryCallMCPriceTest1.csv", headerList)

        for valuesList in listOfValuesList:
            writeToFile("binaryCallMCPriceTest1.csv", valuesList)
            
    def test_02_MonteCarlo(self):
        """
        Varying the time step size
        """
        Asset = 100.0
        Strike = 100.0
        InterestRate = 0.05
        Volatility = 0.2
        Expiration = 1.0
        NumberAssetStep = 100
        TimeStep = Expiration / NumberAssetStep
        NumbreOfSimulation = 10000
        
        listOfValuesList = []

        for i in range(0, 10):
            """
            Price with Black and Scholes and Monte Carlo and print the differences
            """                    
            BS = BlackScholes(Asset, Strike, InterestRate, Volatility, Expiration)
            BS.price()
            
            MC = MonteCarlo(Asset, Strike, InterestRate, Volatility, Expiration, TimeStep, NumberAssetStep, NumbreOfSimulation)
            MC.price()
            
            err = BS.getPrice() - MC.getPrice()
            self.assertGreater(BS.getPrice(), 0.0)

            valuesList = [Asset, Strike, InterestRate, Volatility, Expiration, TimeStep, NumberAssetStep
                          , NumbreOfSimulation, BS.getPrice(), MC.getPrice(), err, math.fabs(err/BS.getPrice())]
            
            listOfValuesList.append(valuesList)

            NumberAssetStep += 100
            TimeStep = Expiration / NumberAssetStep

        headerList = ['Asset', 'Strike', 'IntRate', 'Vol', 'Expiration', 'TimeStep', 'NumberAssetStep'
                      , 'NumbreOfSimulation', 'BSPrice', 'MCPrice', 'ErrorWithBS', 'ErrWithBSPer']

        writeToFile("binaryCallMCPriceTest2.csv", headerList)

        for valuesList in listOfValuesList:
            writeToFile("binaryCallMCPriceTest2.csv", valuesList)
