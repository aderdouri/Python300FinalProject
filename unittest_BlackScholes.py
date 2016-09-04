"""
Date: Summer 2016
File name: unittest_BlackScholes.py
Version: 1 
Author: Abderrazak DERDOURI
Course: PYTHON 300: System Development with Python

Description: BlackScholes tests

Notes: Abderrazak_DERDOURI_Python300Final.pdf

Revision History:
"""


from unittest import TestCase
from BlackScholes import normcdf
from BlackScholes import BlackScholes

class BlackScholesTests(TestCase):

    def test_normcdf(self):
        """
        test from known standard normal table
        """
        self.assertEqual(round(normcdf(0), 2), 0.5)
        self.assertEqual(round(normcdf(1.64), 2), 0.95)
        self.assertEqual(round(normcdf(2.33), 2), 0.99)

    def test_01_BS(self):
        """
        Examples are taken from verified literature
        Calculate the value of a derivative that pays off $100, and paramaters described bellow
        With these values, we expected to find that d2 = −0.1826, and N(d2)=0.4276. 
        Therefore, the derivative has a value 0.417 = 41.7$.
        """
        Asset = 480.0
        Strike = 500.0
        InterestRate = 0.05
        Volatility = 0.2
        Expiration = 0.5 # in year
        ValueDate = 0.0

        BS = BlackScholes(Asset, Strike, InterestRate, Volatility, Expiration, ValueDate)
        BS.price()

        self.assertEqual(round(BS.d2(), 3), -0.183)
        self.assertEqual(round(normcdf(BS.d2()), 3), 0.428)
        self.assertEqual(round(BS.getPrice(), 3), 0.417)