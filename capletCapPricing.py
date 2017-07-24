"""
Date: Monday, 24 July 2017
File name: 
Version: 1 
Author: Abderrazak DERDOURI
Subject: CQF Final Project

Description: In this module we test a basic pricing of caplet and cap using Black formula

Notes: AbderrazakDerdouriCQFFinalProject.pdf

Run: python -m unittest capletCapPricing.CplCapPricingTests.testCplPricing
     python -m unittest capletCapPricing.CplCapPricingTests.testCapPricing

Revision History:
"""

import pandas as pd
import LiteLibrary
from unittest import TestCase

class CplCapPricingTests(TestCase):
    def testCplPricing(self):
        """
        Caplet pricing using balck formula
        """
        print('Caplet pricing using Black formula')
        print('==================================')
        dateT0 = pd.to_datetime('25-01-2005')
        dateTj_1 = pd.to_datetime('25-01-2006')
        dateTj = pd.to_datetime('25-04-2006')
        deltaT0_Tj_1 = LiteLibrary.year_fraction(dateT0, dateTj_1)
        deltaTj_1Tj = LiteLibrary.year_fraction(dateTj_1, dateTj)
    
        D_T0_Tj_1 = 0.9774658
        D_T0_Tj = 0.9712884

        Strike = 0.02361
        sigma_caplet = 0.2015

        F_T0_Tj_1_Tj = LiteLibrary.liborRate(D_T0_Tj_1, D_T0_Tj, deltaTj_1Tj)        

        capletPrice = LiteLibrary.blackCapletValue(D_T0_Tj, deltaTj_1Tj, F_T0_Tj_1_Tj, Strike, deltaT0_Tj_1, sigma_caplet)
        print('capletPrice: {0}'.format(capletPrice))

    def testCapPricing(self):        
        """
        Cap pricing as a sum of caplets prices using balck formula
        """
        print('Cap pricing using Black formula')
        print('==================================')
        Strike = 0.02
        N = 1000000
        sigma_cap = 0.4468

        sigma_caplet = [0.5129, 0.4874, 0.4658, 0.4339, 0.4339]
        tau = [0.5, 0.5, 0.5, 0.5, 0.5]
        fwdRate = [0.010125, 0.010796, 0.012189, 0.013995, 0.016210]
        DF = [0.99462, 0.99375, 0.99112, 0.98648, 0.98183]
        Ti_1 = [0.5096, 1.0, 1.5068, 2.0055, 2.5041]

        capValueSumCaplets = 0.0
        length = len(fwdRate)
        for i in range(length):
            capletValue = N*LiteLibrary.blackCapletValue(DF[i], tau[i], fwdRate[i], Strike, Ti_1[i], sigma_caplet[i])
            print('capletValue: {0}'.format(capletValue))
            capValueSumCaplets = capValueSumCaplets + capletValue

        print('capValueSumCaplets: {0}'.format(capValueSumCaplets))
        print('------------------------------')

        capValue = 0.0
        for i in range(length):
            capletValue = N*LiteLibrary.blackCapletValue(DF[i], tau[i], fwdRate[i], Strike, Ti_1[i], sigma_cap)
            print('capletValue: {0}'.format(capletValue))
            capValue = capValue + capletValue

        print('capValueSumCaplets: {0}'.format(capValue))     
        
        self.assertEqual(round(capValueSumCaplets), round(capValue))