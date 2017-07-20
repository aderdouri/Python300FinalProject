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
import numpy as np
from datetime import date
import re
#pip install xlrd
from sys import exit
from collections import OrderedDict


from scipy.optimize import root, fsolve
import matplotlib.pyplot as plt
import xlsxwriter


def interpolateDF(df1, df2,  tau, tau1, tau2):
    """
    log-linear interpolation of the discounting factors.
    """
    return ( np.power(df2, (tau - tau1).days / (tau2 - tau1).days) 
            * np.power(df1, (tau2 - tau).days / (tau2 - tau1).days) )

def interpolateVol(sigma1, sigma2,  tau, tau1, tau2):
    """
    linear interpolation of volatilities.
    """
    return ( (tau2-tau).days/(tau2-tau1).days*sigma1 + (tau-tau1).days/(tau2-tau1).days*sigma2 )

def getNextDate(Date, Period):
    if (5==Date.weekday()):
        day += 2
    elif (6==Date.weekday()):
        day += 1
    year = Date.year
    month = Date.month
    day = Date.day

    if ('3M'==Period):
        return pd.datetime(year, month + 3, day)
    elif ('6M'==Period):
        return pd.datetime(year, month + 6, day)
    elif ('9M'==Period):
        return pd.datetime(year, month + 9, day)
    elif ('Y'==Period):
        return pd.datetime(year + 1 , month, day)
    return Date

def day_count(start_date, end_date):
    """Returns number of days between start_date and end_date, using Actual/360 convention"""
    return (end_date - start_date).days

def year_fraction365(start_date, end_date):
    """Returns fraction in years between start_date and end_date, using Actual/365 convention"""
    return day_count(start_date, end_date) / 365.0


def year_fraction(start_date, end_date):
    """Returns fraction in years between start_date and end_date, using Actual/360 convention"""
    return day_count(start_date, end_date) / 360.0

def year_fraction365(start_date, end_date):
    """Returns fraction in years between start_date and end_date, using Actual/365 convention"""
    return day_count(start_date, end_date) / 365.0

def bsm_call_value(S0, K, T, r, sigma):
    """
    Valuation of European call option in BSM model.
    Analytical formula.
    Parameters
    ==========
    S0 : float
    initial stock/index level
    K : float
    strike price
    T : float
    maturity date (in year fractions)
    r : float
    constant risk-free short rate
    sigma : float
    volatility factor in diffusion term
    Returns
    =======
    value : float
    present value of the European call option
    """
    from math import log, sqrt, exp
    from scipy import stats

    S0 = float(S0)
    d1 = (log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = (log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    value = (S0 * stats.norm.cdf(d1, 0.0, 1.0) - K * exp(-r * T) * stats.norm.cdf(d2, 0.0, 1.0))

    # stats.norm.cdf —> cumulative distribution function
    # for normal distribution
    return value

def capletValue(DF_t_Ti, tau_i, L_t_Ti_1_Ti, K, Ti_1, r, sigma):
    """
    Caplet value for Notional = 1
    """
    return DF_t_Ti*tau_i*bsm_call_value(L_t_Ti_1_Ti, K, Ti_1, r, sigma)

def liborRate(DF1, DF2, delta):
    """
    DF1 = B(T0, Tn-1), DF2 = B(T0, Tn)
    delta = delta(n-1,n)
    """
    return (DF1/DF2-1)/delta

def date_diff(row):
    #print(row['PrevDate'], row['Date'])
    #print(year_fraction(row['PrevDate'], row['Date']))
    return year_fraction(row['PrevDate'], row['Date'])


def writeDataFrame(df):  
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter('PreProcessedMarketData.xlsx', engine='xlsxwriter')
    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name='PreProcessedMarketData')
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
