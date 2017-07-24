"""
Date: Monday, 24 July 2017
File name: LiteLibrary.py
Version: 1 
Author: Abderrazak DERDOURI
Subject: CQF Final Project

Description: Implement Black formula and interpolation methods

Notes: AbderrazakDerdouriCQFFinalProject.pdf

Revision History:
"""

import pandas as pd
import numpy as np
from datetime import date
import re
from collections import OrderedDict
from math import log, sqrt, exp
from scipy import stats

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
    """
    Add Period of 3M, 6M, 9M or xY to Date and return Date + Period
    Skip Saturday and Sunday
    """
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
    """
    Returns number of days between start_date and end_date
    using Actual/360 convention
    """
    return (end_date - start_date).days

def year_fraction365(start_date, end_date):
    """
    Returns fraction in years between start_date and end_date
    using Actual/365 convention
    """
    return day_count(start_date, end_date) / 365.0


def year_fraction(start_date, end_date):
    """
    Returns fraction in years between start_date and end_date
    using Actual/360 convention
    """
    return day_count(start_date, end_date) / 360.0

def year_fraction365(start_date, end_date):
    """
    Returns fraction in years between start_date and end_date
    using Actual/365 convention
    """
    return day_count(start_date, end_date) / 365.0

def normcdf(d):
    """
    Returns cumulative distribution function for normal distribution
    """
    return stats.norm.cdf(d, 0.0, 1.0)

def bsm_call_value(S0, K, T, r, sigma):
    """
    Valuation of European call option in BSM model. Analytical formula.
    Parameters
    ==========
    S0 : initial stock/index level (float)
    K : strike price (float)
    T : maturity date (in year fractions) (float)
    r : constant risk-free short rate
    sigma : volatility factor in diffusion term (float)    
    Returns
    =======
    value : present value of the European call option (float)    
    """
    S0 = float(S0)
    d1 = (log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = (log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    value = (S0 * normcdf(d1) - K * normcdf(d2))

    return value

def capletValue(DF_t_Ti, tau_i, L_t_Ti_1_Ti, K, Ti_1, r, sigma):
    """
    Caplet value for Notional = 1
    """
    return DF_t_Ti*tau_i*bsm_call_value(L_t_Ti_1_Ti, K, Ti_1, r, sigma)

def blackCapletValue(DF_t_Ti, tau_i, L_t_Ti_1_Ti, K, Ti_1, sigma):
    """
    Black Analytic formula for pricing Caplet
    """
    return DF_t_Ti*tau_i*bsm_call_value(L_t_Ti_1_Ti, K, Ti_1, 0.0, sigma)

def liborRate(DF1, DF2, delta):
    """
    DF1 = D(T0, Tj-1), DF2 = D(T0, Tj)
    delta = delta(j-1, j)
    """
    return (DF1/DF2-1)/delta

def date_diff(row):
    """
    Returns diff in days
    """
    return year_fraction(row['PrevDate'], row['Date'])


def writeDataFrame(newdf, newSheetName): 
    """
    Write a new Pandas DataFrame as new sheet to an existing xlsx file
    """
    df = pd.ExcelFile('lmmData.xlsx')
    sheetNames = df.sheet_names  # get all sheet names       
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter('lmmData.xlsx', engine='xlsxwriter')
    for sheetName in sheetNames:
        dfSheet = df.parse(sheetName)  # read a specific sheet to DataFrame
        dfSheet.to_excel(writer, sheet_name=sheetName)
        
    # Write each dataframe to a different worksheet.
    newdf.to_excel(writer, sheet_name=newSheetName)
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
