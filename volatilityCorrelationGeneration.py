"""
Date: Monday, 24 July 2017
File name: volatilityCorrelationGeneration
Version: 1 
Author: Abderrazak DERDOURI
Subject: CQF Final Project

Description: 

Notes: AbderrazakDerdouriCQFFinalProject.pdf

Run: python -m unittest volatilityCorrelationGeneration.VolCorrGenTests.test01VolCorrGen
     python -m unittest volatilityCorrelationGeneration.VolCorrGenTests.test01VolCorrGen
     python -m unittest volatilityCorrelationGeneration.VolCorrGenTests.test01VolCorrGen
    
Revision History:
"""

from unittest import TestCase
import math
import numpy as np
import pandas as pd
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def generateExponentialParametricCorr(beta, maturity_grid):
    grid_length = len(maturity_grid)
    grid = range(grid_length)
    #Create a null correlation matrix
    corr_matrix = np.zeros ( (grid_length, grid_length) )
    for i in grid:
        for j in grid:
            corr_matrix[i, j] = np.exp(-beta*abs(maturity_grid[i]-maturity_grid[j]))
    return corr_matrix

def generateParametricCorr(alpha, beta, rho_inf, maturity_grid):
    """
    Function which generate a matrix correlation 
    using the double exponential parameterization
    @var alpha: alpha parameter 
    @var beta: alpha parameter
    @var rho_inf: alpha parameter
    @var maturity_grid: represent a list containing the forward rate
    maturities which we are going to model.
    For example, if we want to model the semi-annual forward rate
    maturing between 1 and 5 years from now, we will set 
    maturity_grid = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    """
    grid_length = len(maturity_grid)
    grid = range(grid_length)
    #Create a null correlation matrix
    corr_matrix = np.zeros ( (grid_length, grid_length) )
    print(corr_matrix)

    for i in grid:
        for j in grid:
            first_e = np.exp(-alpha*abs(maturity_grid[i]-maturity_grid[j]))
            second_e = np.exp(-beta*abs(maturity_grid[i]-maturity_grid[j]))
            corr_matrix[i, j] = (rho_inf + (1-rho_inf)*first_e*second_e)

    return corr_matrix


class VolCorrGenTests(TestCase):
    def test01VolCorrGen(self):
        print("Exponential Parametric Matrix Correlation")

        df = pd.read_excel('marketData.xlsx', sheetname='processedMarketData')      
        start_idx = df[df['TenorTi']=='T1Y'].index[0]    
        end_idx = df[df['TenorTi']=='T2Y3M'].index[0]    
        
        maturity_grid = [1, 1.25, 1.5, 2, 2.25]
        # Make data.
        beta = 0.05
        #maturity_grid = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
        #maturity_grid = np.linspace(1, 30, 30) #.reshape(-1, 1)
        corr_matrix = generateExponentialParametricCorr(beta, maturity_grid)
        print(corr_matrix)

    def test02VolCorrGen(self):
        """
        Demonstrates plotting a 3D surface colored with the coolwarm color map.
        The surface is made opaque by using antialiased=False.
        Also demonstrates using the LinearLocator and custom formatting for the
        z axis tick labels.
        """

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Make data.
        beta = 0.05
        #maturity_grid = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
        maturity_grid = np.linspace(0.5, 30, 30) #.reshape(-1, 1)
        corr_matrix = generateExponentialParametricCorr(beta, maturity_grid)
        print(corr_matrix)

        x=range(0, 30)
        y=range(0, 30)
        x, y = np.meshgrid(x, y)
        z = corr_matrix

        # Plot the surface.
        surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        #surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
        # Customize the z axis.
        
        ax.set_zlim(0.1, 1.0)
        #ax.zaxis.set_major_locator(LinearLocator(10))
        #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        #plt.title('Receiver SwaptionPrice StrikeLevel')
        ax.set_xlabel('Tl(Years)')
        ax.set_ylabel('Tk(Years)')
        ax.set_zlabel('Correlation')
        plt.title('Exponential Parametric Correlation')
        #plt.savefig('ExponentialParametricCorrelation')
        plt.show()   

    def test03VolCorrGen(self):        
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Make data.
        alpha = 0.1
        beta = 0.1
        rho_inf = 0.4
        #maturity_grid = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
        maturity_grid = np.linspace(0.5, 30, 30)#.reshape(-1, 1)
        corr_matrix = generateParametricCorr(alpha, beta, rho_inf, maturity_grid)
        #print(corr_matrix)

        x = range(0, 30)
        y = range(0, 30)
        x, y = np.meshgrid(x, y)
        z = corr_matrix

        # Plot the surface.
        surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        #surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)

        # Customize the z axis.
        ax.set_zlim(0.01, 1.0)
        ax.set_xlim(0.0, 30.0)
        ax.set_ylim(0.0, 30.0)
        #ax.zaxis.set_major_locator(LinearLocator(10))
        #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        #Forward-forward correlation matrix generated using the exponential
        #decay control, Equation (6.24), with the following set of parameters: β = 0.05, ρ∞ = 0.4

        ax.set_xlabel('Tl(Years)')
        ax.set_ylabel('Tk(Years)')
        ax.set_zlabel('Correlation')
        plt.title('Parametric Correlation')
        #plt.savefig('ParametricCorrelation')
        #plt.title('Forward-forward correlation matrix generated using the exponential')
        plt.show()