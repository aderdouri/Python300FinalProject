"""
Date: Monday, 24 July 2017
File name: volCorrelationGen.py
Version: 1 
Author: Abderrazak DERDOURI
Subject: CQF Final Project

Description: Implement parametric correlation generation functions

Notes: AbderrazakDerdouriCQFFinalProject.pdf

Run: python -m unittest volCorrelationGen.VolCorrGenTests.test01ExpVolCorrGen
     python -m unittest volCorrelationGen.VolCorrGenTests.test02VolParamCorrGen
     python -m unittest volCorrelationGen.VolCorrGenTests.test03DExpoVolCorrGen
    
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


def genExponentialCorr(beta, maturity_grid):
    """
    Function which generate a matrix correlation 
    using the exponential parameterization
    @var beta: beta parameter
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
    for i in grid:
        for j in grid:
            corr_matrix[i, j] = np.exp(-beta*abs(maturity_grid[i]-maturity_grid[j]))
    return corr_matrix

def genParametricCorr(beta, rho_inf, maturity_grid):
    """
    Function which generate a matrix correlation 
    using the exponential parameterization with decay control
    @var beta: beta parameter
    @var rho_inf: decay control parameter
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

    for i in grid:
        for j in grid:
            first_e = np.exp(-beta*abs(maturity_grid[i]-maturity_grid[j]))
            corr_matrix[i, j] = (rho_inf + (1-rho_inf)*first_e)

    return corr_matrix

def genDoubleExponentialCorr(alpha, beta, rho_inf, maturity_grid):
    """
    Function which generate a matrix correlation 
    using the double exponential parameterization with decay control
    @var alpha: alpha parameter 
    @var beta: beta parameter
    @var rho_inf: decay control parameter
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

    for i in grid:
        for j in grid:
            first_e = np.exp(-beta*abs(maturity_grid[i]-maturity_grid[j]))
            second_e = np.exp(-alpha*min(maturity_grid[i], maturity_grid[j]))
            corr_matrix[i, j] = (rho_inf + (1-rho_inf)*first_e*second_e)

    return corr_matrix

class VolCorrGenTests(TestCase):
    def oldtest01ExpVolCorrGen(self):
        """
        Exponential Parametric Matrix Correlation Generation test
        """
        print("Exponential Parametric Matrix Correlation")
        df = pd.read_excel('lmmData.xlsx', sheetname='lmmCalibration')      
        start_idx = df[df['TenorTi']=='T1Y'].index[0]    
        end_idx = df[df['TenorTi']=='T2Y3M'].index[0]    
        
        maturity_grid = [1, 1.25, 1.5, 2, 2.25]
        beta = 0.05
        corr_matrix = genExponentialCorr(beta, maturity_grid)
        print(corr_matrix)

    def test01ExpVolCorrGen(self):
        """
        Exponential Parametric Matrix Correlation Generation test
        """
        print("Exponential Parametric Matrix Correlation")
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        beta = 0.05
        maturity_grid = np.linspace(0.5, 30, 30) #.reshape(-1, 1)
        corr_matrix = genExponentialCorr(beta, maturity_grid)

        x =range(0, 30)
        y = range(0, 30)
        x, y = np.meshgrid(x, y)
        z = corr_matrix

        # Plot the surface.
        surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        # Customize the z axis.        
        ax.set_zlim(0.1, 1.0)
        
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        ax.set_xlabel('Ti(Years)')
        ax.set_ylabel('Tj(Years)')
        ax.set_zlabel('Correlation')
        plt.title('Exponential Parametric Correlation, beta = '+ str(beta))
        #plt.savefig('ExponentialParametricCorrelation')
        plt.show()   

    def test02VolParamCorrGen(self):
        """
        plotting a 3D correlation surface colored with the coolwarm color map.
        The surface is made opaque by using antialiased=False.
        """
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        beta, rho_inf = 0.1, 0.4
        maturity_grid = np.linspace(0.5, 30, 30) #.reshape(-1, 1)
        corr_matrix = genParametricCorr(beta, rho_inf, maturity_grid)

        x =range(0, 30)
        y = range(0, 30)
        x, y = np.meshgrid(x, y)
        z = corr_matrix

        # Plot the surface.
        surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        # Customize the z axis.        
        ax.set_zlim(0.1, 1.0)
        
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        ax.set_xlabel('Ti(Years)')
        ax.set_ylabel('Tj(Years)')
        ax.set_zlabel('Correlation')
        plt.title('Exponential Parametric Correlation, beta, rho_inf= '+ str(beta) + ', ' + str(rho_inf))
        #plt.savefig('ExponentialParametricDecayCorrelation')
        plt.show()   

    def test03DExpoVolCorrGen(self):      
        """
        Double exponential parameterization correlation generation test
        """
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        alpha, beta, rho_inf  = 0.1, 0.1, 0.4

        maturity_grid = np.linspace(0.5, 30, 30)#.reshape(-1, 1)
        corr_matrix = genDoubleExponentialCorr(alpha, beta, rho_inf, maturity_grid)

        x = range(0, 30)
        y = range(0, 30)
        x, y = np.meshgrid(x, y)
        z = corr_matrix

        # Plot the surface.
        surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        # Customize the z axis.
        ax.set_zlim(0.1, 1.0)
        ax.set_xlim(0.0, 30.0)
        ax.set_ylim(0.0, 30.0)
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        #Forward-forward correlation matrix generated using the exponential
        #decay control, Equation (6.24), with the following set of parameters: β = 0.05, ρ∞ = 0.4
        ax.set_xlabel('Ti(Years)')
        ax.set_ylabel('Tj(Years)')
        ax.set_zlabel('Correlation')
        plt.title('D. Exponential Parametric Corr, alpha, beta, rho_inf= '+ str(alpha) + ', ' + str(beta) + ', ' + str(rho_inf))
        #plt.savefig('ParametricCorrelation')
        #plt.title('Forward-forward correlation matrix generated using the exponential')
        plt.show()