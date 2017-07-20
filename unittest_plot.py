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

from unittest import TestCase
import sys

import math
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generateExponentialParametricMatrixCorrelation(beta, maturity_grid):
    grid_length = len(maturity_grid)
    grid = range(grid_length)
    #Create a null correlation matrix
    corr_matrix = np.zeros ( (grid_length, grid_length) )
    for i in grid:
        for j in grid:
            corr_matrix[i, j] = np.exp(-beta*abs(maturity_grid[i]-maturity_grid[j]))
    return corr_matrix

def generateExponentialParametricMatrixCorrelation(alpha, beta, rho_inf, maturity_grid):
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
    for i in grid:
        for j in grid:
            first_e = np.exp(-alpha*abs(maturity_grid[i]-maturity_grid[j]))
            second_e = np.exp(-beta*abs(maturity_grid[i]-maturity_grid[j]))
            corr_matrix[i, j] = (rho_inf + (1-rho_inf)*first_e*second_e)

    return corr_matrix


class Plot_tests(TestCase):
    def test_Plot_01(self):
        print("Exponential Parametric Matrix Correlation")

    def test_Plot_02(self):
        '''
        ======================
        3D surface (color map)
        ======================
        Demonstrates plotting a 3D surface colored with the coolwarm color map.
        The surface is made opaque by using antialiased=False.
        Also demonstrates using the LinearLocator and custom formatting for the
        z axis tick labels.
        '''

        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
        import numpy as np


        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Make data.
        beta = 0.05
        #maturity_grid = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
        maturity_grid = np.linspace(1, 30, 30) #.reshape(-1, 1)
        corr_matrix = generateExponentialParametricMatrixCorrelation(beta, maturity_grid)
        print(corr_matrix)

        x=range(0, 30)
        y=range(0, 30)
        x, y = np.meshgrid(x, y)
        z = corr_matrix

        # Plot the surface.
        surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        #surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)

        # Customize the z axis.
        '''
        ax.set_zlim(-1.01, 1.01)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        '''

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()

    def test_Plot_03(self):
        '''
        ======================
        3D surface (color map)
        ======================
        Demonstrates plotting a 3D surface colored with the coolwarm color map.
        The surface is made opaque by using antialiased=False.
        Also demonstrates using the LinearLocator and custom formatting for the
        z axis tick labels.
        '''

        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
        import numpy as np


        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Make data.
        alpha = 0.1
        beta = 0.1
        rho_inf = 0.4
        #maturity_grid = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
        maturity_grid = np.linspace(1, 30, 59) #.reshape(-1, 1)
        corr_matrix = generateExponentialParametricMatrixCorrelation(alpha, beta, rho_inf, maturity_grid)
        print(corr_matrix)

        x=range(0, 59)
        y=range(0, 59)
        x, y = np.meshgrid(x, y)
        z = corr_matrix

        # Plot the surface.
        surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        #surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)

        # Customize the z axis.
        '''
        ax.set_zlim(-1.01, 1.01)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        '''

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()

