import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import math


def plotData(X, y):
    plt.figure()
    plt.scatter(X, y)
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    plt.show()
    pass


def computeCost(X,y,theta):
    J = 0
    m = X.shape[0]
    multiplied = np.matmul(X,theta)
    J = (1/(2*m))*np.sum((multiplied-y)**2)
    return J


def gradientDescent(X, y, theta, alpha, iterations):
    m = len(y)
    # J_history = zeros(num_iters, 1)
    # theta_size = size(X,2)
    # new_theta = theta
    pass

def ex1():
    data = pd.read_csv('ex1data1.txt', sep=",", header=None, prefix=None)
    X = np.array(data[0].values)
    y = np.array(data[1].values)
    m = len(y)

    # plotData(X, y)
    ones = np.ones((m, 1))
    X = np.column_stack((ones, X)) 
    theta = ((0),(0))
    iterations = 1500
    alpha = 0.01
    print(theta)
    J = computeCost(X, y, theta)
    print('With theta = [0 ; 0]\nCost computed {0}'.format(J))
    print('Expected cost value (approx) 32.07\n')

    J = computeCost(X, y, ((-1),(2)))
    print('\nWith theta = [-1 ; 2]\nCost computed = %f\n', J)
    print('Expected cost value (approx) 54.24\n')


    theta = gradientDescent(X, y, theta, alpha, iterations)
pass


if __name__ == '__main__':
    ex1()
