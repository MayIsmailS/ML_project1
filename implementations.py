"""some helper functions for project 1."""
import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from plots import gradient_descent_visualization
from ipywidgets import IntSlider, interact
import math
from plots import *

def compute_loss(y, tx, w):
    """Calculate the loss.
    """
    n = len(y)
    e = y - tx@w
    return (0.5/n)*np.transpose(e)@e

def rmse(y, tx, w):
    e = y - tx@w
    mse = (1/len(y))*(e.T@e)
    return math.sqrt(2*mse)

def least_squares(y, tx):
    """calculate the least squares solution."""
    gram = np.transpose(tx)@tx
    b = np.transpose(tx)@y
    wstar = np.linalg.solve(gram, b)
    e = y - tx@wstar
    mse = (1/len(y))*(e.T@e)
    return wstar, mse

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def build_poly_row(row, d):
    res = row
    for i in range(2,d+1):
        res = np.concatenate((res, row**i), 0)
    return res

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    res = np.zeros((x.shape[0], (degree)*x.shape[1]))
    for j in range(x.shape[0]):
        res[j] = build_poly_row(x[j], degree)
    res = np.c_[np.ones(x.shape[0]), res]
    return res

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    aI = lambda_*2*tx.shape[0]*np.eye(tx.shape[1])
    gram = np.transpose(tx)@tx + aI
    b = np.transpose(tx)@y
    wstar = np.linalg.solve(gram, b)
    e = y - tx@wstar
    mse = (1/(2*len(y)))*np.transpose(e)@e
    return wstar, mse

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    n = len(y)
    e = y - tx@w
    return (-1/n)*np.transpose(tx)@e

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = np.zeros(tx.shape[1])
    loss = 0
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx, w)
        gr = compute_gradient(y, tx, w)
        w = w - gamma*gr
    return loss, w
