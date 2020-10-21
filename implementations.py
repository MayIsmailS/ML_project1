"""some helper functions for project 1."""
import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from plots import gradient_descent_visualization
from ipywidgets import IntSlider, interact
import math
from plots import *

def mse_cost(y, tx, w):#compute loss
    """Compute Least Squares Error or MSE"""
    n, = y.shape
    e = y - np.dot(tx,w)
    loss = np.dot(e.transpose(),e)/ (2*n)
    return loss

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
def learning_by_stochastic_gradient_descent(y, tx, w, batch_size, gamma, gradient, cost_function):
    """Same as stochastic_gradient_descent function but only 1 step- no max_iter parameter"""
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size): # 1 epoch
        grad = gradient(minibatch_y, minibatch_tx, w)
        loss = cost_function(minibatch_y, minibatch_tx,w)
        w -= gamma*grad
    return loss, w

def least_squares_SGD(y, tx, initial_w, max_iter, gamma):
    """Linear regression using Stochastic Gradient Descent."""
    # parameters
    batch_size = 1
    gradient = lambda y,tx,w: mse_gradient(y,tx,w)
    cost_function = lambda y,tx,w: mse_cost(y,tx,w)

    # initialising w
    w = initial_w

    # start the logistic regression
    for iter in range(max_iter):
        loss, w = learning_by_stochastic_gradient_descent(y, tx, w, batch_size, gamma, gradient, cost_function)

    loss = cost_function(y,tx,w) # loss of last w found

    return w, loss

def logistic_regression(y, tx, initial_w, max_iter, gamma):
    """Logistic regression using Stochastic Gradient Descent."""
    # parameters
    batch_size = 1
    gradient = lambda y,tx,w: nll_gradient(y,tx,w)
    cost_function = lambda y,tx,w: nll_cost(y,tx,w)

    # initialising w
    w = initial_w

    # start the logistic regression
    for iter in range(max_iter):
        loss, w = learning_by_stochastic_gradient_descent(y, tx, w, batch_size, gamma, gradient, cost_function)

    loss = cost_function(y,tx,w) # loss of last w found

    return w, loss

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

def mse_gradient(y, tx, w):#compute_gradient
    """Compute the gradient."""
    n = y.shape[0]
    e = y - np.dot(tx,w)
    grad = -np.dot(tx.transpose(),e)/n # shape (d,1)
    return grad

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = np.zeros(tx.shape[1])
    loss = 0
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx, w)
        gr = compute_gradient(y, tx, w)
        w = w - gamma*gr
    return loss, w
def sigmoid(t):
    """apply the sigmoid function on t."""
    return np.exp(t)/(1 + np.exp(t))

def nll_cost(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)

def nll_gradient(y, tx, w):
    """compute the gradient of negative log likelihood cost function."""
    product = np.dot(tx, w)
    left_term = sigmoid(product) - y
    grad = np.dot(np.transpose(tx), left_term)
    return grad

def accuracy(y_pred, y):
    return np.mean(y_pred == y)
