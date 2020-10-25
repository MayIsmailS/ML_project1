"""some helper functions for project 1."""
import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from plots import gradient_descent_visualization
from ipywidgets import IntSlider, interact
from plots import *

def mse_cost(y, tx, w): # compute loss
    """Compute Least Squares Error or MSE"""
    n, = y.shape
    e = y - np.dot(tx,w)
    loss = np.dot(e.transpose(),e)/ (2*n)
    return loss

def rmse(y, tx, w):
    mse = mse_cost(y, tx, w)
    return np.sqrt(2*mse)

def mse_gradient(y, tx, w):#compute_gradient
    """Compute the gradient."""
    n = y.shape[0]
    e = y - np.dot(tx,w)
    grad = -np.dot(tx.transpose(),e)/n
    return grad

def sigmoid(t):
    """apply the sigmoid function on t."""
    return np.exp(t)/(1 + np.exp(t))

def nll_cost(y, tx, w):
    """compute the cost by negative log likelihood."""
    # y containing -1,1 labels must be changed to 0,1 labels
    y[np.where(y == -1)] = 0
    
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    
    y[np.where(y == 0)] = -1
    
    return np.squeeze(- loss)

def nll_penalized_cost(y, tx, w, lambda_):
    """compute the cost by negative log likelihood with some penalty."""
    loss = nll_cost(y, tx, w)
    penalty = lambda_ * w.T.dot(w) / 2
    return loss + penalty

def nll_gradient(y, tx, w):
    """compute the gradient of negative log likelihood cost function."""
    product = np.dot(tx, w)
    left_term = sigmoid(product) - y
    grad = np.dot(np.transpose(tx), left_term)
    return grad

def least_squares(y, tx):
    """calculate the least squares solution."""
    gram = np.transpose(tx)@tx
    b = np.transpose(tx)@y
    wstar = np.linalg.solve(gram, b)
    e = mse_cost(y, tx, wstar)
    return wstar, mse

def learning_by_stochastic_gradient_descent(y, tx, w, batch_size, gamma, lambda_, type_):
    """Same as stochastic_gradient_descent function but only 1 step- no max_iter parameter"""
        
    # defining gradient and cost functions
    if type_ == "mse":
        gradient = lambda y, tx, w: mse_gradient(y, tx, w)
        cost_function = lambda y, tx, w: mse_cost(y, tx, w)
        
    elif type_ == "nll":
        gradient = lambda y, tx, w: nll_gradient(y, tx, w)
        cost_function = lambda y, tx, w: nll_cost(y, tx, w)
    
    elif type_ == "nll_p":
        gradient = lambda y, tx, w: nll_gradient(y, tx, w)
        cost_function = lambda y, tx, w, lambda_: nll_penalized_cost(y, tx, w, lambda_)
    
    # computing w (1 epoch of stochastic gradient descent)
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        grad = gradient(minibatch_y, minibatch_tx, w)
        w -= gamma*grad
    
    # computing loss
    if lambda_ == None:
        loss = cost_function(y, tx, w)
    else:
        loss = cost_function(y, tx, w, lambda_)
        
    return loss, w

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using Stochastic Gradient Descent."""
    # parameters
    batch_size = 1
    lambda_ = None
    type_ = "mse"
    
    # initialising w
    w = initial_w

    # start the logistic regression
    for n_iter in range(max_iters):
        loss, w = learning_by_stochastic_gradient_descent(y, tx, w, batch_size, gamma, lambda_, type_)

    loss = mse_cost(y,tx,w) # loss of last w found

    return w, loss

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = np.zeros(tx.shape[1])
    loss = 0
    
    for n_iter in range(max_iters):
        loss = mse_cost(y, tx, w)
        gr = mse_gradient(y, tx, w)
        w = w - gamma*gr
    
    loss = mse_cost(y, tx, w)
    
    return loss, w

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using Stochastic Gradient Descent."""
    # parameters
    batch_size = 1
    lambda_ = None
    type_ = "nll"

    # initialising w
    w = initial_w

    # start the logistic regression
    for n_iter in range(max_iters):
        loss, w = learning_by_stochastic_gradient_descent(y, tx, w, batch_size, gamma, lambda_, type_)

    loss = nll_cost(y,tx,w) # loss of last w found

    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Logistic regression using Stochastic Gradient Descent."""
    # parameters
    batch_size = 1
    type_ = "nll_p"

    # initialising w
    w = initial_w

    # start the logistic regression
    for n_iter in range(max_iters):
        loss, w = learning_by_stochastic_gradient_descent(y, tx, w, batch_size, gamma, lambda_, type_)

    loss = nll_penalized_cost(y, tx, w, lambda_) # loss of last w found

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
    mse = mse_cost(y, tx, wstar)
    return wstar, mse

def polynomial_regression(y, x):
    """Constructing the polynomial basis function expansion of the data,
       and then running least squares regression."""
    degrees = range(1, 20)


    for ind, degree in enumerate(degrees):
        res = build_poly(x, degree)
        wstar, mse = least_squares(y, res)
        rms = rmse(y, res, wstar)
    return wstar, mse

def accuracy(y_pred, y):
    return np.mean(y_pred == y)
