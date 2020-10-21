"""some helper functions for project 1."""
import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from plots import gradient_descent_visualization
from ipywidgets import IntSlider, interact
import math
from plots import *
from implementations import *

def cross_validation_rd(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    x_te = x[k_indices[k]]
    y_te = y[k_indices[k]]
    x_tr = np.delete(x, k_indices[k], 0)
    y_tr = np.delete(y, k_indices[k], 0)

    res_tr = build_poly(x_tr, degree)
    res_te = build_poly(x_te, degree)

    w_tr, mse_tr = ridge_regression(y_tr, res_tr, lambda_)
    loss_tr = rmse(y_tr, res_tr, w_tr)
    loss_te = rmse(y_te, res_te, w_tr)
    y_pred = predict_labels(w_tr, res_te)
    acc = np.mean(y_pred == y_te)
    return loss_tr, loss_te, acc

def cross_validation_demo_rd(y, x, degree):
    seed = 1
    k_fold = 4
    lambdas = np.logspace(-8, -1, 15)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    acc = []
    stds = []
    for ind, lambda_ in enumerate(lambdas):
        rms_tr_m = 0
        rms_te_m = 0
        acc_m = []
        ac = 0
        for k in range(k_fold):
            rms_tr, rms_te,  ac= cross_validation_rd(y, x, k_indices, k, lambda_, degree)
            rms_tr_m += rms_tr
            rms_te_m += rms_te
            acc_m.append(ac)
        print("experience num {i} with lambda = {j} and acc = {k}".format(i = ind, j = lambda_, k = np.mean(acc_m)) )
        acc.append(np.mean(acc_m))
        stds.append(np.std(acc_m))
    cross_validation_visualization(lambdas, acc, stds, "accuracy", "std")
def cross_validation_GD(y, x, k_indices, k, gamma, max_iters):
    x_te = x[k_indices[k]]
    y_te = y[k_indices[k]]
    x_tr = np.delete(x, k_indices[k], 0)
    y_tr = np.delete(y, k_indices[k], 0)

    y_tr, x_tr = build_model_data(x_tr, y_tr)
    y_te, x_te = build_model_data(x_te, y_te)

    w_initial = np.zeros(x_tr.shape[1])

    mse_tr, w_tr = least_squares_GD(y_tr, x_tr, w_initial, max_iters, gamma)
    y_pred = predict_labels(w_tr, x_te)
    acc = sum(y_pred == y_te)/len(y_te)
    return acc

def cross_validation_demo_GD(y, tX, gammas, max_iter, k_fold, seed):
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    acc = []
    stds = []
#     mse_te = []
    for gamma in gammas:
        acc_m = []
        #ms_te_m = 0
        for k in range(k_fold):
            acc_m.append(cross_validation_GD(y, tX, k_indices, k, gamma, max_iter))
           # ms_te_m += ms_te
        acc.append(np.mean(acc_m))
        stds.append(np.std(acc_m))
        #mse_te.append(ms_te_m/k_fold)
    cross_validation_visualization(gammas, acc, stds, "acc", "std")
