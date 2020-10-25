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

def cross_validation_SGD(y, x, k_indices, k, max_iters, batch_size, gamma, lambda_, type_, i):
    """return the accuracy and test loss for subgroup k of the cross validation
    using SGD."""
    
    indx = k_indices[k] # selecting kth group (changed from k-1 to k)
    
    # separating data (k'th subgroup in test, others used for trainning)
    x_te = x[indx]
    y_te = y[indx]
    
    x_tr = np.delete(x, indx, axis=0)
    y_tr = np.delete(y, indx, axis=0)
    
    # preparing data (reducing dimensionality and adding column of 1's)   
    x_tr, _ = clean_data(x_tr, i)
    x_te, _ = clean_data(x_te, i)
    x_tr, x_te = PCA_2(x_tr, x_te)
    y_tr, tx_tr = build_model_data(x_tr, y_tr)
    y_te, tx_te = build_model_data(x_te, y_te)
    
    # initializing w
    d = tx_tr.shape[1]
    w = np.zeros(d)
    
    # training model and computing w_star
    for i in range(max_iters):
        loss, w = learning_by_stochastic_gradient_descent(y_tr, tx_tr, w, batch_size, gamma, lambda_, type_)
                
        if i%50 == 0:
            print("Gradient Descent({bi}/{ti}): loss={l}".format(bi=i, ti=max_iters - 1, l=loss))
    
    # computing loss for test data and accuracy for train data
    if type == "mse":
        loss = mse_cost(y_tr, tx_tr, w)
        
    elif type_ == "nll":
        loss = nll_cost(y_tr, tx_tr, w)
    
    elif type_ == "nll_p":
        loss = nll_penalized_cost(y_tr, tx_tr, w, lambda_)
    
    y_te_pred = predict_labels(w, tx_te)
    acc = sum(y_te_pred == y_te)/len(y_te) #accuracy(y_te_pred, y_te)
    return acc, loss, w

def cross_validation_demo_SGD(y, x, k_fold, seed, type_, max_iters, gammas,i,lambda_=None, batch_size=1):
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    # define lists to store the loss of training data and test data
    accuracies = []
    losses = []
    
    # cross validation (repeatedly call previous function)
    for gamma in gammas:
        acc_tmp = []
        losses_tmp  = []
        
        for fold in range(k_fold):
            acc, loss, w = cross_validation_SGD(y, x, k_indices, fold, max_iters, batch_size, gamma, lambda_, type_, i)
            acc_tmp.append(acc)
            losses_tmp.append(loss)
        
        # plot of accuracies and losses
        accuracies.append(np.mean(acc_tmp))
        losses.append(np.mean(losses_tmp))
        label_1 = "accuracies "  + str(i)
        label_2 = "losses " + str(i)
    
    cross_validation_visualization(gammas, accuracies, losses, label_1, label_2)

    return gammas, accuracies, losses
