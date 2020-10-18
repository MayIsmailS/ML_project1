# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
import matplotlib.pyplot as plt


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

            
# added functions

# from serie 2
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def gradient_descent_visualization(
        gradient_losses, gradient_ws,
        grid_losses, grid_w0, grid_w1,
        mean_x, std_x, height, weight, n_iter=None):
    """Visualize how the loss value changes until n_iter."""
    fig = base_visualization(
        grid_losses, grid_w0, grid_w1, mean_x, std_x, height, weight)

    ws_to_be_plotted = np.stack(gradient_ws)
    if n_iter is not None:
        ws_to_be_plotted = ws_to_be_plotted[:n_iter]

    ax1, ax2 = fig.get_axes()[0], fig.get_axes()[2]
    ax1.plot(
        ws_to_be_plotted[:, 0], ws_to_be_plotted[:, 1],
        marker='o', color='w', markersize=10)
    pred_x, pred_y = prediction(
        ws_to_be_plotted[-1, 0], ws_to_be_plotted[-1, 1],
        mean_x, std_x)
    ax2.plot(pred_x, pred_y, 'r')

    return fig

# from serie 4
def cross_validation_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")
    
    
# cleaning data (made functions)

def standardize(x):
    ''' Returns standardized data, assuming no feature is constant.
    '''
    centered_data = x - np.mean(x, axis=0)          
    std_data = centered_data / np.std(centered_data, axis=0)
    
    return std_data

def remove999(x):
    for i in range(x.shape[1]):
        mean = np.sum(x[x[:, i] != -999, i])/len(x[x[:, i] != -999, i])
        x[x[:, i] == -999, i] = mean
    return x

def clean_data(x, i):
    res_x = x[x[:, 22] == i] # we choose points such that 22nd feature == i
    indices = []
    
    std_feature = np.std(res_x, axis = 0)
    for i in range(len(std_feature)): # features
        if std_feature[i] == 0: # whole column is constant
            indices.append(j)
    res_x = np.delete(res_x, indices, 1) # delete redundant features
    res_x = remove999(res_x) # replace remaining undefined values by mean
    res_x = standardize(res_x) # assuming no std == 0
    return res_x, indices

def clean_data_old(x, i):
    res_x = x[x[:, 22] == i] # we choose points such that 22nd feature == i
    indices = []
    
    std_feature = np.std(res_x, axis = 0)
    for j in range(x.shape[1]): # features
        if np.all(res_x.T[j] == -999): # whole column is equal to -999
            indices.append(j)
    res_x = np.delete(res_x, indices, 1) # delete redundant features
    print(len(indices))
    res_x = remove999(res_x) # replace remaining undefined values by mean
    res_x = standardize(res_x)
    return res_x, indices

def arraySortedOrNot(arr):
 
    # Calculating length
    n = len(arr)
 
    # Array has one or no element or the
    # rest are already checked and approved.
    if n == 1 or n == 0:
        return True
 
    # Recursion applied till last element
    return arr[0] <= arr[1] and arraySortedOrNot(arr[1:])

def PCA(x):
    covariance_matrix= np.cov(x.T) # why transposed?
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix) # happens to be ordered
    variance_explained = [(i/sum(eigen_values))*100 for i in range(len(eigen_values))]
    cumulative_variance_explained = np.cumsum(variance_explained)
    #if arraySortedOrNot(eigen_values):
    #    print("not sorted!")
    i = 0
    while(cumulative_variance_explained[i] < 95):
        i += 1
    projection_matrix = (eigen_vectors.T[:][:i+1]).T # why transposed? 
    X_pca = x.dot(projection_matrix)
    return X_pca

def build_model_data(x, y):
    """Form (y,tX) to get regression data in matrix form."""
    tx = np.c_[np.ones(y.shape[0]), x]
    print("changed")
    return y, tx
