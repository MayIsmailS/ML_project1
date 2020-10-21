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
            indices.append(i)
    res_x = np.delete(res_x, indices, 1) # delete redundant features
    res_x = remove999(res_x) # replace remaining undefined values by mean
    res_x = standardize(res_x) # assuming no std == 0
    return res_x, indices

def PCA(x):
    covariance_matrix= np.cov(x.T) # why transposed?
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix) # happens to be ordered
    variance_explained = [(i/sum(eigen_values))*100 for i in range(len(eigen_values))]
    cumulative_variance_explained = np.cumsum(variance_explained)
    i = 0
    while(cumulative_variance_explained[i] < 95):
        i += 1
    projection_matrix = (eigen_vectors.T[:][:i+1]).T # why transposed? 
    X_pca = x.dot(projection_matrix)
    return X_pca

def build_model_data(y,x):
    """Form (y,tX) to get regression data in matrix form."""
    tx = np.c_[np.ones(y.shape[0]), x]
    return y, tx

def build_tx(x):
    """Form (y,tX) to get regression data in matrix form."""
    tx = np.c_[np.ones(x.shape[0]), x]
    return tx    

# some serie
def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

# new definitions

def cross_validation(y, x, k_indices, k, initial_w, batch_size, max_iters, gamma, gradient, cost_function):
    """return the training and test loss of ridge regression for subgroup k of the
        cross validation."""
    
    indx = k_indices[k] # selecting kth group (changed from k-1 to k)
    
    # separating data (k'th subgroup in test, others in train)
    tx_te = x[indx]
    y_te = y[indx]
    
    tx_tr = np.delete(x, indx, axis=0)
    y_tr = np.delete(y, indx, axis=0)
    
    losses_tr, weights = stochastic_gradient_descent(y_tr, tx_tr, initial_w, batch_size, max_iters, gamma, gradient, cost_function)
    
    # calculate the loss for train and test data (use function given!!)
    loss_tr = cost_function(y_tr, tx_tr, weights[-1])
    loss_te = cost_function(y_te, tx_te, weights[-1])
    
    return loss_tr, loss_te, weights[-1]


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma, gradient, cost_function):
    """Stochastic gradient descent algorithm for a specific cost function and gradient given."""
    losses = []
    ws = [initial_w]
    w = initial_w
    
    # calculating gradient
    
    for i in range(max_iters): # for each epoch, we perform gradient descent on each mini-batch
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size): 
            grad = gradient(minibatch_y, minibatch_tx, w)
            loss = cost_function(minibatch_y, minibatch_tx,w) 
            w = w - gamma*grad
            
            ws.append(w)
            losses.append(loss)
            
            if i%50 == 0:
                
                print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                    bi=i, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    
    return losses, ws

def mse_cost(y, tx, w):
    """Compute Least Squares Error or MSE"""
    n, = y.shape
    e = y - np.dot(tx,w)
    loss = np.dot(e.transpose(),e)/ (2*n)
    return loss

def compute_rmse(mse):
    """Return rmse given mse"""
    return np.sqrt(2*mse)

def mse_gradient(y, tx, w):
    """Compute the gradient."""    
    n = y.shape[0]
    e = y - np.dot(tx,w)
    grad = -np.dot(tx.transpose(),e)/n # shape (d,1)    
    return grad

# serie 5
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

def accuracy(y, y_pred):
    """ Compute accuracy of predicted output. """
    total_samples = len(y)
    correct = 0
    for i in range(total_samples):
        if y[i] == y_pred:
            correct += 1
    return correct/total_samples

def accuracy_cooler(y, y_pred):
    """ Compute accuracy of predicted output. """
    product = y*y_pred
    errors = np.count_nonzero( product == -1) # if mistake, product will give -1
    return errors/len(y)

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
    
