"""some helper functions for project 1."""
import csv
import numpy as np


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

def standardize(x_tr, x_te):
    ''' Returns standardized data, assuming no feature is constant.
    '''
    recentered_x_tr = x_tr - np.mean(x_tr, axis=0)
    recentered_x_te = x_te - np.mean(x_tr, axis=0)
    
    std_x_tr = recentered_x_tr / np.std(recentered_x_tr, axis=0)
    std_x_te = recentered_x_te / np.std(recentered_x_tr, axis=0)

    return std_x_tr, std_x_te

def remove999(x_tr, x_te):
    for i in range(x_tr.shape[1]):
        mean = np.sum(x_tr[x_tr[:, i] != -999, i])/len(x_tr[x_tr[:, i] != -999, i])
        x_tr[x_tr[:, i] == -999, i] = mean
        x_te[x_te[:, i] == -999, i] = mean
    
    return x_tr, x_te

def clean_data(x_tr, x_te, i):
    res_x_tr = x_tr[x_tr[:, 22] == i] # we choose points such that 22nd feature == i
    res_x_te = x_te[x_te[:, 22] == i]
    
    indices = []

    std_feature = np.std(res_x_tr, axis = 0)
    for i in range(len(std_feature)): # features
        if std_feature[i] == 0: # whole column is constant
            indices.append(i)
    
    res_x_tr = np.delete(res_x_tr, indices, 1) # delete redundant features
    res_x_te = np.delete(res_x_te, indices, 1)
    
    res_x_tr, res_x_te = remove999(res_x_tr, res_x_te) # replace remaining undefined values by mean
    res_x_tr, res_x_te = standardize(res_x_tr, res_x_te) # assuming no std == 0
    return res_x_tr, res_x_te

def PCA(x):
    covariance_matrix= np.cov(x.T)
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
#     print(eigen_values)
    variance_explained = []
    for i in eigen_values:
        variance_explained.append((i/sum(eigen_values))*100)
    cumulative_variance_explained = np.cumsum(variance_explained)
    #print(cumulative_variance_explained)
    i = 0
    while(cumulative_variance_explained[i] < 95):
        i += 1
    projection_matrix = (eigen_vectors.T[:][:i+1]).T
    X_pca = x.dot(projection_matrix)
    return X_pca

def build_model_data(x, y):
    """Form (y,tX) to get regression data in matrix form."""
    tx = np.c_[np.ones(x.shape[0]), x]
    return y, tx

def PCA_2(x_train, x_test):
    covariance_matrix_train= np.cov(x_train.T)
    covariance_matrix_test= np.cov(x_test.T)
    eigen_values_train, eigen_vectors_train = np.linalg.eig(covariance_matrix_train)
    eigen_values_test, eigen_vectors_test = np.linalg.eig(covariance_matrix_test)
#     print(eigen_values)
    variance_explained = []
    for i in eigen_values_train:
        variance_explained.append((i/sum(eigen_values_train))*100)
    cumulative_variance_explained = np.cumsum(variance_explained)
    #print(cumulative_variance_explained)
    i = 0
    while(cumulative_variance_explained[i] < 95):
        i += 1
    projection_matrix_train = (eigen_vectors_train.T[:][:i+1]).T
    X_pca_train = x_train.dot(projection_matrix_train)
    projection_matrix_test = (eigen_vectors_test.T[:][:i+1]).T
    X_pca_test = x_test.dot(projection_matrix_test)
    return X_pca_train, X_pca_test

def predict(y_pred, tX_test, ids_test):
    yt0, yt1, yt2, yt3 = y_pred[0], y_pred[1], y_pred[2], y_pred[3]
    cpt0, cpt1, cpt2, cpt3 = 0, 0, 0, 0
    sorted_y_pred = np.zeros(tX_test.shape[0])
    for i in range(tX_test.shape[0]):
        if tX_test[i][22] == 0:
            sorted_y_pred[i] = yt0[cpt0]
            cpt0 += 1
        if tX_test[i][22] == 1:
            sorted_y_pred[i] = yt1[cpt1]
            cpt1 += 1
        if tX_test[i][22] == 2:
            sorted_y_pred[i] = yt2[cpt2]
            cpt2 += 1
        if tX_test[i][22] == 3:
            sorted_y_pred[i] = yt3[cpt3]
            cpt3 += 1
    create_csv_submission(ids_test, sorted_y_pred, "res")
