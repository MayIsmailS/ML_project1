#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from ipywidgets import IntSlider, interact
from plots import *
from implementations import *
from validation import *

def main():
    print("start")
    print("loading data")
    DATA_TRAIN_PATH = "train.csv"
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    DATA_TEST_PATH = 'test.csv'
    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
    
    print("start predictions")
    max_iters = 400
    gammas_chosen_nll = {0: 0.004281332398719396, 
                         1: 0.0014384498882876629, 
                         2: 0.01, 
                         3: 0.0014384498882876629
                        }
    # cross validation accuracies obtained after 200 iterations:
    # 0.693950676595404, 0.6022516248839369, 0.4382642528188026, 0.6789839379173435
    # 0.694, 0.602, 0.438, 0.679
    y_pred = []
    
    for i in range(4):
        print("group ", i)
        
        # prepare data
        x_tr, ind = clean_data(tX, i)
        x_te, ind = clean_data(tX_test, i)
        y_tr = y[tX[:, 22] == i]
        
        x_tr, x_te = PCA_2(x_tr, x_te)
        y_tr, tx_tr = build_model_data(x_tr, y_tr)
        _, tx_te = build_model_data(x_te, x_te)
        
        # initialize w
        d = tx_tr.shape[1]
        w = np.zeros(d)

        w, loss = logistic_regression(y_tr, tx_tr, w, max_iters, gammas_chosen_nll[i])
        print("loss", loss)
        print(predict_labels(w, tx_te))
        y_pred.append(predict_labels(w, tx_te))
    
    print("final step")
    predict(y_pred, tX_test, ids_test)
    print("end")
main()

# 0.573 accuracy, 0.229 F-1 score
# 0.690 accuracy, 0.356 F-1 score
