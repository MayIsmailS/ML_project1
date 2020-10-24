import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from ipywidgets import IntSlider, interact
import math
from plots import *
from implementations import *
from validation import *

gammas_chosen_nll_p = {0: 0.00941204967268067,
                       1: 0.03162277660168379,
                       2: 0.0005134832907437549,
                       3: 0.001725210549942041
                      }

def main():
    print("start")
    print("loading data")
    DATA_TRAIN_PATH = "train.csv"
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    DATA_TEST_PATH = 'test.csv'
    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
    
    print("start predictions")
    max_iters = 300
    lambda_ = 0.01
    y_pred = []
    
    for i in range(4):
        
        # prepare data
        x_tr, x_te = clean_data(tX, tX_test, i)
        y_tr = y[tX[:, 22] == i]
        
        x_tr, x_te = PCA_2(x_tr, x_te)
        y_tr, tx_tr = build_model_data(x_tr, y_tr)
        _, tx_te = build_model_data(x_te, x_te)
        
        # initialize w
        d = tx_tr.shape[1]
        w = np.zeros(d)

        w, loss = reg_logistic_regression(y_tr, tx_tr, lambda_, w, max_iters,  gammas_chosen_nll_p[i])
        print("loss", loss)
        print(predict_labels(w, tx_te))
        y_pred.append(predict_labels(w, tx_te))
    
    print("final step")
    predict(y_pred, tX_test, ids_test)
main()

# 0.706 accuracy, 0.449 F-1 score