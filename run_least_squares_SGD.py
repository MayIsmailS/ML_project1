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
    gammas_chosen_mse = {0: 0.015848931924611134,
                        1: 0.015848931924611134,
                        2: 0.015848931924611134,
                        3: 0.0001688831057012037
                        }
    # cross validation accuracies obtained after 200 iterations:
    # 0.6748438625990871, 0.6428092437841741, 0.6331983484198824, 0.6583649160801299
    # 0.675, 0.643, 0.633, 0.658
    y_pred = []
    
    for i in range(4):
        print("group ", i)
        
        # prepare data
        x_tr = clean_data(tX, i)
        x_te = clean_data(tX_test, i)
        y_tr = y[tX[:, 22] == i]
        
        x_tr, x_te = PCA_2(x_tr, x_te)
        y_tr, tx_tr = build_model_data(x_tr, y_tr)
        _, tx_te = build_model_data(x_te, x_te)
        
        # initialize w
        d = tx_tr.shape[1]
        w = np.zeros(d)

        w, loss = least_squares_SGD(y_tr, tx_tr, w, max_iters, gammas_chosen_mse[i])
        print("loss", loss)
        print(predict_labels(w, tx_te))
        y_pred.append(predict_labels(w, tx_te))
    
    print("final step")
    predict(y_pred, tX_test, ids_test)
    print("end")
main()

# 0.684 accuracy, 0.430 F-1 score