#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from ipywidgets import IntSlider, interact
import math
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

    degrees = [8, 11, 11, 11]
    print("start predictions")
    y_pred = []
    for i in range(4):
        x, ind = clean_data(tX, i)
        xt, ind = clean_data(tX_test, i)
        y_tr = y[tX[:, 22] == i]
        x, xt = PCA_2(x, xt)
        x = build_poly(x, degrees[i])
        xt = build_poly(xt, degrees[i])
        w, _ = least_squares(y_tr, x)
        print(predict_labels(w, xt))
        y_pred.append(predict_labels(w, xt))
    print("final step")
    predict(y_pred, tX_test, ids_test)
main()
