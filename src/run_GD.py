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
    print("start predictions")
    y_pred = []
    for i in range(4):
        x, ind = clean_data(tX, i)
        xt, ind = clean_data(tX_test, i)
        y_tr = y[tX[:, 22] == i]
        x, xt = PCA_2(x, xt)
        _, x = build_model_data(x, y_tr)
        _, xt = build_model_data(xt, y_tr)
        w, _ = least_squares_GD(y_tr, x, [], 300, 7.19685673e-02)
        print(predict_labels(w, xt))
        y_pred.append(predict_labels(w, xt))
    print("final step")
    predict(y_pred, tX_test, ids_test)
main()
