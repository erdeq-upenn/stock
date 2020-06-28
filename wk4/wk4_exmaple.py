#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 22:33:09 2020

@author: Dequan
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

#import tensorflow_docs as tfdocs
#import tensorflow_docs.plots
#import tensorflow_docs.modeling

#dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
#dataset_path
#
#column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
#                'Acceleration', 'Model Year', 'Origin']
#raw_dataset = pd.read_csv(dataset_path, names=column_names,
#                      na_values = "?", comment='\t',
#                      sep=" ", skipinitialspace=True)
#
#dataset = raw_dataset.copy()
#dataset.tail()
#


def example():
    rng = np.random.RandomState(42)
    x = 10 * rng.rand(100)                                      # sparse sample
    
    def model(x, sigma=0.3):
        fast_oscillation = np.sin(5 * x)
        slow_oscillation = np.sin(0.5 * x)
        #atc = x-2                                  # auto correlation 
        noise = sigma * rng.randn(len(x))
    
        return slow_oscillation + fast_oscillation + noise #+atc
    
    y = model(x)
    plt.errorbar(x, y, 0.3, fmt='o');
    
    
    from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
    forest = RandomForestRegressor(200)
    forest.fit(x[:, None], y)
    
    xfit = np.linspace(0, 10, 1000)                             # dense sample
    yfit = forest.predict(xfit[:, None])
    ytrue = model(xfit, sigma=0)
    
#    plt.errorbar(x, y, 0.3, fmt='o', alpha=0.5)
    plt.figure()
    plt.plot(xfit, yfit, '-r');
    plt.plot(xfit, ytrue, '-k', alpha=0.5);

def rf_digits():
    from sklearn.datasets import load_digits
    digits = load_digits()
    digits.keys()

    # set up the figure
    fig = plt.figure(figsize=(6, 6))  # figure size in inches
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    
    # plot the digits: each image is 8x8 pixels
    for i in range(64):
        ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
        ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
        
        # label the image with the target value
        ax.text(0, 7, str(digits.target[i]))
    
    # ---------------------------------------------
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    Xtrain, Xtest, ytrain, ytest = train_test_split(digits.data, digits.target,
                                                    random_state=0)
    model = RandomForestClassifier(n_estimators=1000)
    model.fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    # ---------------------------------------------
#    from sklearn import metrics
#    print(metrics.classification_report(ypred, ytest))
    
    from sklearn.metrics import confusion_matrix
    mat = confusion_matrix(ytest, ypred)
    plt.figure()
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
    plt.xlabel('true label')
    plt.ylabel('predicted label');
    
    
    
if __name__ == '__main__':
#    example()
    rf_digits()
    
    # https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview
    # metrics: RMSE 
    