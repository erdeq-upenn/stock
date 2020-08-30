# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 10:27:47 2020

@author: dequa
"""

# general imports 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
import glob
# print(tf.__version__)

# Define functions
def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(False)
    
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)

def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast

def create_time_steps(length):
  return list(range(-length, 0))
def show_plot(plot_data, delta, title):
  labels = ['History', 'True Future', 'Model Prediction']
  marker = ['.-', 'rx', 'go']
  time_steps = create_time_steps(plot_data[0].shape[0])
  if delta:
    future = delta
  else:
    future = 0

  plt.title(title)
  for i, x in enumerate(plot_data):
    if i:
      plt.plot(future, plot_data[i], marker[i], markersize=10,
               label=labels[i])
    else:
      plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
  plt.legend()
  plt.xlim([time_steps[0], (future+5)*2])
  plt.xlabel('Time-Step')
  return plt


def listfiles(path):
    files = glob.glob(path+'\\*')
    return files 

# files = listfiles('D:\\gits\\stock\\code\\data')


def readfile(file,col_to_read):
    df =  pd.read_csv(file)
    df = df[col_to_read]
    return df

col_to_read = ['openPrice','turnoverVol']
df = readfile(files[-1],col_to_read)

def run_md(df):
    series = df['openPrice'].to_numpy()
    time = np.array(df.index)
    plt.figure(figsize=(10, 6))
    plot_series(time, series)
    
    split_time = 1000
    time_train = time[:split_time]
    x_train = series[:split_time]
    time_valid = time[split_time:]
    x_valid = series[split_time:]
    
    window_size = 30
    batch_size = 32
    shuffle_buffer_size = 1000
    BATCH_SIZE = 128
    BUFFER_SIZE = 10000
    
    # model 1 
    
    tf.keras.backend.clear_session()
    tf.random.set_seed(51)
    np.random.seed(51)
    train_set = windowed_dataset(x_train, window_size=30, batch_size=32, shuffle_buffer=shuffle_buffer_size)
    
    model = tf.keras.models.Sequential([
      tf.keras.layers.Conv1D(filters=10, kernel_size=5,
                          strides=1, padding="causal",
                          activation="relu",
                          input_shape=[None, 1]),
      tf.keras.layers.LSTM(10, return_sequences=True),
      # tf.keras.layers.LSTM(30, return_sequences=True),
      tf.keras.layers.Dense(30, activation="relu"),
      tf.keras.layers.Dense(10, activation="relu"),
      tf.keras.layers.Dense(1),
       tf.keras.layers.Lambda(lambda x: x * 2400)
    ])
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        1e-6,
        decay_steps=20,
        decay_rate=0.96,
        staircase=True)
    
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=["mae"])
    history = model.fit(train_set,epochs=50,verbose=1)
    # run focast 
    rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
    rnn_forecast = rnn_forecast[split_time - window_size:-1,-1, 0]
    
    
    plt.figure(figsize=(10, 6))
    plot_series(time_valid, x_valid)
    plot_series(time_valid, rnn_forecast)
    
    plt.figure()
    plt.plot(history.history['loss'])
    
    
run_md(df)
