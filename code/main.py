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

def readfile(file,col_to_read):
    df =  pd.read_csv(file)
    df = df[col_to_read]
    return df

def run_md(file): 

    df = readfile(file,col_to_read)
    ss = file.split('\\')[-1][:-4]
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
    BATCH_SIZE = 32
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
    history = model.fit(train_set,epochs=50,verbose=0)
    # run focast 
    rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
    rnn_forecast = rnn_forecast[split_time - window_size:-1,-1, 0]
    
    
    plt.figure(figsize=(10, 6))
    plot_series(time_valid, x_valid)
    plot_series(time_valid, rnn_forecast)
    plt.legend()
    plt.savefig('plt\\plt_hist_%s'%ss)

    plt.figure()
    plt.plot(history.history['loss'])
    plt.savefig('plt\\plt_%s.png'%ss)

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)


  

#$$$$$$$$$$$$$$$$$$$$$$$$$
  
def multi_md(file):
    df = pd.read_csv(file)
    ss = file.split('\\')[-1][:-4]
    TRAIN_SPLIT = 1000
    features_considered = ['openPrice', 'turnoverVol']
    features = df[features_considered]
    features.index = df['tradeDate']

    dataset = features.values
    data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
    data_std = dataset[:TRAIN_SPLIT].std(axis=0)
    dataset = (dataset-data_mean)/data_std
    
    # Single step model
    def multivariate_data(dataset, target, start_index, end_index, history_size,
                          target_size, step, single_step=False):
      data = []
      labels = []
    
      start_index = start_index + history_size
      if end_index is None:
        end_index = len(dataset) - target_size
    
      for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])
    
        if single_step:
          labels.append(target[i+target_size])
        else:
          labels.append(target[i:i+target_size])
    
      return np.array(data), np.array(labels)
    
    
    past_history = 60
    future_target = 7
    STEP = 1
    
    
    # multistep model 
    future_target = 7
    x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 1], 0,
                                                     TRAIN_SPLIT, past_history,
                                                     future_target, STEP)
    x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 1],
                                                 TRAIN_SPLIT, None, past_history,
                                                 future_target, STEP)
    
    train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
    train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    
    val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
    val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()
    
    def rev(ds):
        return ds*data_std+data_mean
    
    def multi_step_plot(history, true_future, prediction,ss): 
        
      plt.figure(figsize=(12, 6))
      num_in = create_time_steps(len(history))
      num_out = len(true_future)
    
      plt.plot(num_in, np.array(history[:, 1]), label='History')
      plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
               label='True Future')
      if prediction.any():
        plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
                 label='Predicted Future')
      plt.legend(loc='upper left')
      plt.savefig('plt_pred_%s.png'%ss)
      plt.show()
      print('Success plt %s' %ss)
      
    
      
      
    # multistep model 
      
    multi_step_model = tf.keras.models.Sequential()
    multi_step_model.add(tf.keras.layers.LSTM(16,
                                              return_sequences=True,
                                              input_shape=x_train_multi.shape[-2:]))
    multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
    multi_step_model.add(tf.keras.layers.Dense(7))
    
    multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mse')
    EVALUATION_INTERVAL = 200
    EPOCHS = 20
    multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                              steps_per_epoch=EVALUATION_INTERVAL,
                                              validation_data=val_data_multi,
                                              validation_steps=50,verbose=1)
    
    def plot_train_history(history, title,ss):
      loss = history.history['loss']
      val_loss = history.history['val_loss']
    
      epochs = range(len(loss))
    
      plt.figure()
    
      plt.plot(epochs, loss, 'b', label='Training loss')
      plt.plot(epochs, val_loss, 'r', label='Validation loss')
      plt.title(title)
      plt.ylim([0,5])
      plt.legend()
      plt.savefig('plt\\plt_pred7_hist_%s.png'%ss)
      plt.show()
      
    plot_train_history(multi_step_history,
                       'Single Step Training and validation loss',ss)
    
    for x, y in val_data_multi.take(1):
      multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0],ss)
      

#$$$$$$$$$$$$$$$$$$$$$$$$$

# =============================================================================
# This is the section of main file execution
# =============================================================================   
global col_to_read
col_to_read = ['openPrice','turnoverVol']

files = listfiles('D:\\gits\\stock\\code\\data')
file = files[-1]

window_size = 30
batch_size = 32
shuffle_buffer_size = 1000
BATCH_SIZE = 32
BUFFER_SIZE = 10000
for file in files[5:]:
    print('run %s file' %file)
    # run_md(file)
    multi_md(file)
