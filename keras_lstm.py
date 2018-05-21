#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 10:11:58 2018

@author: luogan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation


from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY,YEARLY
from matplotlib.finance import quotes_historical_yahoo_ohlc, candlestick_ohlc
#import matplotlib
import tushare as ts
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import date2num
import datetime
import numpy as np
from pandas import DataFrame
from numpy import row_stack,column_stack
import pandas

df=ts.get_hist_data('000673',start='2016-01-15',end='2018-05-17')
dd=df[['open','high','low','close']]

#print(dd.values.shape[0])

dd1=dd .sort_index()

dd2=dd1.values.flatten()

dd3=pandas.DataFrame(dd1['close'])

dd4=dd1['close']

mm=10
length=100
def load_data(df, sequence_length=length, split=0.8):

    #df = pd.read_csv(file_name, sep=',', usecols=[1])
    #data_all = np.array(df).astype(float)

    data_all = np.array(df).astype(float)
    scaler = MinMaxScaler()
    data_all = scaler.fit_transform(data_all)
    data = []
    for i in range(len(data_all) - sequence_length - mm):
        data.append(data_all[i: i + sequence_length + mm])
    reshaped_data = np.array(data).astype('float64')
    #np.random.shuffle(reshaped_data)
    # 对x进行统一归一化，而y则不归一化
    x = reshaped_data[:, :-mm]
    y = reshaped_data[:, -mm:]
    split_boundary = int(reshaped_data.shape[0] * split)
    train_x = x[: split_boundary]
    test_x = x[split_boundary:]

    train_y = y[: split_boundary]
    test_y = y[split_boundary:]

    return train_x, train_y, test_x, test_y, scaler


def build_model():
    # input_dim是输入的train_x的最后一个维度，train_x的维度为(n_samples, time_steps, input_dim)
    model = Sequential()
    model.add(LSTM(input_dim=1, output_dim=100, return_sequences=True))
    #model.add(LSTM(6, input_dim=1, return_sequences=True))
    #model.add(LSTM(6, input_shape=(None, 1),return_sequences=True))

    """
    #model.add(LSTM(input_dim=1, output_dim=6,input_length=10, return_sequences=True))
    #model.add(LSTM(6, input_dim=1, input_length=10, return_sequences=True))
    model.add(LSTM(6, input_shape=(10, 1),return_sequences=True))
    """
    print(model.layers)
    #model.add(LSTM(100, return_sequences=True))
    #model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(output_dim=mm))
    model.add(Activation('tanh'))

    model.compile(loss='mse', optimizer='adam')
    return model


def train_model(train_x, train_y, test_x, test_y):
    model = build_model()

    try:
        model.fit(train_x, train_y, batch_size=300, nb_epoch=1000, validation_split=0.1)
        predict = model.predict(test_x)
        predict = np.reshape(predict, (predict.size, ))
    except KeyboardInterrupt:
        print(predict)
        print(test_y)
    print(predict)
    print(test_y)
    '''
    try:
        fig = plt.figure(1)
        plt.plot(predict, 'r:')
        plt.plot(test_y, 'g-')
        plt.legend(['predict', 'true'])
    except Exception as e:
        print(e)
    '''
    return predict, test_y


if __name__ == '__main__':
    #train_x, train_y, test_x, test_y, scaler = load_data('international-airline-passengers.csv')
    train_x, train_y, test_x, test_y, scaler =load_data(dd4, sequence_length=length, split=0.8)
    
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
    
    predict_y, test_y = train_model(train_x, train_y, test_x, test_y)
    predict_y = scaler.inverse_transform([[i] for i in predict_y])
    test_y = scaler.inverse_transform(test_y)
    
    pp=predict_y[-mm:].flatten()
    tt= test_y[-1]
    #print()
    def kk(g):
        vv=[]
        for i in range(len(g)-1):
            rr=(g[i+1]-g[i])/g[i]
            vv.append(rr)
        return vv
            
    pp1=kk(pp) 
    tt1=kk(tt)

    
    fig2 = plt.figure(2)
    plt.plot(pp, 'g:')
    plt.plot(tt, 'r-')
    plt.show()
    