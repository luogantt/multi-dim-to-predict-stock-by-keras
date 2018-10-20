#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 10:11:58 2018

@author: luogan
"""
import datetime
from pymongo import MongoClient
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation


from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY,YEARLY
#from matplotlib.finance import quotes_historical_yahoo_ohlc, candlestick_ohlc
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
from dateutil.parser import parse


#import take_data

df=ts.get_hist_data('002230',start='2016-12-15',end='2018-08-21')
df=df.sort_index()


'''
##############################
date=df.index
date1=list(map(parse,date))
df['date']=date1
df=df.sort_values(by='date')
##############################
'''
#df=take_data.fetch_data('601857')

dd1=df[['open','high','p_change','low','close','volume']]

#dd1=df[['openhigh','p_change']]

#dd1=df['open']

#dd4=dd1['close']

mm=3
length=30

#对于rnn 核心不在于如何理解神经网络，而在于如何构建数据
def load_data(df, sequence_length=length, split=0.8):

    #df = pd.read_csv(file_name, sep=',', usecols=[1])
    #data_all = np.array(df).astype(float)

    #改变数据格式
    data_all = np.array(df).astype(float)

    #通过最大最小归一化·
    scaler = MinMaxScaler()
    data_all = scaler.fit_transform(data_all)
    #这行三行代码是所有rnn的核心
    data = []
    for i in range(len(data_all) - sequence_length - mm+1):
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

    train_y1=train_y[:,:,2]
    test_y1=test_y[:,:,2]
    return train_x, train_y1, test_x, test_y1, scaler


def build_model():
    # input_dim是输入的train_x的最后一个维度，train_x的维度为(n_samples, time_steps, input_dim)
    model = Sequential()
    model.add(LSTM(input_dim=6, output_dim=300, return_sequences=True))
    #model.add(LSTM(6, input_dim=1, return_sequences=True))
    #model.add(LSTM(6, input_shape=(None, 1),return_sequences=True))

    """
    #model.add(LSTM(6, input_dim=1, input_length=10, return_sequences=True))
    model.add(LSTM(6, input_shape=(10, 1),return_sequences=True))
    """
    print(model.layers)
    #model.add(LSTM(1000, return_sequences=True))
    #model.add(LSTM(1000, return_sequences=True))
    model.add(LSTM(100, return_sequences=False))
    
    model.add(Dense(output_dim=100))
    
    model.add(Dense(output_dim=100))
    
    model.add(Dense(output_dim=mm))
    model.add(Activation('tanh'))

    model.compile(loss='mse', optimizer='adam')
    return model


def train_model(train_x, train_y, test_x, test_y):
    model = build_model()

    try:
        model.fit(train_x, train_y, batch_size=60, nb_epoch=50, validation_split=0.1)
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

def over_lap(dg):
    m,n=dg.shape

    d=[]
    for i in range(m):
        c=[]
        for j in range(n):
            if min(i-j,j)>=0:
                c.append([i-j,j])
        d.append(c) 


    m=m-1
    for p in range(n):
        h=[]

        xl=m+p
        for i in range(m,m-n+p+1,-1):
            h.append(i)    
        h1=np.array(h)
        h2=xl-h1+1 
        h3=list(h2)

        kk=list(zip(h,h3))
        if len(kk)>0:
            d.append(kk)

    return d     
#dd=over_lap(dg)




#train_x, train_y, test_x, test_y, scaler = load_data('international-airline-passengers.csv')
train_x, train_y, test_x, test_y, scaler =load_data(dd1, sequence_length=length, split=0.8)

train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 6))
test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 6))


#train_y=train_y[:,:,2]
#test_y=test_y[:,:,2]
predict_y1, test_y1 = train_model(train_x, train_y, test_x, test_y)

predict_y2=predict_y1.reshape(-1,3)

mm=dd1['p_change'].max()
nn=dd1['p_change'].min()
predict_y=predict_y2*(mm-nn)+nn
test_y=test_y1*(mm-nn)+nn

#predict_y = scaler.inverse_transform([[i] for i in predict_y])
#test_y = scaler.inverse_transform(test_y)



py=over_lap(predict_y)
ty=over_lap(test_y)

def avg_pc(index_set,xx_y):

    rr=[]
    for p in index_set:
        v=[]
        for q in p:
            #q1=list(q)
            value=xx_y[q[0],q[1]]
            #print(value)
            v.append(value)
        #print(v)
        aver=sum(v)/(len(v))
        rr.append(aver)
    return(rr)
pre_list=avg_pc(py,predict_y)
test_list=avg_pc(ty,test_y)







'''
for  p in range(predict_y.shape[0]):

    ##fig2 = plt.figure(2)
    plt.plot(predict_y[p], 'g:')
    ttt=list(range(len(predict_y)))
    #plt.scatter(ttt,predict_y,s=30,c='green',marker='o',alpha=0.5,label='predict')
  
    #plt.scatter(ttt,test_y,s=10,c='blue',marker='x',alpha=0.5,label='test_y')

    plt.plot(test_y[p], 'r-')
    plt.show()
''' 



client=client = MongoClient('61.129.70.183', 27017)
db=client.stock1.lstm
tt=datetime.datetime.now()
db.insert({'test':test_list,'pre_list':pre_list,'time':tt})

dd3=dd1.iloc[-82:]
dd3['h_change']=0

dd4=dd3[['close','high','h_change','p_change']].values


m,n=dd4.shape

for k in range(1,m):
    this=dd4[k][1]
    before=dd4[k-1][0]
    hc=(this-before)/before
    dd4[k][2]=hc
    
    
'''
plt.rcParams['figure.figsize'] = (24.0, 8.0)
plt.plot(pre_list, 'g:',label='predict')
plt.plot(test_list, 'r:',label='test_y')
plt.plot(list(dd4[:,2]*100), 'b:',label='test_y')


'''

client=client = MongoClient('61.129.70.183', 27017)
db=client.stock1.lstm
dd=db.find({})

dd1=list(dd)
de=pd.DataFrame(dd1)

de1=de.iloc[4]

pre_list=de1['pre_list']
test_list=de1['test']

