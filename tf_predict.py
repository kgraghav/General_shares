import numpy as np
import pandas as pd
import math as mth
import matplotlib.pyplot as plt
from numpy import array
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import tensorflow as tf
import keras as ks
from keras.models import Sequential
from keras.layers import Dense
from keras import layers
from keras import callbacks
from keras.preprocessing.sequence import TimeseriesGenerator
# define dataset
series = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15])

data=pd.read_csv('AAPL.csv');
# data=data.set_index('Unnamed: 0')
data=data['Close'].values
# mean=pd.read_csv('mean.csv')['Mean'].values

# mean=series;


pred_len=103
n_features=1
n_input = 35
batch_size=1
epochs=250
steps_per_epoch=10
verbose=3

mean_plot=data[:150];

mean=mean_plot.copy()
data_len=mean.shape[0]
# define generator
fit_ind=np.array(range(mean.shape[0]))
fit_X,pred_X,fit_Y,pred_Y=train_test_split(fit_ind,mean,train_size=0.7,
                                           random_state=None,
                                           shuffle=False)
mean=mean.reshape(data_len,n_features)

generator = TimeseriesGenerator(fit_Y, fit_Y, length=n_input,
                                batch_size=batch_size)
# define model
model = Sequential()
# model.add(layers.Input(shape=n_input))
model.add(layers.LSTM(512, input_shape=(n_input,n_features),
                      return_sequences=(True),activation='relu'))
model.add(layers.LSTM(256,
                      return_sequences=(True),activation='relu'))
# model.add( Dense(256, activation="relu"))
model.add( Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(1))
print('model created')
def scheduler(epoch,lr):
    # print(lr,epoch)
    new_lr=lr;
    if epoch<=1:
        new_lr=0.01
    else:
        # new_lr=(-0.01/(3000))*epoch+0.01
        if new_lr>50**-4:
            new_lr=.1/epoch
        elif new_lr<50**-4:
            new_lr=1/epoch 
    # new_lr=0.01
    return new_lr

mycallbacks=[callbacks.LearningRateScheduler(scheduler)]
opt = tf.keras.optimizers.Adam()
model.compile(loss="mape", optimizer=opt,metrics=['accuracy'])
print('model compiled')
# fit model
mdl = model.fit(generator, epochs=epochs, callbacks=[mycallbacks],
                verbose=verbose,
                    steps_per_epoch=steps_per_epoch)
pred_ind=pred_X.shape[0]-pred_X.shape[0]%n_input;
pred_X=pred_X[:pred_ind]
pred_Y=pred_Y[:pred_ind]
evaluation=model.evaluate(pred_X.reshape(-1,n_input,n_features),
                          pred_Y.reshape(-1,n_input,n_features))
plt.subplots()
plt.plot(mdl.history['loss'])
plt.plot(evaluation[1],color='red')
plt.show()
    
# make a one step prediction out of sample
ind=fit_ind;
mean_pred=mean[:,]
for i in range(pred_len):
    try:
        x_input = mean_pred[mean_pred.shape[0]-n_input:mean_pred.
                            shape[0]].reshape(1,n_input,n_features)
        mean_pred=np.append(mean_pred,(model.predict(x_input)[0][0] ))
    except:
        continue
print(i)


ind=fit_ind;
mean_pred_Y=mean[fit_X]
for i in range(len(pred_X)):
    try:
        x_input_pred = mean_pred_Y[i-n_input:i].reshape(1,n_input,n_features)
        mean_pred_Y=np.append(mean_pred_Y,(model.predict(x_input_pred)[0][0] ))
    except:
        continue
print(i)

mean_pred_Y=np.array(
    mean_pred_Y[-n_input:mean_pred_Y.reshape(-1).shape[0]+1]).reshape(-1)

print('Test Score: ',
      sklearn.metrics.mean_absolute_percentage_error(pred_Y,mean_pred_Y))
print('Val Score: ',
      sklearn.metrics.mean_absolute_percentage_error(
          data[-n_input:-1].reshape(-1),mean_pred[-n_input:-1]))

plt.subplots()
plt.plot(range(mean_plot.shape[0]),mean_plot)
# plt.scatter(pred_X,pred_Y,s=7,color='red')
plt.scatter(range(mean_pred.shape[0]),mean_pred,s=2,color='red')
plt.plot(data)
# plt.plot(fit_X,fit_Y)
# plt.scatter(pred_X,pred_Y,s=3)
plt.show()

plt.subplots()
# plt.plot(range(mean_plot.shape[0]),mean_plot)
# plt.scatter(pred_X,pred_Y,s=7,color='red')
# plt.scatter(range(mean_pred.shape[0]),mean_pred,s=2)
# plt.plot(fit_X,fit_Y)
plt.scatter(pred_X,pred_Y,s=3)
plt.scatter(pred_X,mean_pred_Y,color='green',s=3)
# plt.plot(pred_X,pred_Y)
plt.plot(data)
plt.show()



#%%
ML_TF_REG_FIT

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 20:48:49 2022

@author: raghakg
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 21:54:35 2022

@author: raghakg
"""

import numpy as np
import pandas as pd
import math as mth
import matplotlib.pyplot as plt
from numpy import array
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import tensorflow as tf
import keras as ks
from keras.models import Sequential
from keras.layers import Dense
from keras import layers
from keras import callbacks
from keras.preprocessing.sequence import TimeseriesGenerator
# define dataset
series = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15])
#%%
data=pd.read_csv('RLE00254_05_01_04_22_DAR543.csv');
# data=data.set_index('Unnamed: 0')
data=data.loc[:,['AIR_SUPPLY_PRESS','DOSING_RATE']]


##################### Filters #########################


data_ind=10000;
# n_features=1
# n_input = 2
batch_size=1
epochs=250
steps_per_epoch=None
verbose=1

data=data[:data_ind]


# define generator
X=data.iloc[:,1].values
Y=data.iloc[:,1].values;
fit_X,pred_X,fit_Y,pred_Y=train_test_split(X,Y,train_size=0.7,
                                           random_state=None,
                                           shuffle=False)
# mean=mean.reshape(data_len,n_features)
# generator = TimeseriesGenerator(fit_X, fit_Y, length=n_input,
#                                 batch_size=batch_size)
n_input=len(fit_X)
# define model
model = Sequential()
model.add(layers.Input(shape=1))
# model.add(layers.LSTM(512,input_shape=(n_input),
#                       return_sequences=(True),activation='relu'))
# model.add(layers.LSTM(256,
#                       return_sequences=(True),activation='relu'))
model.add( Dense(256, activation="relu"))
model.add( Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(1))
print('model created')
def scheduler(epoch,lr):
    # print(lr,epoch)
    new_lr=lr;
    if epoch<=1:
        new_lr=0.01
    else:
        # new_lr=(-0.01/(3000))*epoch+0.01
        if new_lr>50**-4:
            new_lr=.1/epoch
        elif new_lr<50**-4:
            new_lr=1/epoch 
    # new_lr=0.01
    return new_lr

mycallbacks=[callbacks.LearningRateScheduler(scheduler)]
opt = tf.keras.optimizers.Adam()
model.compile(loss="mape", optimizer=opt,metrics=['accuracy'])
print('model compiled')
# fit model
fit_X=fit_X.reshape(n_input)
fit_Y=fit_Y.reshape(n_input)
mdl = model.fit(fit_X,fit_Y, epochs=epochs, callbacks=[mycallbacks],
                verbose=verbose,
                    steps_per_epoch=steps_per_epoch)
# evaluation=model.evaluate(pred_X.reshape(-1,n_input,n_features),
#                           pred_Y.reshape(-1,n_input,n_features))
# plt.subplots()
# plt.plot(mdl.history['loss'])
# plt.plot(evaluation[1],color='red')
# plt.show()
    
predicted_Y=model.predict(pred_X)

# mean_pred_Y=np.array(
#     mean_pred_Y[-n_input:mean_pred_Y.reshape(-1).shape[0]+1]).reshape(-1)

# print('Test Score: ',
#       sklearn.metrics.mean_absolute_percentage_error(fit_Y,mean_pred))
# print('Val Score: ',
#       sklearn.metrics.mean_absolute_percentage_error(
#           data[-n_input:-1].reshape(-1),mean_pred[-n_input:-1]))

# plt.subplots()
# plt.plot(range(mean_plot.shape[0]),mean_plot)
# # plt.scatter(pred_X,pred_Y,s=7,color='red')
# plt.scatter(range(mean_pred.shape[0]),mean_pred,s=2,color='red')
# plt.plot(data)
# # plt.plot(fit_X,fit_Y)
# # plt.scatter(pred_X,pred_Y,s=3)
# plt.show()

plt.subplots()
# plt.plot(range(mean_plot.shape[0]),mean_plot)
# plt.scatter(pred_X,pred_Y,s=7,color='red')
# plt.scatter(range(mean_pred.shape[0]),mean_pred,s=2)
# plt.plot(fit_X,fit_Y)
plt.scatter(pred_X,pred_Y,s=3)
plt.scatter(pred_X,predicted_Y,color='orange',s=3)
# plt.plot(pred_X,pred_Y)
# plt.plot(data)
plt.show()
#%%
import numpy as np
import pandas as pd
import math as mth
import matplotlib.pyplot as plt
from numpy import array
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import tensorflow as tf
import keras as ks
from keras.models import Sequential
from keras.layers import Dense
from keras import layers
from keras import callbacks
from keras.preprocessing.sequence import TimeseriesGenerator
data=pd.read_csv('RLE00254_05_01_04_22_DAR543.csv');
# data=data.set_index('Unnamed: 0')
# data=data.loc[:,['AIR_SUPPLY_PRESS','DOSING_RATE']]
data=data.dropna()
################################# LR ######################################
X=data.iloc[:,3:].values;
Y=data.iloc[:,-1].values
fit_X,pred_X,fit_Y,pred_Y=train_test_split(X,Y,train_size=0.7,
                                           random_state=None,
                                           shuffle=False)

reg = LinearRegression().fit(fit_X, fit_Y);

predicted_Y=reg.predict(pred_X)

plt.subplots()
# plt.plot(range(mean_plot.shape[0]),mean_plot)
# plt.scatter(pred_X,pred_Y,s=7,color='red')
# plt.scatter(range(mean_pred.shape[0]),mean_pred,s=2)
# plt.plot(fit_X,fit_Y)
plt.plot(pred_Y)
plt.plot(predicted_Y,color='orange')
# plt.plot(pred_X,pred_Y)
# plt.plot(data)
plt.show()
