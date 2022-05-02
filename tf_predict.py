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
mean=data['Close'].values
# mean=pd.read_csv('mean.csv')['Mean'].values

# mean=series;

fit_ind=np.array(range(mean.shape[0]))

pred_len=50
n_features=1
n_input = 35
batch_size=1
epochs=500
steps_per_epoch=15
verbose=1;

mean_plot=mean;

# define generator
fit_X,pred_X,fit_Y,pred_Y=train_test_split(fit_ind,mean,train_size=0.7,
                                           random_state=None,
                                           shuffle=False)
mean=mean.copy()
data_len=mean.shape[0]
mean=mean.reshape(data_len,n_features)

generator = TimeseriesGenerator(mean, mean, length=n_input,
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
mean_pred_Y=pred_Y[:n_input]
for i in range(n_input,len(pred_X)):
    try:
        x_input_pred = mean_pred_Y[i-n_input:i].reshape(1,n_input,n_features)
        mean_pred_Y=np.append(mean_pred_Y,(model.predict(x_input_pred)[0][0] ))
    except:
        continue
print(i)

mean_pred_Y=np.array(mean_pred_Y)

print('Score: ',r2_score(pred_Y,mean_pred_Y))

plt.subplots()
plt.plot(range(mean_plot.shape[0]),mean_plot)
# plt.scatter(pred_X,pred_Y,s=7,color='red')
plt.scatter(range(mean_pred.shape[0]),mean_pred,s=2,color='red')
# plt.plot(fit_X,fit_Y)
# plt.scatter(pred_X,pred_Y,s=3)
plt.show()

plt.subplots()
# plt.plot(range(mean_plot.shape[0]),mean_plot)
# plt.scatter(pred_X,pred_Y,s=7,color='red')
# plt.scatter(range(mean_pred.shape[0]),mean_pred,s=2)
# plt.plot(fit_X,fit_Y)
# plt.scatter(pred_X,pred_Y,s=3)
plt.plot(pred_X,mean_pred_Y,color='green')
plt.plot(pred_X,pred_Y)
plt.show()
