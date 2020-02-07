# ライブラリのインポート
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import oandapy
import configparser
import datetime
from datetime import datetime, timedelta
import pytz
# Kerasの使用するコンポーネントをインポート
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import GRU
from keras.optimizers import Adam
from keras.layers import Dropout
from keras.callbacks import EarlyStopping 
from google.colab import drive
from sklearn.model_selection import train_test_split

# DATA取得===========================================================
df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/api-usdjpy-1h-0530_short.csv')

X_in =np.load('/content/drive/My Drive/Colab Notebooks/X_0530.npy')
Y_in = np.load('/content/drive/My Drive/Colab Notebooks/Y_0530.npy')
#=====================================================================

#=========================================================================

X_train, X_test, Y_train, Y_test = train_test_split(X_in, Y_in, train_size=0.8)
X_train, X_validation, Y_train, Y_validation = \
    train_test_split(X_train, Y_train, train_size=0.8)

# 学習=====================================================================
# LSTMのモデルを設定
def weight_variable(shape, name=None):
    return np.random.normal(scale=.01, size=shape)

early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=1)

inputs = X_train
output_size = 1
neurons = 30
dropout = 0.25
activ_func = "linear"

model = Sequential()
model.add(GRU(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
model.add(Dropout(dropout))
model.add(Dense(units=output_size, kernel_initializer=weight_variable))
model.add(Activation(activ_func))
 
model.compile(loss='mae',
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999))

# ランダムシードの設定
np.random.seed(202)
 
# データを流してフィッティングさせましょう
model.fit(X_train, Y_train, 
          epochs=30, batch_size=1, verbose=2,
          validation_data=(X_validation, Y_validation),
          callbacks=[early_stopping], shuffle=True)

#=================評価1_学習の進捗状況=========================================
"""
  fig, ax1 = plt.subplots(1,1)
 
  ax1.plot(yen_history.epoch, yen_history.history['loss'])
  ax1.set_title('TrainingError')
 
  if yen_model.loss == 'mae':
      ax1.set_ylabel('Mean Absolute Error (MAE)',fontsize=12)
  else:
      ax1.set_ylabel('Model Loss',fontsize=12)
  ax1.set_xlabel('# Epochs',fontsize=12)
  plt.show()

#=================評価2_訓練データの予測結果===================================
# 訓練データから予測をして正解レートと予測したレートをプロット
fig, ax1 = plt.subplots(1,1)

ax1.plot(df[df['time']< split_date]['time'][window_len:],
         train['close'][window_len:], label='Actual', color='blue')

ax1.plot(df[df['time']< split_date]['time'][window_len:],
         ((np.transpose(yen_model.predict(train_lstm_in))+1) * train['close'].values[:-window_len])[0],
         label='Predicted', color='red')
"""

# テストデータを使って予測＆プロット
fig, ax1 = plt.subplots(1,1)
ax1.plot(df[df['time']>= split_date]['time'][window_len:],
         test['close'][window_len:], label='Actual', color='blue')
ax1.plot(df[df['time']>= split_date]['time'][window_len:],
         ((np.transpose(yen_model.predict(test_lstm_in))+1) * test['close'].values[:-window_len])[0], 
         label='Predicted', color='red')
ax1.grid(True)
