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
from google.colab import drive

# dictからDataFrameへ変換

res = pd.read_csv('/content/drive/My Drive/Colab Notebooks/api-usdjpy-1h-0530_short.csv')

# SMA200を計算
res['sma20'] = res['closeAsk'].rolling(20).mean()

# SMA20が算出不可な最初の19行をドロップ
res = res[19:]
res = res.reset_index(drop=True)

# 日付の調整===============================================================
# 文字列 => datatime
def iso_jp(iso):
    date = None
    try:
        date = datetime.strptime(iso,'%Y-%m-%dT%H:%M:%S.%fZ')
        date = pytz.utc.localize(date).astimezone(pytz.timezone("Asia/Tokyo"))
    except ValueError:
        try:
            date = datetime.strptime(iso,'%Y-%m-%dT%H:%M:%S.%fz')
            date = dt.astimezone(pytz.timezone("Asia/Tokyo"))
        except ValueError:
            pass
    return date

# datetime => 表示用文字列
def date_string(date):
    if date is None:
        return ''
    return date.strftime('%Y/%m/%d %H:%M:%S')
        
# ISOから認識しやすい日付へ変換
res['time'] = res['time'].apply(lambda x: iso_jp(x))
res['time'] = res['time'].apply(lambda x: date_string(x))
#==========================================================================

# 必要なデータへ切り分け（askのみ）
df = res[['time', 'openAsk', 'closeAsk', 'highAsk', 'lowAsk', 'volume','sma20']]
df.columns = ['time', 'open', 'close', 'high', 'low', 'volume','sma20']

#データ分割================================================================

# 訓練とテストで日付区切る
split_date = '2019/02/05 09:00:00'
train, test = df[df['time'] < split_date], df[df['time']>=split_date]
del train['time']


#=========================================================================

#LSTMモデルへ訓練させるための前処理=======================================

# windowを設定
window_len = 24
del df['time']

# LSTMへの入力用に処理（訓練）
X = []
for i in range(len(df) - window_len):
    temp = df[i:(i + window_len)].copy()
    for col in train:
       temp.loc[:, col] = temp[col] / temp[col].iloc[0] - 1
    X.append(temp)
Y = (df['close'][window_len:].values / df['close'][:-window_len].values)-1

# PandasのデータフレームからNumpy配列へ変換しましょう
X = [np.array(train_lstm_input) for train_lstm_input in X]
X = np.array(X)

np.save('/content/drive/My Drive/Colab Notebooks/X_0530.npy', X)
np.save('/content/drive/My Drive/Colab Notebooks/Y_0530.npy', Y)
