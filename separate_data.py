"""

データ分割位置確認用

"""


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
#=====================================================================

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
df = res[['time', 'openAsk', 'closeAsk', 'highAsk', 'lowAsk', 'volume']]
df.columns = ['time', 'open', 'close', 'high', 'low', 'volume']

#データ分割================================================================
# データフレームの399件〜410件を表示
df[(int(len(df)*0.8)+1):(int(len(df)*0.8))+11]
