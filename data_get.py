# ライブラリのインポート
import pandas as pd
from google.colab import drive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import oandapy
import configparser
import datetime
from datetime import datetime, timedelta
import pytz

# DATA取得===========================================================
# OANDA API アクセストークンと口座ID
ACCOUNT_ID = "***-***-********-***"
ACCESS_TOKEN = "********************************************************************"

# デモアカウントでAPI呼び出し
oanda = oandapy.API(environment="practice",
                    access_token=ACCESS_TOKEN)

# 日付を1日毎ずらしてリストを作成

end = "2019-05-30T00:00:00.000000Z"
end = datetime.strptime(end, '%Y-%m-%dT%H:%M:%S.%fZ')
start = []
for i in range(500):
    end = end + timedelta(days=-1)
    weekday = end.weekday()
    if weekday < 5:
        start.append(end.isoformat())
#import pdb; pdb.set_trace()
# ドル円1分足データを取得
inst = "USD_JPY"
gran = "H1"
res = pd.DataFrame([])
length = 100
for i in range(length):
    dic_api = oanda.get_history(instrument=inst,
                                granularity=gran,
                                start=start[i+1] + '.000000Z',
                                end=start[i] + '.000000Z')
    res_be = pd.DataFrame(dic_api['candles'])
    res = pd.concat([res, res_be], axis=0).reset_index(drop=True)
print(res)

# データフレームからCSVファイルへ書き出し
res.to_csv('/content/drive/My Drive/Colab Notebooks/api-usdjpy-1h-0530_short.csv')
