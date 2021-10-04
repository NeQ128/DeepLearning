#在進行深度學習訓練模型之前，首先下載鐵達尼號的旅客資料集
import urllib.request
import os
#匯入 urllib 模組，用於下載檔案
#匯入 os 模組，用於確認檔案是否存在

url = 'https://hbiostat.org/data/repo/titanic3.xls' #鐵達尼號旅客資料集的下載網址
filepath = 'Titanic/titanic3.xls' #欲儲存檔案的位置與檔名
if not os.path.isfile(filepath): #如果系統找不到鐵達尼號旅客資料集的檔案，便會開始下載
    result = urllib.request.urlretrieve(url,filepath)
    print('downloaded : ',result) #下載成功後，顯示訊息

#匯入需要的模組進行資料預處理
import numpy as np
import pandas as pd

all_df = pd.read_excel(filepath) #利用 Pandas 的 read_excel() 讀取 titanic3.xls 的檔案
#print(all_df[:2]) #查看旅客資料集裡的前2筆資料

'''
鐵達尼號旅客資料集的欄位說明
pclass : 艙等 ， 1 = 頭等艙、2 = 二等艙、3 = 三等艙
survived : 是否生存，0 = 否、1 = 是
name : 姓名
sex : 性別，male = 男性、female = 女性
age : 年齡
sibsp : 手足與配偶一同搭船的人數
parch : 雙親與子女一同搭船的人數
ticket : 船票號碼
fare : 旅客費用
cabin : 艙位號碼
embarked : 登船港口，C = Cherbourg、Q = Queenstown、S = Southampton
'''

#在進行訓練前，首先進行資料的預處理
#製作一份「List」列出欲留下進行訓練的欄位
cols = ['survived','pclass','sex','age','sibsp','parch','fare','embarked']
train_df = all_df.loc[:,cols]
#print(train_df[:2]) #查看篩選後的旅客資料集裡的前2筆資料

#在進行訓練時不能有 Null空值，必須先找出來處理
#print(train_df.isnull().sum())

age_mean = train_df['age'].mean() #計算平均的年齡
train_df.loc[:,'age'] = train_df['age'].fillna(age_mean) #用平均的年齡填滿年齡的空欄位

fare_mean = train_df['fare'].mean() #計算平均的費用
train_df.loc[:,'fare'] = train_df['fare'].fillna(fare_mean) #用平均的費用填滿費用的空欄位

#觀察「登船港口」裡出現最多的位置
#print(train_df['embarked'].value_counts())
embarker_max = train_df['embarked'].value_counts().idxmax() #取得數量最大的登船港口
train_df.loc[:,'embarked'] = train_df['embarked'].fillna(embarker_max) #用最多的數量來填滿登船港口的空欄位

#檢查是否已經處理完空欄位
#print(train_df.isnull().sum())

#「性別」的欄位是文字，必須轉換為數字才能進行訓練
train_df.loc[:,'sex'] = train_df['sex'].map({'female':0,'male':1}).astype(int) #利用 map()，把性別裡的 female 轉換成 0、male 轉換成 1

#將「Embarked」欄位進行「One-hot Encoding」轉換
train_df = pd.get_dummies(data=train_df,columns=['embarked'])
#get_dummies()，date = 要轉換的 DataFrame、columns = 要轉換的欄位 

#把「DataFrame」轉換成「array」才能進行訓練
ndarray = train_df.values
#print(ndarray.shape)
#print(ndarray[:2]) 
#查看轉換成「array」後的資料維度與前2筆資料內容

#將資料分成「Features 特徵值」與「Label 真實值」
labels = ndarray[:,0] #labels 的內容為每一筆資料的第一個欄位(「survived」)
features = ndarray[:,1:] #features 的內容為每一筆資料第一個欄位以後的內容

#匯入 sklearn 的 preprocessing 模組，將資料進行標準化
from sklearn import preprocessing

minmax_scale = preprocessing.MinMaxScaler(feature_range=(0,1)) #設定標準化的範圍為 0 ~ 1
features_normalize = minmax_scale.fit_transform(features) #將「features 特徵值」的資料數值進行標準化
#print(features_normalize[:2]) #查看標準化後的「features」的前2筆資料

#製作 msk遮罩，將資料集以 8 : 2 的比例分為「訓練資料」與「測試資料」
msk = np.random.rand(len(features)) < 0.8
train_features = features_normalize[msk]
train_labels = labels[msk]
test_features = features_normalize[~msk]
test_labels = labels[~msk]

#Keras 多元感知器(MLP)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#建立線性堆疊模型
model = Sequential()

#建立「輸入層」與「隱藏層」
model.add(Dense(units=400, #定義「隱藏層」的神經元為 400
                input_dim = 9, #設定「輸入層」的神經元為 9 (Features 特徵值 內容的數量)
                kernel_initializer='uniform', #使用「uniform distribution 連續型均勻分佈亂數」，初始化「weight(權重)」與「bias (偏差)」
                activation='relu')) #定義激活函數「relu」

#建立第二層「隱藏層」
model.add(Dense(units=40, #定義該「隱藏層」的神經元為 100
                kernel_initializer='uniform', #使用「uniform distribution 連續型均勻分佈亂數」，初始化「weight(權重)」與「bias (偏差)」
                activation='relu')) #定義激活函數「relu」

#建立「輸出層」
model.add(Dense(units=1, #定義「輸出層」的神經元為 1
                kernel_initializer='uniform', #使用「uniform distribution 連續型均勻分佈亂數」，初始化「weight(權重)」與「bias (偏差)」
                activation='sigmoid')) #定義激活函數為「sigmoid」

#查看模型的摘要
print(model.summary())

#定義訓練方式
model.compile(loss='binary_crossentropy', #設定損失函數
              optimizer='adam', #設定訓練時的最優化方法
              metrics=['accuracy']) #設定評估模型的方式為 accuracy準確率

#開始執行訓練
train_histroy = model.fit(x = train_features, #訓練資料影像
                          y = train_labels, #訓練資料真實值
                          validation_split=0.2, #將資料拆分為 80%訓練資料、20%驗證資料
                          epochs = 30, #設定訓練週期
                          batch_size = 100, #設定每一批次訓練的資料筆數
                          verbose = 2) #設定顯示訓練過程
#將訓練完成的資料紀錄放在 train_histroy

from tensorflow import keras
import _Drow as drow_pic
#在訓練完模型後可將模型儲存，並在以後以讀取的方式直接使用
#儲存訓練模型，參數為欲存放的位置
model.save('Titanic/Titanic_model.h5')
#讀取訓練模型，參數為模型的位置
model = keras.models.load_model('Titanic/Titanic_model.h5')

#繪製訓練的過程 : 訓練的過程、訓練資料的準確率、驗證資料的準確率
drow_pic.show_train_histroy(train_histroy,'accuracy','val_accuracy')
print('「訓練資料」與「驗證資料」的準確率差距 : ',train_histroy.history['accuracy'][-1] - train_histroy.history['val_accuracy'][-1])
#「訓練資料 train」與「驗證資料 validation」的準確率越來越高
#drow_pic.show_train_histroy(train_histroy,'loss','val_loss')

#使用 model.evaluate 進行模型準確率評估，將評估後的準確率存進 scores
#test_features : 測試資料的特徵值
#test_labels : 測試資料的真實值
scores = model.evaluate(test_features,test_labels)
print('模型準確率 accuracy = ',scores[1])

#利用 predict 預測每一位旅客的生存機率
prediction = model.predict(test_features)
print('預測結果 : ',prediction[:5]) #查看訓練模型預測的前五位旅客的生存機率