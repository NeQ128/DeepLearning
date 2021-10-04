import numpy as np
import pandas as pd
from tensorflow import keras

#設定隨機參數 seed ，令每次結果相同
np.random.seed(10)

from keras.datasets import mnist
#載入Mnist手寫數字資料集
(x_train_image , y_train_label) , \
(x_test_image , y_test_label) = mnist.load_data()
#print('x_train_image 的資料維度 : ',x_train_image.shape)
#print('y_train_label 的資料維度 : ',y_train_label.shape)
#(60000, 28, 28) : 60000 筆，大小為 28 * 28 的圖片
#(60000,) : 60000 筆對應圖片的真實值

import _Drow as drow_pic

#顯示單張圖片
#drow_pic.plot_image(x_train_image[0])
#顯示多張圖片，參數為：圖片、真實值、預測結果、起始位置、顯示筆數(預設10、最大25)
#drow_pic.plot_images_labels_prediction(x_train_image,y_train_label,[],0,20)

#features 數字影像的資料預處理
x_train = x_train_image.reshape(60000,28,28,1).astype('float32')
x_test = x_test_image.reshape(10000,28,28,1).astype('float32')
#print('維度轉換後的 x_train 資料維度 : ',x_train.shape)
#print('維度轉換後的 x_test 資料維度',x_test.shape)
#(60000, 28, 28, 1) : 將 60000 筆二維(28 * 28)的影像轉換為 60000 筆三維(28 * 28 * 1)的矩陣，再以 astype 轉換為 float 的數字

#將 x_train 與 x_test 的影像數字標準化
#image的數字為 0 ~ 255 ，所以最簡單的標準化方式為除以255
x_train_normalize = x_train / 255
x_test_normalize = x_test / 255
#print('x_train_normalize 標準化後的內容 : ',x_train_normalize[0])
#print('x_test_normalize 標準化後的內容',x_test_normalize[0])

from keras.utils import np_utils
#將「label 真實值」的數值以「One-hot encoding」的方式轉換為 0 與 1 的組合
y_train_OneHot = np_utils.to_categorical(y_train_label)
y_test_OneHot = np_utils.to_categorical(y_test_label)
#print('y_train_label 前五項的內容 : ',y_train_label[:5])
#print('y_train_OneHot 前五項的內容',y_train_OneHot[:5])

#Keras 多元感知器(MLP)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
#卷積神經網路(CNN)
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
'''
卷積層，將一個影像經過「卷積運算」後產生多個影像
「卷積運算」：
1. 先以隨機方式產生「filter weight 濾鏡」
2. 由影像的上至下、左至右依序選取濾鏡大小的「矩陣」
3. 透過選取的「矩陣」與「濾鏡」的乘積，產生新的影像
「卷積運算」經過不同的濾鏡產生的多個影像，可幫助提取影像的各個特徵，例：邊緣、線條、角

在「卷積運算」產生多個影像後經過「Max Pool 縮小取樣」轉換，可將例 4 * 4 的影像縮小為 2 * 2
縮小影像可減少運算所需的時間，並且減小在辨識時位置所產生的影響
參數與計算量的下降，同時也控制「過擬合」的現象
'''

#建立線性堆疊模型
model = Sequential()

#一個完整的卷積運算包含一個「卷積層」與「池化層」
#建立「卷積層」
model.add(Conv2D(filters=16, #建立16個濾鏡 filter weight
                 kernel_size=(5,5), #設定每一個濾鏡大小為 5 * 5
                 padding='same', # 卷積運算產生的卷積影像大小不變
                 input_shape=(28,28,1), #第一、二維代表影像的大小 28 * 28 ，第三維為影像顏色，單色黑白灰階為1
                 activation='relu')) #定義激活函數「relu」
#建立「池化層」
#「池化層」執行縮減取樣，將「卷積層」產生的 16 個 28 * 28 影像縮小為 16 個 14 * 14 影像
model.add(MaxPooling2D(pool_size=(2,2))) 

#建立第二層「卷積層」
model.add(Conv2D(filters=36, #建立36個濾鏡 filter weight
                 kernel_size=(5,5), #設定每一個濾鏡大小為 5 * 5
                 padding='same', # 卷積運算產生的卷積影像大小不變
                 activation='relu')) #定義激活函數「relu」
#建立「池化層」
#「池化層」執行縮減取樣，將「卷積層」產生的 36 個 14 * 14 影像縮小為 36 個 7 * 7 影像
model.add(MaxPooling2D(pool_size=(2,2))) 

#加入「Drop層」，其每次訓練迭帶時隨機捨棄隱藏層 25% 的神經元，避免「過擬合 overfitting」
model.add(Dropout(0.25))

#建立「平坦層」
#將「池化層2」建立的 36 個 7 * 7 的影像轉換成一維的向量 36 * 7 * 7 = 1764
model.add(Flatten())

#建立「隱藏層」
model.add(Dense(units=1000, #定義「隱藏層」的神經元為 1000
                kernel_initializer='normal', #使用「normal distribution 常態分佈亂數」，初始化「weight(權重)」與「bias (偏差)」
                activation='relu')) #定義激活函數「relu」
#激活函數「sigmoid」與「relu」
#sigmood :　神經刺激小於臨界值會忽略；神經刺激大於臨界值時開始對神經刺激有反應；當神經刺激達到一定程度則開始鈍化
#f(x) = 1 / 1 + e ** -x : x < -5 時，y 接近 0 ；x 範圍在 -5 ~ 5 之間，x 越大 y 越大；x > 5 時，y 趨近 1
#relu : 當神經刺激小於臨界值會疏略；當神經刺激大於臨界值時開始有反應
#f(x) = max( 0 , x) : x < 0 時，y = 0；x > 0 時，y = x

#加入「Drop層」，其每次訓練迭帶時隨機捨棄隱藏層 50% 的神經元，避免「過擬合 overfitting」
model.add(Dropout(0.5))

#建立第二層「隱藏層」
model.add(Dense(units=100, #定義該「隱藏層」的神經元為 100
                kernel_initializer='normal', #使用「normal distribution 常態分佈亂數」，初始化「weight(權重)」與「bias (偏差)」
                activation='relu')) #定義激活函數「relu」

#加入「Drop層」，隨機捨棄該隱藏層 50% 的神經元                
model.add(Dropout(0.5))

#建立「輸出層」
model.add(Dense(units=10, #定義「輸出層」的神經元為 10 ，對應 0 ~ 9 共10個數字
                kernel_initializer='normal', #使用「normal distribution 常態分佈亂數」，初始化「weight(權重)」與「bias (偏差)」
                activation='softmax')) #定義激活函數為「softmax」，「softmax」可將神經元的輸出轉換為預測每一個數字的機率

#查看模型的摘要
print(model.summary())
#Param : 1765000 = 1764 * 1000 + 1000  h1 = X:上一層神經元(輸入層) * W1:該層神經元 + B1:該層神經元
#        100100 = 1000 * 100 + 100   h2 = h1:上一層神經元(隱藏層) * W2 + B2
#        1010 = 100 * 10 + 10      y = h2:上一層神經元(隱藏層) * W3 + B3
#總Params 1880962 = 416 + 14436 + 1765000 + 100100 + 1010

#定義訓練方式
model.compile(loss='categorical_crossentropy', #設定損失函數
              optimizer='adam', #設定訓練時的最優化方法
              metrics=['accuracy']) #設定評估模型的方式為 accuracy準確率

#開始執行訓練
train_histroy = model.fit(x = x_train_normalize, #訓練資料影像
                          y = y_train_OneHot, #訓練資料真實值
                          validation_split=0.2, #將資料拆分為 80%訓練資料、20%驗證資料
                          epochs = 10, #設定訓練週期
                          batch_size = 200, #設定每一批次訓練的資料筆數
                          verbose = 2) #設定顯示訓練過程
#將訓練完成的資料紀錄放在 train_histroy

#在訓練完模型後可將模型儲存，並在以後以讀取的方式直接使用
#儲存訓練模型，參數為欲存放的位置
model.save('Models/Mnist_CNN_model.h5')
#讀取訓練模型，參數為模型的位置
model = keras.models.load_model('Models/Mnist_CNN_model.h5')

#繪製訓練的過程 : 訓練的過程、訓練資料的準確率、驗證資料的準確率
drow_pic.show_train_histroy(train_histroy,'accuracy','val_accuracy')
print('「訓練資料」與「驗證資料」的準確率差距 : ',train_histroy.history['accuracy'][-1] - train_histroy.history['val_accuracy'][-1])
#「訓練資料 train」與「驗證資料 validation」的準確率差距縮小，表示加入「Drop層」後改善了「過擬合」的現象
#drow_pic.show_train_histroy(train_histroy,'loss','val_loss')

#使用 model.evaluate 進行模型準確率評估，將評估後的準確率存進 scores
#x_test_normalize : 測試資料 標準化後的影像
#y_test_OneHot : 測試資料 One-hot encoding 轉換後的真實值
scores = model.evaluate(x_test_normalize,y_test_OneHot)
print('模型準確率 accuracy = ',scores[1])

#使用訓練完成的模型進行預測
prediction = model.predict_classes(x_test_normalize)
print('預測結果 : ',prediction)
#繪製多筆影像，並顯示其「真實值」與模型的「預測結果」
drow_pic.plot_images_labels_prediction(x_test_image,y_test_label,prediction,730,25)

#利用 Pandas 建立「混淆矩陣 confusion matrix」幫助觀察模型
import pandas as pd
crosstab = pd.crosstab(y_test_label,prediction,rownames=['label'],colnames=['predict'])
print(crosstab)

df = pd.DataFrame({'label':y_test_label,'predict':prediction})
#print(df[:5]) #列出測試資料前5項的「真實值」與其「預測結果」
#觀察「混淆矩陣」，並列出預測失誤較多的影像筆數
print(df[(df.label == 4) & (df.predict == 9)])
#繪製出預測錯誤的影像
drow_pic.plot_image(x_test_image[2130])