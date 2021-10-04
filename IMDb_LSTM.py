#在進行深度學習訓練模型之前，首先下載IMDb網路電影資料集
import urllib.request
import os
import tarfile
#匯入 urllib 模組，用於下載檔案
#匯入 os 模組，用於確認檔案是否存在
#匯入 tarfile 模組，用於解壓縮檔案

url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz' #IMDb網路電影資料集的下載網址
filepath = 'IMDb/aclImdb_v1.tar.gz' #欲儲存檔案的位置與檔名
if not os.path.isfile(filepath): #如果系統找不到IMDb網路電影資料集的壓縮檔，便會開始下載
    result = urllib.request.urlretrieve(url,filepath)
    print('downloaded : ',result) #下載成功後，顯示訊息

if not os.path.exists('IMDb/aclImdb'): #如果系統找不到IMDb網路電影資料集的資料夾，便會開始解壓縮檔案
    tfile = tarfile.open(filepath,'r:gz')
    result = tfile.extractall('IMDb/')
    print('extarct completed') #解壓完成後，顯示訊息

import IMDb_func
#匯入自己建立的 IMDb_func，使用裡面的 Function

#IMDb_func.read_files : 參數欲讀取的資料類型，分為「train」與「test」
#返回「資料類型」的所有文字資料「text」與評價真實值「label」
train_text,y_train = IMDb_func.read_files('train')
test_text,y_test = IMDb_func.read_files('test')

'''
模型訓練時只接受數字，所以必須將「文字內容」轉換為「數字List」
利用 Keras 提供的 Tokenizer 模組，可將「文字」依照出現次數轉換為字典
建立文字字典後，再將文字轉換為「數字List」
'''
from keras.preprocessing.text import Tokenizer
#匯入 Tokenizer 模組，「文字內容」轉換為字典
#當希望提高預測準確率時，嘗試建立較大的字典
token = Tokenizer(num_words=3800) #利用 Tokenizer 建立一個 token，參數為建立 3800 個字的字典
token.fit_on_texts(train_text) #讀取訓練資料的文字內容，並依照出現次數將前 3800 個單字列進字典裡
#print(token.document_count) #利用 token.document_count 屬性，查看 token 讀取了多少文章
#print(token.word_index) #利用 token.word_index 屬性，查看 token 轉換後的內容

#建立 token.word_index 字典後，再將字典轉換為數字List
#利用 token.texts_to_sequences，將資料的「文字內容」透過字典轉換為「數字List」
train_text_seq = token.texts_to_sequences(train_text)
test_text_seq = token.texts_to_sequences(test_text)

#由於每一筆文字內容的字數都不固定，如果要訓練就必須讓每一筆資料的長度相同
from keras.preprocessing import sequence
#匯入 Keras 提供的 sequence 模組，透過截長補短的方式，另每一筆的資料長度相同
#當希望提高預測準確率時，嘗試留下較多的文字
x_train = sequence.pad_sequences(train_text_seq,maxlen=380)
x_test = sequence.pad_sequences(test_text_seq,maxlen=380)
#maxlen : 設定轉換後的文字長度，maxlen = 380 讓轉換後的文字長度為 380
#文字長度不足 380 的部分，在前面用 0 補齊；文字長度超過 380 的資料，則從前面開始截去，留下後面的 380

#完成資料的預處理後，匯入訓練模型所需的模組
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers.embeddings import Embedding
#Keras 提供「Embedding層」，可將「數字List」轉換為「向量List」
#透過向量，讓語意較相近的單字在向量空間裡也較接近
from keras.layers.recurrent import LSTM
#「LSTM 長短期記憶」，專門用來解決「RNN 遞歸神經網路」的「long-term dependencies」問題
#Ct : 「Cell」是 LSTMs 的記憶細胞的狀態 (cell state)
#LSTM通過「Gate 閘門」的機制控制「Cell 記憶細胞」的狀態，刪除或增加其中的訊息
#It : 「Input Gate 輸入閘門」用於決定哪些訊息要被增加到「Cell」
#Ft : 「Forger Gate 忘記閘門」用於決定哪些訊息要從「Cell」刪減
#Ot : 「Output Gate 輸出閘門」用於決定哪些訊息要從「Cell」輸出
#有了「Gate 閘門」機制，LSTM就可以記住長期記憶

#建立線性堆疊模型
model = Sequential()

#加入「Embeddin層」，將「數字List」轉換為「向量List」
model.add(Embedding(output_dim=32, #輸出的維度，將「數字List」轉換為 32 維度的向量
                    input_dim=3800, #輸入的維度，先前建立存放 3800 個字的字典
                    input_length=380)) #「數字List」的長度，每一筆資料的長度為 380

#加入「Drop層」，其每次訓練迭帶時隨機捨棄隱藏層 20% 的神經元，避免「過擬合 overfitting」
model.add(Dropout(0.2))

#加入「LSTM層」
model.add(LSTM(units=32)) #定義「LSTM層」擁有 32 個記憶神經元

#加入「隱藏層」
model.add(Dense(units=256, #定義「隱藏層」的神經元為 256
                activation='relu')) #定義激活函數「relu」

#加入「Drop層」，其每次訓練迭帶時隨機捨棄隱藏層 35% 的神經元，避免「過擬合 overfitting」
model.add(Dropout(0.35))

#建立「輸出層」
model.add(Dense(units=1, #定義「輸出層」的神經元為 1，輸出 1 代表正面評價、0 代表負面評價
                activation='sigmoid')) #定義激活函數為「sigmoid」

#查看模型的摘要
print(model.summary())

#定義訓練方式
model.compile(loss='binary_crossentropy', #設定損失函數
              optimizer='adam', #設定訓練時的最優化方法
              metrics=['accuracy']) #設定評估模型的方式為 accuracy準確率

#開始執行訓練
train_histroy = model.fit(x = x_train, #訓練資料的數字List
                          y = y_train, #訓練資料的真實值
                          validation_split=0.2, #將資料拆分為 80%訓練資料、20%驗證資料
                          epochs = 10, #設定訓練週期
                          batch_size = 100, #設定每一批次訓練的資料筆數
                          verbose = 2) #設定顯示訓練過程
#將訓練完成的資料紀錄放在 train_histroy

from keras.models import load_model
#在訓練完模型後可將模型儲存，並在以後以讀取的方式直接使用
#儲存訓練模型，參數為欲存放的位置
model.save('IMDb/IMDb_LSTM_model.h5')
#讀取訓練模型，參數為模型的位置'''
model = load_model('IMDb/IMDb_LSTM_model.h5')

#使用 model.evaluate 進行模型準確率評估，將評估後的準確率存進 scores
#x_test : 測試資料的數字List
#y_test : 測試資料的真實值
scores = model.evaluate(x_test,y_test)
print('模型準確率 accuracy = ',scores[1])

#使用訓練完成的模型進行預測
prediction = (model.predict(x_test) > 0.5).astype('int32')
print('預測結果 : ',prediction[:10]) #查看預測結果的前10筆資料

#利用 IMDb_func.display_Sentiment，查看模型的預測結果與其真實值
#text : 欲查看的文字資料
#label : 文字資料的真實值
#predict : 訓練模型的預測結果
#idx : 欲查看的資料位置
IMDb_func.display_Sentiment(test_text,y_test.reshape(-1),prediction.reshape(-1),10)