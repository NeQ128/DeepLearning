import re
#匯入 Regular Expression 模組，用於處理正規表示式

#建立一個 Function，用來處理文章裡的 tag
def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('',text) #將文章裡符合正規表示式的片段轉換為空字串，並傳回處理後的文章

import os 
import numpy as np
#匯入 os 模組，用於確認檔案是否存在
#匯入 numpy 模組，用於存取資料型別

#建立一個 Function，用來讀取資料夾裡的檔案
#file_type : 欲讀取的資料類型，分為「train」與「test」
def read_files(file_type):
    path = 'IMDb/aclImdb/' #設定讀取檔案的路徑
    file_list = [] #存放檔案路徑的List
    positive_path = path + file_type + '/pos/' #定義「資料類型」的「正面評價」的目錄路徑
    for f in os.listdir(positive_path): #打開設定好的資料目錄，並讀取每一個檔案的檔名
        file_list += [positive_path+f] #將路徑加檔名存進的List
    negative_path = path + file_type + '/neg/' #定義「資料類型」的「負面評價」的目錄路徑
    for f in os.listdir(negative_path): #打開設定好的資料目錄，並讀取每一個檔案的檔名
        file_list += [negative_path+f] #將路徑加檔名存進的List
    print('file_type : ',file_type,' , files : ',len(file_list)) #讀取全部資料檔名後顯示訊息，與讀取的資料個數

    all_labels = np.array([1] * 12500 + [0] * 12500) 
    #建立一個「Label 真實值」的資料，前 12500 筆為「 1 正面評價」，後 12500 筆為「0 負面評價」
    #由於訓練模型時型別只接受「numpy.ndarray」，所以再將 List 轉換為 ndarray
    all_texts = [] #存放每一筆電影評價所有文字內容的List
    for f in file_list: #透過檔案路徑的List，讀取所有檔案
        with open(f,encoding='utf8') as file: #開啟檔案，並設置讀取編碼為 utf8
            all_texts += [rm_tags(' '.join(file.readlines()))]
            #使用 readlines() 讀取檔案的內容，並用已經建立好的 func 移除 tag，最後加進List
    print('file_type : ',file_type,' read completed') #在存取完成所有資料後，顯示訊息
    return all_texts,all_labels #返回存放所有文字內容的「texts」與評價真實值的「labels」

#建立一個 Function，用來查看訓練模型的結果
#text : 欲查看的文字資料
#label : 文字資料的真實值
#predict : 訓練模型的預測結果
#idx : 欲查看的資料位置，預設為0
def display_Sentiment(text,label,predict,idx = 0):
    Sentiment_Dict = {1:'正面的',0:'負面的'}
    print(text[idx])
    print('label 真實值 : ',Sentiment_Dict[label[idx]],' , predict 預測結果 : ',Sentiment_Dict[predict[idx]])
