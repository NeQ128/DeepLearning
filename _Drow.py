import matplotlib.pyplot as plt

#繪製單張圖片
def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2,2)
    plt.imshow(image,cmap = 'binary') #繪製方式為黑白灰階
    plt.show()


#繪製多張圖片，並顯示 真實值 與 預測結果
#參數為：
#images : 影像
#labes : 真實值
#prediction : 預測結果
#idx : 起始位置的index，預設為0
#num : 顯示的筆數。預設為10，最大不超過25
#label_dict : 定義Label名稱轉換，預設為None
def plot_images_labels_prediction(images,labels,prediction,idx = 0,num = 10,label_dict = None):
    fig = plt.gcf()
    fig.set_size_inches(7,9)
    if num > 25:
        num = 25
    for i in range(0,num):
        ax = plt.subplot(5,5,i+1) #建立 sub子圖形為 5行 5列
        ax.imshow(images[idx],cmap = 'binary')
        if label_dict != None:
            title = 'label = '+str(label_dict[labels[idx]]) #設定標題，如果有傳入名稱轉換 Dict 則顯示圖像名稱
        else:
            title = 'label = '+str(labels[idx]) #設定標題，顯示圖像真實值
        color = 'black'
        if len(prediction) > 0: #如果有傳入預測結果，則一併顯示預測結果
            if str(labels[idx]) != str(prediction[idx]):
                color = 'red' #如果「預測結果」不等於「真實值」，則標題顯示為紅色
            if label_dict != None:
                title += ' ,\n prediction = '+str(label_dict[prediction[idx]])
            else:
                title += ' ,\n prediction = '+str(prediction[idx])
        
        ax.set_title(title,fontsize = 10,color = color)
        #不顯示刻度
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()

#顯示每一周期訓練的 accuracy準確率 與 loss誤差
#參數為：
#train_histroy : 訓練過程的紀錄
#train : 訓練資料的執行結果
#validation : 驗證資料的執行結果
def show_train_histroy(train_histroy,train,validation):
    plt.plot(train_histroy.history[train])
    plt.plot(train_histroy.history[validation])
    plt.title('Train Histroy')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc = 'upper left')
    plt.show()

#列出訓練模型預測該影像對各個答案的機率
#參數為：
#image : 影像
#label : 真實值
#prediction : 預測結果
#predicted_probability : 預測結果的各別機率
#label_dict : 定義Label名稱轉換
#idx : 欲查看資料的位置，預設為0
def show_Predicted_Probability(image,label,prediction,predicted_probability,label_dict,idx = 0):
    print('label :',label_dict[label[idx]])
    print('predict : ',label_dict[prediction[idx]])
    for i in range(len(label_dict)):
        print(label_dict[i] + ' Probability : %1.9f' %(predicted_probability[idx][i]))
    plt.figure(figsize=(2,2))
    plt.imshow(image[idx])
    plt.show()

#顯示每一周期訓練的圖形
#參數為：
#epoch_list : 訓練次數的List
#label_list : 欲顯示內容的List
#label_name : 欲顯示內容的name
def show_tensor_train_histroy(epoch_list,label_list,label_name):
    fig = plt.gcf()
    fig.set_size_inches(6,3)
    plt.plot(epoch_list,label_list,label=label_name)
    plt.xlabel('epoch')
    plt.ylabel(label_name)
    plt.legend([label_name],loc='upper left')
    plt.show()