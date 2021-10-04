import numpy as np
import tensorflow._api.v2.compat.v1 as tf
tf.disable_eager_execution() #關閉 Eager 即時執行模式

from tensorflow.examples.tutorials import input_data
#讀取 tensorflow.examples.tutorials.input_data 裡的「Mnist資料集」

mnist = input_data.read_data_sets('TensorFlow/Mnist/',one_hot=True)
#第一次執行時會先確認目錄裡有沒有檔案，沒有的話就會下載「Mnist資料集」

#查看「Mnist資料集」
print('train : ',mnist.train.num_examples)
print('validation : ',mnist.validation.num_examples)
print('test : ',mnist.test.num_examples)
#「Mnist資料集」可分為三部分
#train : 訓練資料 55000 筆
#validation : 驗證資料 5000 筆
#test : 測試資料 10000 筆

#查看訓練資料
print('tarin images : ',mnist.train.images.shape)
print('tarin labels : ',mnist.train.labels.shape)
#每個影像由 784 的數字組成 (28 * 28)
#查看訓練資料 images 的第一筆資料
#print(mnist.train.images[0])
#數值皆於 0 ~ 1，所有數值皆已進行過標準化了

import _Drow as drow_pic
#匯入已建立的函數來查看影像
#drow_pic.plot_image(mnist.train.images[0].reshape(28,28)) #因為資料的影像是 784 的一維資料，必須用 reshape() 轉換成 28 * 28 的二維資料

#查看訓練資料 labels 的第一筆資料
#print(mnist.train.labels[0])
#由於在讀取資料集時，「one_hot」參數為「True」，所以 label 的資料已經過 One-hot Encoding 的轉換
#使用 np.argmax() 將資料轉換回 One-hot 前的資料
#print(np.argmax(mnist.train.labels[0]))

#顯示多張圖片，參數為：圖片、真實值、預測結果、起始位置、顯示筆數(預設10、最大25)
#drow_pic.plot_images_labels_prediction(mnist.train.images.reshape(len(mnist.train.images),28,28), #將所有 image 資料經過 reshape() 處理為 28 * 28 的資料
#                                       np.array([np.argmax(label) for label in mnist.train.labels]), #傳入經過 np.argmax() 處理的 labels 資料
#                                       [],0,20)

#使用 mnist.train.next_batch() 批次讀取資料
batch_images,batch_labels = mnist.train.next_batch(batch_size=100) #下一批次讀取 100 筆資料
#查看讀取的批次資料
#print('batch_images.shape : ',batch_images.shape)
#print('batch_labels.shape : ',batch_labels.shape)
#print(batch_images[0])
#print(batch_labels[0])

#建立「Layer函數」模擬神經網路
#inputs : 輸入的二維「placeholder」
#input_dim : 輸入的神經元數量
#output_dim : 輸出的神經元數量
#activation : 傳入激活函數，預設為None
def layer(inputs,input_dim,output_dim,activation=None):
    W = tf.Variable(tf.random_normal([input_dim,output_dim])) #以常態分佈產生 [輸入神經元 , 輸出神經元]長度的 Weight 權重
    B = tf.Variable(tf.random_normal([1,output_dim])) #以常態分佈產生 [1 , 輸出神經元]長度的 Bias 偏差
    XWB = tf.matmul(inputs,W) + B #將 (X * W) + B 進行矩陣運算
    if activation is None:
        outputs = XWB
    else: #如果有傳入激活函數，用傳入的激活函數進行轉換
        outputs = activation(XWB)
    return outputs

#建立「X 輸入層」的「placeholder」
X = tf.placeholder('float',[None,784]) #[資料筆數，不固定長度所以設 None , 輸入的影像資料為 784 (28 * 28)]

#建立第一層「H1 隱藏層」
H1 = layer(inputs=X, #輸入資料 X
          input_dim=784, #輸入資料的長度 784
          output_dim=1000, #輸出資料的長度(隱藏層的神經元) 1000
          activation=tf.nn.relu) #傳入激活函數 relu

#建立第二層「H2 隱藏層」
H2 = layer(inputs=H1, #輸入資料「H1」的輸出
          input_dim=1000, #輸入資料的長度 1000
          output_dim=256, #輸出資料的長度(隱藏層的神經元) 256
          activation=tf.nn.relu) #傳入激活函數 relu

#建立「Y 輸出層」
Y_predict = layer(inputs=H2, #輸入資料，傳入隱藏層 H
          input_dim=256, #輸入資料的長度，隱藏層輸出(神經元) 256
          output_dim=10) #輸出資料，預測結果總共有 10

#定義訓練方式
#建立訓練資料「label 真實值」的「placeholder」
Y_label = tf.placeholder('float',[None,10]) #[資料筆數，不固定長度所以設 None , 輸入的真實值，經過 One-hot 轉換，共有 10 個 0 跟 1 對應 0 ~ 9]

#定義「loss function 損失函數」
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( #使用「cross_entropy 交叉熵」並計算其平均
                               logits=Y_predict, #傳入模型預測值
                               labels=Y_label)) #傳入資料真實值

#定義最優化方法
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_function)
#使用「AdamOptimizer」並設定「learning_rate」，然後使用定義的「loss_function」計算「loss 誤差」更新模型「Weight 權重」與「Bias 偏差」使 loss 最小化

#評估模型準確率的函數
#用 equal() 判斷「真實值」與「預測值」是否相等，相等傳回 1、反之傳回 0
correct_prediction = tf.equal(tf.argmax(Y_label,1),
                             tf.argmax(Y_predict,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
#先將「correct_prediction 預測正確筆數」資料傳換為 float 後再求其平均，得到模型預測準確率

#在進行訓練前，首先定義訓練參數
trainEpochs = 15 #訓練的週期，訓練次數 15
batchSize = 100 #每一批次的資料筆數
totalBatchs = int(mnist.train.num_examples / batchSize) #每次訓練的批次，資料總筆數 / 每一批次的筆數
epoch_list = [] #紀錄訓練的次數
accuracy_list = [] #紀錄每次訓練的準確率
loss_list = [] #紀錄每次訓練的誤差

#匯入 time 時間模組，用來計算訓練的時間
from time import time
start_time = time() #紀錄起始時間

#開始進行訓練
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(trainEpochs): #欲訓練的次數
        epoch_time = time()
        for batch in range(totalBatchs): #欲訓練的批次
            batch_x,batch_y = mnist.train.next_batch(batchSize) #讀取每一批次要訓練的資料
            sess.run(optimizer,feed_dict={X:batch_x,Y_label:batch_y}) #將資料傳進「計算圖」開始訓練
        loss,acc = sess.run([loss_function,accuracy], #取得每一次訓練完成後的「loss 誤差」與「accuracy 準確率」
                    feed_dict={X:mnist.validation.images,Y_label:mnist.validation.labels}) #傳入驗證資料
        print('Train Epoch : %2d' % (epoch+1),' Loss = : {:.9f}'.format(loss),' Accuracy = ',acc,' Time : ',int(time() - epoch_time),'s')
        #訓練完成後顯示訊息，內容為「Epoch 第幾次訓練」、「Loss 誤差」、「Accuracy 準確率」、「Time 每次的訓練時間」
        epoch_list.append(epoch+1) #將訓練的次數加進 epoch_list
        loss_list.append(loss) #將 loss 誤差 加進 loss_list 紀錄
        accuracy_list.append(acc) #將 acc 準確率 加進 accuracy_list 紀錄

    duration = int(time() - start_time) #計算訓練總時間並顯示
    print('Train Finished , takes : ',duration,'s')

    #用自訂的函數匯出誤差與準確率的圖形
    drow_pic.show_tensor_train_histroy(epoch_list,loss_list,'loss')
    drow_pic.show_tensor_train_histroy(epoch_list,accuracy_list,'accuracy')

    #評估模型的準確率
    print('Accuracy : ',sess.run(accuracy,feed_dict={X:mnist.test.images,Y_label:mnist.test.labels})) #傳入測試資料的影像與真實值

    #使用建立的模型預測
    prediction_result = sess.run(tf.argmax(Y_predict,1),feed_dict={X:mnist.test.images}) #執行預測時將測試資料的影像傳入「X」的 placeholder
    #將預測的結果用 tf.argmax() 轉換為 0 ~ 9 的數字

    #查看預測結果的前10筆資料
    print(prediction_result[:10])

    #繪製多筆影像，並顯示其「真實值」與模型的「預測結果」
    drow_pic.plot_images_labels_prediction(mnist.test.images.reshape(len(mnist.test.images),28,28), #將所有 image 資料經過 reshape() 處理為 28 * 28 的資料
                                       np.array([np.argmax(label) for label in mnist.test.labels]), #傳入經過 np.argmax() 處理的 labels 資料
                                       prediction_result,0,20) #模型預測的結果、欲顯示的資料起始位置、欲顯示的資料筆數
    
#建立「Tensorboard Graph」，以圖形化方式查看「計算圖」
tf.summary.merge_all() #將要顯示在「TensorBoard」上的資料整合
tf_log = tf.summary.FileWriter('TensorFlow/Log/MLP',sess.graph) #將要顯示在「TensorBoard」上的資料寫入 Log檔，並儲存在目錄下的路徑
#在「cmd 命令提示字元」啟用虛擬環境後輸入「tensorboard --logdir=儲存路徑」後取得「localhost:port號」便可查看「計算圖」