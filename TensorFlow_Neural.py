#以「矩陣運算」來模擬神經網路的訊息傳導
#用數學公式模擬，輸出與接收神經雲的運作方式:
#y1 = activation function(x1 * w11 + x2 * w21 + ... + b1)
#y2 = activation function(x1 * w12 + x2 * w22 + ... + b2)
#公式為 : Y = activation(X * W + B)
#輸出 = 激活函數 (輸入 * 權重 + 偏差)

#以「TensorFlow 張量運算」模擬神經網路
import numpy as np
import tensorflow._api.v2.compat.v1 as tf
tf.disable_eager_execution() #關閉 Eager 即時執行模式

X = tf.Variable([[0.4,0.2,0.4]])
W = tf.Variable([[-0.5,-0.2],
                 [-0.3,0.4],
                 [-0.5,0.2]])
B = tf.Variable([[0.1,0.2]])
XWB = tf.matmul(X,W)+B #將「X」與「W」進行矩陣相乘後加「B」
Y_relu = tf.nn.relu(XWB) #將「XWB」進行激活函數「relu」後得到「Y」
Y_sigmoid = tf.nn.sigmoid(XWB) #將「XWB」進行激活函數「sigmoid」後得到「Y」
Y_softmax = tf.nn.softmax(XWB) #將「XWB」進行激活函數「softmax」後得到「Y」

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('XWB : ',sess.run(XWB))
    print('Y_relu : ',sess.run(Y_relu))
    print('Y_sigmoid : ',sess.run(Y_sigmoid))
    print('Y_softmax : ',sess.run(Y_softmax))

#以常態分佈的亂數產生「Weight 權重」與「Bias 偏差」的初始值
W = tf.Variable(tf.random_normal([3,2]))
B = tf.Variable(tf.random_normal([1,2]))
X = tf.Variable([[0.4,0.2,0.4]])
XWB = tf.matmul(X,W)+B #將「X」與「W」進行矩陣相乘後加「B」
Y_relu = tf.nn.relu(XWB) #將「XWB」進行激活函數「relu」後得到「Y」
Y_sigmoid = tf.nn.sigmoid(XWB) #將「XWB」進行激活函數「sigmoid」後得到「Y」
Y_softmax = tf.nn.softmax(XWB) #將「XWB」進行激活函數「softmax」後得到「Y」

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    (_W,_B,_XWB) = sess.run((W,B,XWB)) #執行一次 sess.run() 取得多個 TensorFlow 變數
    print('W : ',_W)
    print('B : ',_B)
    print('XWB : ',_XWB)
    print('Y_relu : ',sess.run(Y_relu))
    print('Y_sigmoid : ',sess.run(Y_sigmoid))
    print('Y_softmax : ',sess.run(Y_softmax))

#以「placeholder」傳入X值
X = tf.placeholder('float',[None,3]) #定義「X」是 [長度不固定 , 3] 的二維「placeholder」 
W = tf.Variable(tf.random_normal([3,2]))
B = tf.Variable(tf.random_normal([1,2]))
Y_relu = tf.nn.relu(tf.matmul(X,W)+B)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    X_array = np.array([[0.4,0.2,0.4]]) #建立 X_array
    (_X,_W,_B,_Y) = sess.run((X,W,B,Y_relu),feed_dict={X:X_array}) #執行「計算圖」時將「X_array」傳入「X」的「placeholder」
    print('X : ',_X)
    print('W : ',_W)
    print('B : ',_B)
    print('Y_relu : ',_Y)

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

X = tf.placeholder('float',[None,4]) #建立「輸入層 X」
H = layer(inputs=X,input_dim=4,output_dim=3,activation=tf.nn.relu) #建立「隱藏層 H」
Y = layer(inputs=H,input_dim=3,output_dim=2) #建立「輸出層 Y」

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    X_array = np.array([[0.4,0.2,0.4,0.5]])
    (Layer_X,Layer_H,Layer_Y) = sess.run((X,H,Y),feed_dict={X:X_array})
    print('input Layer X : ',Layer_X)
    print('hedden Layer H : ',Layer_H)
    print('output Layer Y : ',Layer_Y)









