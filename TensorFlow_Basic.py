'''
TensorFlow 的核心「Computational Graph 計算圖」可分為兩部分：建立「計算圖」與執行「計算圖」
建立「計算圖」:
使用「TensorFlow」提供的模組建立「計算圖」，設計張量運算流程，並且建構各種機器學習與深度學習模型
執行「計算圖」:
建立「計算圖」後，可以建立「Session」執行「計算圖」
在「TensorFlow」中「Session」的作用是在用戶端與執行裝置之間建立連結，有了這個連結，就可以將「計算圖」在各個不同裝置中執行
後續任何與裝置間的資料傳遞，都必須透過「Session」進行並執行「計算圖」後回傳結果。
'''

import tensorflow._api.v2.compat.v1 as tf
#tensorflow._api.v2.compat.v1 : 在 TensorFlow 2 裡使用 TensorFlow 1 的API

tf.disable_eager_execution() #關閉 Eager 即時執行模式
#在「Eager 即時執行模式」之下，TensorFlow會從原先的「declarative 聲明式」變成「imperative 命令式」
#執行任何操作就會直接得到相應的值，不須再通過 sess.run() 取得
#「Eager模式」一旦開啟，便不能被關閉

#建立「計算圖」
#建立 TensorFlow 常數，並查看
ts_c = tf.constant(2,name='ts_c')
#print('ts_c : ',ts_c)
#tf.Tensor : TensorFlow 張量
#shape=() : tensor 的 維度
#dtype : 此張量的資料型態

#建立 TensorFlow 變數，並查看
ts_x = tf.Variable(ts_c+5,name='ts_x')
#print('ts_x : ',ts_x)

#執行「計算圖」之前，首先必須建立「Session」用來連結
#sess = tf.compat.v1.Session() #建立「Session」
#init = tf.compat.v1.global_variables_initializer()
#sess.close() #在建立「Session」後使用 close() 來關閉「Session」
#在程式執行途中可能發生異常或忘記，導致「Session」沒有關閉。可以使用「With」語法來自動關閉
with tf.Session() as sess:
    init = tf.global_variables_initializer() #宣告初始化變數
    sess.run(init) #執行初始化所有 TensorFlow global 變數
    print('ts_c : ',ts_c.eval(session=sess)) #使用 eval() 取得 TensorFlow 變數，使用 eval() 時需傳入 session 參數
    print('ts_x : ',sess.run(ts_x)) #使用 sess.run() 取得 TensorFlow 變數

#建立「placeholder」再透過「feed_dict」傳入參數
width = tf.placeholder('int32',name='width')
height = tf.placeholder('int32',name='height')
area = tf.multiply(width,height,name='area')
#建立2個「placeholder」width(寬) 與 height(高)，然後使用 tf.multiply 將寬高相乘取得 area(面積)
#加入 name 參數設定名稱，能使名稱在「TensorBoard」上顯示讓「計算圖」更易讀

#執行「計算圖」
with tf.Session() as sess:
    init = tf.global_variables_initializer() 
    sess.run(init)
    print('area : ',sess.run(area,feed_dict={width:6,height:8})) #傳入參數 width:6,height:8，得到 area 面積
'''
tf 的常用運算函數
tf.add(x,y) 加、tf.subtract(x,y) 減、tf.multiply(x,y) 乘、tf.divide(x,y) 除
tf.mod(x,y) 餘數、tf.sqrt(x) 平方、tf.abs(x) 絕對值
'''
#建立「Tensorboard Graph」
tf.summary.merge_all() #將要顯示在「TensorBoard」上的資料整合
tf_log = tf.summary.FileWriter('TensorFlow/Log/area',sess.graph) #將要顯示在「TensorBoard」上的資料寫入 Log檔，並儲存在目錄下的路徑
#在「cmd 命令提示字元」啟用虛擬環境後輸入「tensorboard --logdir=儲存路徑」後取得「localhost:port號」便可查看「計算圖」