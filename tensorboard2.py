#Tensorboard 보기 쉽게 정리하기
#플레이스 홀더로 이름 붙이기
import tensorflow as tf

x=tf.placeholder(tf.float32, [None, 2], name="x")
y_=tf.placeholder(tf.float32, [None, 3], name="y_")

#변수에 이름 붙이기
W=tf.Variable(tf.zeros([2,3]), name="W") #가중치
b=tf.Variable(tf.zeros([3]), name="b") #bias

#interface 부분 scope로 묶기
with tf.name_scope('interface') as scope: #tf.name_scope _ 처리를 스코프 단위로 분할 가능
    W=tf.Variable(tf.zeros([2,3]), name="W")
    b=tf.Variable(tf.zeros([3]), name="b")
    #소프트맥스 회귀 정의
    with tf.name_scope('softmax') as scope:
        y=tf.nn.softmax(tf.matmul(x,W)+b)
        
#loss 계산을 scope로 묶기
with tf.name_scope('loss') as scope:
    cross_entropy=-tf.reduce_sum(y_ * tf.log(y))
    
#training계산을 scope로 묶기
with tf.name_scope('training') as scope:
    optimizer=tf.train.GradientDescentOptimizer(0.01)
    trian=optimizer.minimize(cross_entropy)
    
#accuracy계산을 scope로 묶기
with tf.name_scope('accuracy') as scope:
    predict=tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(predict, tf.float32))