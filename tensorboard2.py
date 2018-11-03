#Tensorboard 보기 쉽게 정리하기
#플레이스 홀더로 이름 붙이기
import pandas as pd
import numpy as np
import tensorflow as tf

#csv 파일 로드
data=pd.read_csv("bmi.csv")


#정규화
data["height"]=data["height"]/200
data["weight"]=data["weight"]/100

#레이블을 배열로 변환하기
#- thin=(1,0,0) / normar=(0,1,0) / fat=(0,0,1)
bclass={"thin":[1,0,0],"normal":[0,1,0],"fat":[0,0,1]}
data["label_pat"]=data["label"].apply(lambda x: np.array(bclass[x]))

#테스트를 위한 데이터 분류 
test_csv=data[15000:20000]
test_pat=test_csv[["weight", "height"]]
test_ans=list(test_csv["label_pat"])

#데이터 플레이스

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
    train=optimizer.minimize(cross_entropy)
    
#accuracy계산을 scope로 묶기
with tf.name_scope('accuracy') as scope:
    predict=tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(predict, tf.float32))

#세션 생성
sess=tf.Session()

#tensorboard사용
tw=tf.summary.FileWriter("log_dir", graph=sess.graph)

#세션 실행
sess.run(tf.global_variables_initializer()) #변수 초기화

#학습시키기
for step in range(3500):
    i = (step*100)%14000 #14000과 나눴을 때의 나머지
    rows=data[1+i:1+i+100]
    x_pat=rows[["weight","height"]]
    y_ans=list(rows["label_pat"])
    fd={x:x_pat, y_:y_ans}
    sess.run(train, feed_dict=fd)
    if step%500==0:
        ce=sess.run(cross_entropy, feed_dict=fd)            
        acc=sess.run(accuracy, feed_dict={x:test_pat, y_:test_ans})
        print("step=", step, "cre=", ce, "acc=", acc)


#최종적인 정답률
acc=sess.run(accuracy, feed_dict={x:test_pat, y_:test_ans})
print("정답률=",acc)