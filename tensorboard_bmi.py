#bmi.csv를 읽고 tensorflow를 통해 학습시킴
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

#데이터플로우 그래프 구축하기
#플레이스홀더 선언하기
x=tf.placeholder(tf.float32, [None,2]) # 키와 몸무게 데이터 넣기
Y=tf.placeholder(tf.float32, [None,3]) # 정답 레이블 넣기

#변수 선언하기
W=tf.Variable(tf.zeros([2,3])) #가중치
b=tf.Variable(tf.zeros([3])) #bias

#소프트맥스 함수정의
y=tf.nn.softmax(tf.matmul(x,W)+b) #행렬곱

#모델 훈련
cross_entropy=-tf.reduce_sum(Y*tf.log(y))
optimizer=tf.train.GradientDescentOptimizer(0.01) #0.01: 학습계수
train=optimizer.minimize(cross_entropy)

#정답률 구하기
predict=tf.equal(tf.argmax(y,1), tf.argmax(Y,1))
accuracy=tf.reduce_mean(tf.cast(predict, tf.float32))

#세션 시작
sess=tf.Session()

#tensorboard사용
tw=tf.summary.FileWriter("log_dir",graph=sess.graph)

#세션 실행
sess.run(tf.global_variables_initializer()) #변수 초기화

#학습시키기
for step in range(3500):
    i = (step*100)%14000 #14000과 나눴을 때의 나머지
    rows=data[1+i:1+i+100]
    x_pat=rows[["weight","height"]]
    y_ans=list(rows["label_pat"])
    fd={x:x_pat, Y:y_ans}
    sess.run(train, feed_dict=fd)
    if step%500==0:
        ce=sess.run(cross_entropy, feed_dict=fd)            
        acc=sess.run(accuracy, feed_dict={x:test_pat, Y:test_ans})
        print("step=", step, "cre=", ce, "acc=", acc)

            
#최종적인 정답률
acc=sess.run(accuracy, feed_dict={x:test_pat, Y:test_ans})
print("정답률=",acc)