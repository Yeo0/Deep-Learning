import tensorflow as tf

#상수와 변수 선언
a=tf.constant(100,name="a")
b=tf.constant(200,name="b")
c=tf.constant(300,name="c")
v=tf.Variable(0,name="v")

#곱셈 수행 그래프 정의
calc_op=a+b*c
assign_op=tf.assign(v, calc_op) #연산을 변수에 대입

#세션 생성
sess=tf.Session()

#TensorBoard 사용하기
tw=tf.summary.FileWriter("log_dir", graph=sess.graph)

#세션 실행하기
sess.run(assign_op)
print(sess.run(v))
