import tensorflow.compat.v1 as tf
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config import basedir
import pandas as pd
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt

class CabbageModel():
    def __init__(self):
        self.basedir = os.path.join(basedir, 'model')
        self.path='model/data/price_data.csv'
        self.df = pd.read_csv(self.path, encoding='UTF-8', thousands=',')
        self.x_data = None
        self.y_data = None
        self.W = None
        self.b = None
        self.X_colunms = ['avgTemp', 'minTemp', 'maxTemp', 'rainFall']
        self.sess = None

    def predicate(self, avgTemp, minTemp, maxTemp, rainFall):
        tf.disable_v2_behavior()
        X = tf.placeholder(dtype=tf.float32, shape=[None, 4])
        W = tf.Variable(tf.random_normal([4, 1]), name='weight')
        b = tf.Variable(tf.random_normal([1]), name='bias')
        saver = tf.train.Saver()
        with tf.Session() as self.sess:
            self.sess.run(tf.global_variables_initializer())
            saver.restore(self.sess, os.path.join(self.basedir, 'cabbage', 'cabbage.ckpt-1000'))
            data = [[avgTemp, minTemp, maxTemp, rainFall]]
            arr = np.array(data, dtype=np.float32)
            dict = self.sess.run(tf.matmul(X, W) + b, {X: arr[:4]})
            print(dict)
        return int(dict[0])

    def show(self):
        [self.figure(x) for x in self.X_colunms]

    def figure(self, x):
        df = self.df
        X = df.loc[:,x]
        Y = df.avgPrice.values
        plt.scatter(X, Y, alpha=0.5)
        plt.title(f'{x} and PRICE Scatter plot')
        plt.xlabel(x)
        plt.ylabel('PRICE')
        plt.show()

    def preprocessing(self):
        data = self.df
        # avgTemp,minTemp,maxTemp,rainFall,avgPrice
        xy = np.array(data, dtype=np.float32)
        self.x_data = xy[:, 1:-1]
        self.y_data = xy[:, [-1]]
        # x_data2 = data.iloc[:, 1:-1]
        # y_data2 = data.iloc[:, -1]
        # ic(x_data2)
        # ic(y_data2)


    def create_model(self):  # create model
        self.preprocessing()
        self.sess = tf.Session()
        _ = tf.Variable(initial_value = 'fake_variable')
        self.sess.run(tf.global_variables_initializer())
        # initializing tensor model(모델 템플릿 생성)
        model = tf.global_variables_initializer()

        # 확률변수 데이터(price_data)
        
        # 선형식(가설, hypothesis) 제작 y = Wx+b
        X = tf.placeholder(dtype=tf.float32, shape=[None, 4], name='X')  # placeholder 는 외부에서 주입되는 값.
        Y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='Y')
        W = tf.Variable(tf.random_normal([4, 1]), name='weight')  # Variable 는 내부 변수
        b = tf.Variable(tf.random_normal([1]), name='bias')
        hypothesis = tf.matmul(X, W) + b
        ic(hypothesis)
        # test = np.dot(X, W) + b
        # ic(test)
        # 손실함수
        cost = tf.reduce_mean(tf.square(hypothesis - Y))
        # Optimizer Algorithm
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000005)
        train = optimizer.minimize(cost)
        # 세션 생성
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        # 트레이닝
        for step in range(100000):
            cost_, hypo_, _ = self.sess.run([cost, hypothesis, train], feed_dict={X: self.x_data, Y: self.y_data})
            if step % 500 == 0:
                print('# %d 손실비용: %d'%(step, cost_))
                print('배추가격 : %d'%(hypo_[0]))
        
        # 모델저장
        saver = tf.train.Saver()
        saver.save(self.sess, os.path.join(self.basedir, 'cabbage', 'cabbage.ckpt'), global_step=1000)
        print('저장완료')

    def fit(self):
        pass

    def eval(self):
        pass


if __name__=='__main__':
    tf.disable_v2_behavior()
    s = CabbageModel()
    s.create_model()
