import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import keras
from keras.datasets import mnist

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

import requests
requests.packages.urllib3.disable_warnings()
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context


#Download dataset 
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#data preprocessing 
X_train = X_train.reshape(60000, 784) / 255
X_test = X_test.reshape(10000, 784) / 255

y_train_origin = y_train
y_test_origin = y_test

with tf.Session() as sesh:
    y_train = sesh.run(tf.one_hot(y_train, 10))
    y_test = sesh.run(tf.one_hot(y_test, 10))

#hyper parameters
learning_rate = 0.01
epochs = 50
batch_size = 100
batch_nums = int(X_train.shape[0]/batch_size)

#input
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

#Set up weights and bias 
Weights = tf.Variable(np.zeros((784, 10)).astype(np.float32))
bias = tf.Variable(np.zeros(10).astype(np.float32))

#setup graph, cost function, optimizer
pred = tf.nn.softmax(tf.add(tf.matmul(X, Weights), bias))
cost_func = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(pred), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_func)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        for i in range(batch_nums):
            offset = i * epoch
            x = X_train[offset: offset + batch_size]
            y = y_train[offset: offset + batch_size]

            sess.run(optimizer, feed_dict={X:x, Y:y})
            cost = sess.run(cost_func, feed_dict={X:x, Y:y})
        if not epoch % 2:
            print(f'cost={cost:.4f}')

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    acc = accuracy.eval({X: X_test, Y: y_test})
    print(f'Model Accuracy: {acc * 100:.2f}%')

    dfSize = 20
    predictLabel = []
    fig, axes = plt.subplots(1, dfSize, figsize=(8, 4))
    for img, ax in zip(X_test[:dfSize], axes):
        guess = np.argmax(sess.run(pred, feed_dict={X: [img]}))
        predictLabel.append(guess)

    predictLabel_train = []
    fig, axes = plt.subplots(1, dfSize, figsize=(8, 4))
    for img, ax in zip(X_train[:dfSize], axes):
        guess = np.argmax(sess.run(pred, feed_dict={X: [img]}))
        predictLabel_train.append(guess)

print('\n\nPredict with testSet')
df = pd.DataFrame({'Actual': list(y_test_origin[:dfSize]), 'Predicted': predictLabel})
print(df)
print('\n\nPredict with trainingSet')
df = pd.DataFrame({'Actual': list(y_train_origin[:dfSize]), 'Predicted': predictLabel_train})
print(df)




