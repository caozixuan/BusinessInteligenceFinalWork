from django.shortcuts import render
import jieba
import os
import re
import numpy as np
import tensorflow as tf
import gensim
import jieba.posseg as pseg
import matplotlib as plt
from industry.models import *


def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.txt':
                L.append(os.path.join(root, file))
    return L


def train_data_build():
    file = r'F:\train_data.txt'
    names = file_name('F:\\data')
    for name in names:
        f = open(name, errors='ignore')
        st = f.read()
        with open(file, 'a+') as f:
            seg_list = jieba.cut(st, cut_all=False)
            f.write(" ".join(seg_list))
            f.write('\n')
        f.close()


def train_data():
    from gensim.models import word2vec
    sentences = word2vec.Text8Corpus('F:\\train_data.txt')
    model = word2vec.Word2Vec(sentences, size=50)
    model.save('word2vec_model')


def getXMLContent():
    names = file_name('E:\\temp_data\\tmp')
    rule1 =r'<contenttitle>(.+?)</contenttitle>'
    rule2 = r'<content>(.+?)</content>'
    file = r'D:\news.txt'
    compile_name = re.compile(rule1, re.M)
    compile_name2 = re.compile(rule2, re.M)
    for name in names:
        f = open(name,errors='ignore')
        st=f.read()
        res_name = compile_name.findall(st)
        res_name2 = compile_name2.findall(st)
        counter = 0
        with open(file, 'a+') as f:
            for sentence in res_name:
                seg_list1 = jieba.cut(sentence, cut_all=False)
                seg_list2 = jieba.cut(res_name2[counter], cut_all=False)
                f.write(" ".join(seg_list1))
                f.write(" ".join(seg_list2))
                counter += + 1
                while counter % 2 == 0:
                    f.write('\n')


def buildData():
    rule = r'(.*)行业'
    compile_name = re.compile(rule, re.M)
    names = file_name('E:\\temp_data\\tmp')
    for name in names:
        f = open(name, errors='ignore')
        st = f.read()
        res_name = compile_name.findall(st)
        for sentence in res_name:
            seg_list = jieba.lcut(sentence,cut_all=False)
            word = seg_list[len(seg_list)-2]
            if len(word)<=1:
                continue
            values = pseg.cut(word)
            flag_word = True
            for value, flag in values:
                if flag == 'n':
                    continue
                else:
                    flag_word = False
            if flag_word:
                Dictionary.objects.get_or_create(name=word)


def SVM():
    sess = tf.Session()
    words = Divided.objects.all()
    model = gensim.models.Word2Vec.load('D:\\word2vec\\word2vec_from_weixin\\word2vec\\word2vec_wx')
    x_vals = np.array([model[word.name].tolist() for word in words])
    y_vals = np.array([1 if word.is_industry else -1 for word in words])
    train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace=False)
    test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
    x_vals_train = x_vals[train_indices]
    x_vals_test = x_vals[test_indices]
    y_vals_train = y_vals[train_indices]
    y_vals_test = y_vals[test_indices]
    # 批训练中批的大小
    batch_size = 100
    x_data = tf.placeholder(shape=[None, 256], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    W = tf.Variable(tf.random_normal(shape=[256, 1]))
    b = tf.Variable(tf.random_normal(shape=[1, 1]))
    # 定义损失函数
    model_output = tf.matmul(x_data, W) + b
    l2_norm = tf.reduce_sum(tf.square(W))
    # 软正则化参数
    alpha = tf.constant([0.1])
    # 定义损失函数
    classification_term = tf.reduce_mean(tf.maximum(0., 1. - model_output * y_target))
    loss = classification_term + alpha * l2_norm
    # 输出
    prediction = tf.sign(model_output)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_target), tf.float32))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    #saver = tf.train.Saver()
    # 开始训练
    sess.run(tf.global_variables_initializer())
    loss_vec = []
    train_accuracy = []
    test_accuracy = []
    for i in range(8000):
        rand_index = np.random.choice(len(x_vals_train), size=batch_size)
        rand_x = x_vals_train[rand_index]
        rand_y = np.transpose([y_vals_train[rand_index]])
        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        loss_vec.append(temp_loss)
        train_acc_temp = sess.run(accuracy, feed_dict={x_data: x_vals_train, y_target: np.transpose([y_vals_train])})
        train_accuracy.append(train_acc_temp)
        test_acc_temp = sess.run(accuracy, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
        test_accuracy.append(test_acc_temp)
        if (i + 1) % 100 == 0:
            print('Step #' + str(i + 1) + ' W = ' + str(sess.run(W)) + 'b = ' + str(sess.run(b)))
            print('Loss = ' + str(test_acc_temp))
    #saver.save(sess, "./model/model.ckpt")
    print(train_accuracy)
    print(test_accuracy)
    plt.plot(loss_vec)
    plt.plot(train_accuracy)
    plt.plot(test_accuracy)
    plt.legend(['loss', 'train accuracy', 'test accuracy'])
    plt.ylim(0., 1.)
    plt.show()


