from django.shortcuts import render
import jieba
import os
import re
import numpy as np
import tensorflow as tf
import gensim
import jieba.posseg as pseg
import matplotlib.pyplot as plt
from industry.models import *

rule_up = [r'(.*)上游(.*)',r'使用(.*)']
rule_down = [r'(.*)下游(.*)',r'(.*)行业下游情况(.*)',r'应用于(.*)']
rule_mid = [r'主营业务(.*)']
rule_company = r'(.*)股份有限公司'
fields = ['能源','电力','冶金','化工','机电','电子','交通','房产','建材','医药','农林','安防','服装','包装',
         '环保','玩具','IT','通信','数码','家电','家居','文教','办公','金融','培训','旅游','食品','烟酒','礼品']

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


def buildDivided():
    counter= 238
    while counter <1400:
        word = Dictionary.objects.get(id=counter)
        Divided.objects.get_or_create(name=word.name,is_industry=word.is_industry)
        counter = counter+1


def delete_word():

    words = Dictionary.objects.all()
    model = gensim.models.Word2Vec.load('D:\\word2vec\\word2vec_from_weixin\\word2vec\\word2vec_wx')
    for word in words:
        try:
            x= model[word.name]
        except  KeyError:
            print(word.name)
            Dictionary.objects.get(name = word.name).delete()


def predict_word():
    sess = tf.Session()
    model = gensim.models.Word2Vec.load('D:\\word2vec\\word2vec_from_weixin\\word2vec\\word2vec_wx')
    #x_vals = np.array([model[keyword].tolist()])
    words = Dictionary.objects.all()
    x_vals = np.array([model[word.name].tolist() for word in words])
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
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    with tf.Session() as sess:
        saver.restore(sess, "./model/model.ckpt")
        print(sess.run(prediction, feed_dict={x_data: x_vals}).tolist())
        predict_list = sess.run(prediction, feed_dict={x_data: x_vals}).tolist()
        words=Dictionary.objects.all()
        counter=0
        for result in predict_list:
            if result[0]==1.0:
                Dictionary.objects.filter(id=words[counter].id).update(is_industry=True)
            counter = counter+1
        #precision 95/104   recall:95/125


def is_in_dic(keyword):
        word = Dictionary.objects.filter(name=keyword)
        if len(word)==0:
            return False
        else:
            if word[0].is_industry:
                Industry.objects.get_or_create(name=word[0].name)
                return True
            else:
                return False


def print_data():
    companies = Company.objects.all()
    for company in companies:
        print(company.name)
        ups = company.up_link.all()
        mids = company.mid_link.all()
        downs = company.down_link.all()
        print("上游产业：")
        for up in ups:
            print(up.name)
        print("中游产业：")
        for mid in mids:
            print(mid.name)
        print("下游产业：")
        for down in downs:
            print(down.name)



def collect_data(pattern):
    names = file_name('E:\\bin')#('D:\\temp_data\\temp_data')
    if pattern==0:
        for rule in rule_up:
            compile_name = re.compile(rule, re.M)
            company_rule = re.compile(rule_company, re.M)
            for name in names:
                # print (x+1)
                if os.path.getsize(name)/float(1024)<100:
                    continue
                f = open(name,errors='ignore')
                st = f.read()
                res_name = compile_name.finditer(st)
                company_name = company_rule.finditer(st)
                y=0
                name_company = ""
                for x in company_name:
                    if y==0:
                        name_company=x.group()
                    y = y+1
                if y>0:
                    Company.objects.get_or_create(name=name_company)
                else:
                    continue
                for m in res_name:
                    seg_list = jieba.cut(m.group(), cut_all=False)
                    for word in seg_list:
                        industries = Industry.objects.all()
                        print (word)
                        is_industry = is_in_dic(word)
                        if not is_industry:
                            continue
                        elif is_industry:
                            for industry in industries:
                                if word == industry.name:
                                    company = Company.objects.filter(name=name_company)
                                    company[0].up_link.add(industry)
                                    company[0].save()
    elif pattern==1:
        for rule in rule_mid:
            compile_name = re.compile(rule, re.M)
            company_rule = re.compile(rule_company, re.M)
            for name in names:
                # print (x+1)
                if os.path.getsize(name)/float(1024)<100:
                    continue
                f = open(name,errors='ignore')
                st = f.read()
                res_name = compile_name.finditer(st)
                company_name = company_rule.finditer(st)
                y=0
                name_company = ""
                for x in company_name:
                    if y==0:
                        name_company=x.group()
                    y = y+1
                if y>0:
                    Company.objects.get_or_create(name=name_company)
                else:
                    continue
                for m in res_name:
                    seg_list = jieba.cut(m.group(), cut_all=False)
                    for word in seg_list:
                        industries = Industry.objects.all()
                        print (word)
                        is_industry = is_in_dic(word)
                        if not is_industry:
                            continue
                        elif is_industry:
                            for industry in industries:
                                if word == industry.name:
                                    company = Company.objects.filter(name=name_company)
                                    company[0].mid_link.add(industry)
                                    company[0].save()
    elif pattern==2:
        for rule in rule_down:
            compile_name = re.compile(rule, re.M)
            company_rule = re.compile(rule_company, re.M)
            for name in names:
                # print (x+1)
                if os.path.getsize(name)/float(1024)<100:
                    continue
                f = open(name,errors='ignore')
                st = f.read()
                res_name = compile_name.finditer(st)
                company_name = company_rule.finditer(st)
                y=0
                name_company = ""
                for x in company_name:
                    if y==0:
                        name_company=x.group()
                    y = y+1
                if y>0:
                    Company.objects.get_or_create(name=name_company)
                else:
                    continue
                for m in res_name:
                    seg_list = jieba.cut(m.group(), cut_all=False)
                    for word in seg_list:
                        industries = Industry.objects.all()
                        print (word)
                        is_industry = is_in_dic(word)
                        if not is_industry:
                            continue
                        elif is_industry:
                            for industry in industries:
                                if word == industry.name:
                                    company = Company.objects.filter(name=name_company)
                                    company[0].down_link.add(industry)
                                    company[0].save()


def buildStop():
    for line in open('F:\\stopword.txt'):
        StopWord.objects.get_or_create(name=line)


def collectSentence():
    names = file_name('D:\\temp_data\\temp_data')
    for name in names:
        if os.path.getsize(name) / float(1024) < 100:
            continue
        f = open(name, errors='ignore')
        st = f.read()
        punct = re.compile(r'[。？！：]')
        sentences = punct.split(st)
        words = Dictionary.objects.all()
        counter=1
        for sentence in sentences:
            for word in words:
                if word.is_industry and word.name in sentence:
                    Sentence.objects.create(content=sentence)
                    print(counter)
                    counter = counter+1
                    break


# keyword used to locate industry entity
rules = [r'(.*)上游(.*)', r'下游(.*)', r'(.*)应用于(.*)']
keywords = ['上游', '下游', '应用']

# you should change the path to your own  model path
word2vec_model_path = 'D:\\word2vec\\word2vec_from_weixin\\word2vec\\word2vec_wx'

# you should change the path to your own data path
data_path = 'F:\\tmp_data\\temp_data\\data'

# parameters used while training
sentence_length = 15
word_dim = 256
class_size = 2
rnn_size = 256
num_layers = 2
epoch = 80
batch_size = 20
learning_rate = 0.003

class Model:
    def __init__(self):
        self.input_data = tf.placeholder(tf.float32, [None, sentence_length, word_dim])
        self.output_data = tf.placeholder(tf.float32, [None, sentence_length, class_size])
        fw_cell = tf.nn.rnn_cell.LSTMCell(rnn_size, state_is_tuple=True)
        fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=0.5)
        bw_cell = tf.nn.rnn_cell.LSTMCell(rnn_size, state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=0.5)
        fw_cell = tf.nn.rnn_cell.MultiRNNCell([fw_cell] * num_layers, state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.MultiRNNCell([bw_cell] * num_layers, state_is_tuple=True)
        words_used_in_sent = tf.sign(tf.reduce_max(tf.abs(self.input_data), reduction_indices=2))
        self.length = tf.cast(tf.reduce_sum(words_used_in_sent, reduction_indices=1), tf.int32)
        output, _, _ = tf.nn.static_bidirectional_rnn(fw_cell, bw_cell,
                                                      tf.unstack(tf.transpose(self.input_data, perm=[1, 0, 2])),
                                                      dtype=tf.float32, sequence_length=self.length)
        weight, bias = self.weight_and_bias(2 * rnn_size, class_size)
        output = tf.reshape(tf.transpose(tf.stack(output), perm=[1, 0, 2]), [-1, 2 * rnn_size])
        prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
        self.prediction = tf.reshape(prediction, [-1, sentence_length, class_size])
        self.loss = self.cost()
        optimizer = tf.train.AdamOptimizer(learning_rate)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 10)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def cost(self):
        cross_entropy = self.output_data * tf.log(self.prediction)
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
        mask = tf.sign(tf.reduce_max(tf.abs(self.output_data), reduction_indices=2))
        cross_entropy *= mask
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        cross_entropy /= tf.cast(self.length, tf.float32)
        return tf.reduce_mean(cross_entropy)

    @staticmethod
    def weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)


# evaluate the prediction of the model
def f1(prediction, target, length):
    tp = np.array([0] * (class_size + 1))
    fp = np.array([0] * (class_size + 1))
    fn = np.array([0] * (class_size + 1))
    target = np.argmax(target, 2)
    prediction = np.argmax(prediction, 2)
    for i in range(len(target)):
        for j in range(length[i]):
            if target[i, j] == prediction[i, j]:
                tp[target[i, j]] += 1
            else:
                fp[target[i, j]] += 1
                fn[prediction[i, j]] += 1
    for i in range(class_size):
        tp[class_size] += tp[i]
        fp[class_size] += fp[i]
        fn[class_size] += fn[i]
    precision = []
    recall = []
    fscore = []
    for i in range(class_size):
        precision.append(tp[i] * 1.0 / (tp[i] + fp[i]))
        recall.append(tp[i] * 1.0 / (tp[i] + fn[i]))
        fscore.append(2.0 * precision[i] * recall[i] / (precision[i] + recall[i]))
    print(fscore)
    return fscore[class_size - 1]


def get_train_test_data():
    zero = []
    model = gensim.models.Word2Vec.load('D:\\word2vec\\word2vec_from_weixin\\word2vec\\word2vec_wx')
    for i in range(0, 256):
        zero.append(0.1)
    words = Word.objects.all()
    i = 0
    counter = 0
    inputs = []
    outputs = []
    word_sentence = []
    while i < len(words):
        input = np.empty((15, 256))
        output = np.empty((15, class_size))
        word_test = []
        tmp_words = words[i:i + 15]
        print(tmp_words)
        j = 0
        for word in tmp_words:
            try:
                input[j] = model[word.name]
                word_test.append(word.name)
            except KeyError:
                input[j] = np.array(zero)
                word_test.append(word.name)
            if word.div_type == 0:
                tmp = [1, 0]
            elif word.div_type == 1:
                tmp = [0, 1]
            else:
                tmp = [0, 1]
            output[j] = np.array(tmp)
            counter += 1
            j += 1
        inputs.append(input)
        outputs.append(output)
        word_sentence.append(word_test)
        i += 15
    split_point = int((len(words) / 15) * 0.8)
    train_input = inputs[0:split_point]
    train_output = outputs[0:split_point]
    test_input = inputs[split_point:(len(words) / 15) - 1]
    test_output = outputs[split_point:(len(words) / 15) - 1]
    word_sentence = word_sentence[split_point:(len(words) / 15) - 1]
    return np.array(train_input), np.array(train_output), np.array(test_input), np.array(test_output), word_sentence


def train():
    train_inp, train_out, test_a_inp, test_a_out, word_sentence = get_train_test_data()
    for sentence in word_sentence:
        print(sentence)
    model = Model()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        for e in range(epoch):
            for ptr in range(0, len(train_inp), batch_size):
                sess.run(model.train_op, {model.input_data: train_inp[ptr:ptr + batch_size],
                                          model.output_data: train_out[ptr:ptr + batch_size]})
            if e % 10 == 0:
                save_path = saver.save(sess, "./model/model.ckpt")
                print("model saved in file: %s" % save_path)
            pred, length = sess.run([model.prediction, model.length], {model.input_data: test_a_inp,
                                                                       model.output_data: test_a_out})
            print("epoch %d:" % e)
            print('test_a score:')
            f1(pred, test_a_out, length)
            for z in pred:
                print(z)