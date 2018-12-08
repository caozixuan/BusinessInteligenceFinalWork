from django.shortcuts import render
import jieba
import os
import re
import numpy as np
import tensorflow as tf
import gensim
import jieba.posseg as pseg
from functools import reduce
import matplotlib.pyplot as plt
from industry.models import *
from industry.util import *

# 定义的基本关键词规则
rule_up = [r'(.*)上游(.*)', r'使用(.*)']
rule_down = [r'(.*)下游(.*)', r'(.*)行业下游情况(.*)', r'应用于(.*)']
rule_mid = [r'主营业务(.*)']
rule_company = r'(.*)股份有限公司'
fields = ['能源', '电力', '冶金', '化工', '机电', '电子', '交通', '房产', '建材', '医药', '农林', '安防', '服装', '包装',
          '环保', '玩具', 'IT', '通信', '数码', '家电', '家居', '文教', '办公', '金融', '培训', '旅游', '食品', '烟酒', '礼品']

# 数据文件路径
word2vec_model_path = 'D:\\word2vec\\word2vec_from_weixin\\word2vec\\word2vec_wx'
word2vec_train_path = 'F:\\data'    #效果不好，没有用到
word2vec_train_data = r'F:\train_data.txt'
sougo_news_souce_path = 'E:\\temp_data\\tmp'
sougo_news_save_path = r'D:\news.txt'
report_source_path = 'E:\\bin'
stop_word_path = 'F:\\stopword.txt'
data_path = 'F:\\tmp_data\\temp_data\\data'
# 获取某个文件夹下的所有文件
def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.txt':
                L.append(os.path.join(root, file))
    return L


# word2vec语料库分词
def train_data_build():
    file = word2vec_train_data
    names = file_name(word2vec_train_path)
    for name in names:
        f = open(name, errors='ignore')
        st = f.read()
        with open(file, 'a+') as f:
            seg_list = jieba.cut(st, cut_all=False)
            f.write(" ".join(seg_list))
            f.write('\n')
        f.close()


# word2vec语料库训练
def train_data():
    from gensim.models import word2vec
    sentences = word2vec.Text8Corpus('F:\\train_data.txt')
    model = word2vec.Word2Vec(sentences, size=50)
    model.save('word2vec_model')


# 搜狗语料库从XML中提取新闻内容
def getXMLContent():
    names = file_name(sougo_news_souce_path)
    rule1 = r'<contenttitle>(.+?)</contenttitle>'
    rule2 = r'<content>(.+?)</content>'
    file = sougo_news_save_path
    compile_name = re.compile(rule1, re.M)
    compile_name2 = re.compile(rule2, re.M)
    for name in names:
        f = open(name, errors='ignore')
        st = f.read()
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
    names = file_name(sougo_news_souce_path)
    for name in names:
        f = open(name, errors='ignore')
        st = f.read()
        res_name = compile_name.findall(st)
        for sentence in res_name:
            seg_list = jieba.lcut(sentence, cut_all=False)
            word = seg_list[len(seg_list) - 2]
            if len(word) <= 1:
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


# 支持向量机分类
def SVM():
    sess = tf.Session()
    words = Divided.objects.all()
    model = gensim.models.Word2Vec.load(word2vec_model_path)
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
    # saver = tf.train.Saver()
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
    # saver.save(sess, "./model/model.ckpt")
    print(train_accuracy)
    print(test_accuracy)
    plt.plot(loss_vec)
    plt.plot(train_accuracy)
    plt.plot(test_accuracy)
    plt.legend(['loss', 'train accuracy', 'test accuracy'])
    plt.ylim(0., 1.)
    plt.show()


# 分割出一个训练集
def buildDivided():
    counter = 238
    while counter < 1400:
        word = Dictionary.objects.get(id=counter)
        Divided.objects.get_or_create(name=word.name, is_industry=word.is_industry)
        counter = counter + 1


# 删除语料库中不存在的词汇
def delete_word_SVM():
    words = Dictionary.objects.all()
    model = gensim.models.Word2Vec.load(word2vec_model_path)
    for word in words:
        try:
            x = model[word.name]
        except  KeyError:
            print(word.name)
            Dictionary.objects.get(name=word.name).delete()


# 使用SVM模型训练
def predict_word():
    sess = tf.Session()
    model = gensim.models.Word2Vec.load(word2vec_model_path)
    # x_vals = np.array([model[keyword].tolist()])
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
        words = Dictionary.objects.all()
        counter = 0
        for result in predict_list:
            if result[0] == 1.0:
                Dictionary.objects.filter(id=words[counter].id).update(is_industry=True)
            counter = counter + 1
            # precision 95/104   recall:95/125


# 预测单个词
def predict_single_word_with_svm(word):
    sess = tf.Session()
    model = gensim.models.Word2Vec.load(word2vec_model_path)
    # x_vals = np.array([model[keyword].tolist()])
    x_vals = np.array([model[word]])
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
    flag = False
    with tf.Session() as sess:
        saver.restore(sess, "./model/model.ckpt")
        sess.run(prediction, feed_dict={x_data: x_vals}).tolist()
        predict_list = sess.run(prediction, feed_dict={x_data: x_vals}).tolist()
        words = Dictionary.objects.all()
        for result in predict_list:
            if result[0] == 1.0:
                flag = True
    return flag


# 判断词汇是否在词典中
def is_in_dic(keyword):
    word = Dictionary.objects.filter(name=keyword)
    if len(word) == 0:
        return False
    else:
        if word[0].is_industry:
            Industry.objects.get_or_create(name=word[0].name)
            return True
        else:
            return False


# 打印数据
def print_data_SVM():
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


# 收集企业报告中的数据，patter：0,1,2分别代表上游中游和下游
def collect_data(pattern):
    names = file_name(report_source_path)  # ('D:\\temp_data\\temp_data')
    if pattern == 0:
        for rule in rule_up:
            compile_name = re.compile(rule, re.M)
            company_rule = re.compile(rule_company, re.M)
            for name in names:
                # print (x+1)
                if os.path.getsize(name) / float(1024) < 100:
                    continue
                f = open(name, errors='ignore')
                st = f.read()
                res_name = compile_name.finditer(st)
                company_name = company_rule.finditer(st)
                y = 0
                name_company = ""
                for x in company_name:
                    if y == 0:
                        name_company = x.group()
                    y = y + 1
                if y > 0:
                    Company.objects.get_or_create(name=name_company)
                else:
                    continue
                for m in res_name:
                    seg_list = jieba.cut(m.group(), cut_all=False)
                    for word in seg_list:
                        industries = Industry.objects.all()
                        print(word)
                        is_industry = is_in_dic(word)
                        if not is_industry:
                            continue
                        elif is_industry:
                            for industry in industries:
                                if word == industry.name:
                                    company = Company.objects.filter(name=name_company)
                                    company[0].up_link.add(industry)
                                    company[0].save()
    elif pattern == 1:
        for rule in rule_mid:
            compile_name = re.compile(rule, re.M)
            company_rule = re.compile(rule_company, re.M)
            for name in names:
                # print (x+1)
                if os.path.getsize(name) / float(1024) < 100:
                    continue
                f = open(name, errors='ignore')
                st = f.read()
                res_name = compile_name.finditer(st)
                company_name = company_rule.finditer(st)
                y = 0
                name_company = ""
                for x in company_name:
                    if y == 0:
                        name_company = x.group()
                    y = y + 1
                if y > 0:
                    Company.objects.get_or_create(name=name_company)
                else:
                    continue
                for m in res_name:
                    seg_list = jieba.cut(m.group(), cut_all=False)
                    for word in seg_list:
                        industries = Industry.objects.all()
                        print(word)
                        is_industry = is_in_dic(word)
                        if not is_industry:
                            continue
                        elif is_industry:
                            for industry in industries:
                                if word == industry.name:
                                    company = Company.objects.filter(name=name_company)
                                    company[0].mid_link.add(industry)
                                    company[0].save()
    elif pattern == 2:
        for rule in rule_down:
            compile_name = re.compile(rule, re.M)
            company_rule = re.compile(rule_company, re.M)
            for name in names:
                # print (x+1)
                if os.path.getsize(name) / float(1024) < 100:
                    continue
                f = open(name, errors='ignore')
                st = f.read()
                res_name = compile_name.finditer(st)
                company_name = company_rule.finditer(st)
                y = 0
                name_company = ""
                for x in company_name:
                    if y == 0:
                        name_company = x.group()
                    y = y + 1
                if y > 0:
                    Company.objects.get_or_create(name=name_company)
                else:
                    continue
                for m in res_name:
                    seg_list = jieba.cut(m.group(), cut_all=False)
                    for word in seg_list:
                        industries = Industry.objects.all()
                        print(word)
                        is_industry = is_in_dic(word)
                        if not is_industry:
                            continue
                        elif is_industry:
                            for industry in industries:
                                if word == industry.name:
                                    company = Company.objects.filter(name=name_company)
                                    company[0].down_link.add(industry)
                                    company[0].save()


# 创建停用词词典
def buildStop():
    for line in open(stop_word_path):
        StopWord.objects.get_or_create(name=line)


def collectSentence():
    names = file_name(report_source_path)
    for name in names:
        if os.path.getsize(name) / float(1024) < 100:
            continue
        f = open(name, errors='ignore')
        st = f.read()
        punct = re.compile(r'[。？！：]')
        sentences = punct.split(st)
        words = Dictionary.objects.all()
        counter = 1
        for sentence in sentences:
            for word in words:
                if word.is_industry and word.name in sentence:
                    Sentence.objects.create(content=sentence)
                    print(counter)
                    counter = counter + 1
                    break


# keyword used to locate industry entity
rules = [r'(.*)上游(.*)', r'下游(.*)', r'(.*)应用于(.*)']
keywords = ['上游', '下游', '应用']




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
    model = gensim.models.Word2Vec.load(word2vec_model_path)
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


class Model(object):
    def __init__(self, name):
        self.type_size = 2
        self.word_size = 256
        self.lstm_size = 100
        # self.transe_size = 100
        self.dev = 0.01
        self.hidden_layer = 50
        self.window = 5
        self.scope = "root_train_second" if name == "KA+D" else "root"

        self.predict()
        self.saver = tf.train.Saver(max_to_keep=100)
        self.initializer = tf.global_variables_initializer()

    def entity(self):
        self.entity_in = tf.placeholder(tf.float32, [None, self.word_size])
        self.batch_size = tf.shape(self.entity_in)[0]
        self.kprob = tf.placeholder(tf.float32)
        entity_drop = tf.nn.dropout(self.entity_in, self.kprob)
        return entity_drop

    def attention(self):
        # this method will be overrided by derived classes
        pass

    def context(self):
        # from middle to side
        self.left_in = [tf.placeholder(tf.float32, [None, self.word_size]) \
                        for _ in range(self.window)]
        self.right_in = [tf.placeholder(tf.float32, [None, self.word_size]) \
                         for _ in range(self.window)]

        # from side to middle
        self.left_in_rev = [self.left_in[self.window - 1 - i] for i in range(self.window)]
        self.right_in_rev = [self.right_in[self.window - 1 - i] for i in range(self.window)]

        left_middle_lstm = tf.nn.rnn_cell.LSTMCell(self.lstm_size)
        right_middle_lstm = tf.nn.rnn_cell.LSTMCell(self.lstm_size)
        left_side_lstm = tf.nn.rnn_cell.LSTMCell(self.lstm_size)
        right_side_lstm = tf.nn.rnn_cell.LSTMCell(self.lstm_size)

        with tf.variable_scope(self.scope):
            with tf.variable_scope('lstm'):
                # from side to middle
                left_out_rev, _ = tf.nn.static_rnn(left_middle_lstm, self.left_in_rev, dtype=tf.float32)
            with tf.variable_scope('lstm', reuse=True):
                # from side to middle
                right_out_rev, _ = tf.nn.static_rnn(right_middle_lstm, self.right_in_rev, dtype=tf.float32)

                # from middle to side
                left_out, _ = tf.nn.static_rnn(left_side_lstm, self.left_in, dtype=tf.float32)
                right_out, _ = tf.nn.static_rnn(right_side_lstm, self.right_in, dtype=tf.float32)

        self.left_att_in = [tf.concat([left_out[i], left_out_rev[self.window - 1 - i]], 1) \
                            for i in range(self.window)]
        self.right_att_in = [tf.concat([right_out[i], right_out_rev[self.window - 1 - i]], 1) \
                             for i in range(self.window)]

        left_att, right_att = self.attention()

        left_weighted = reduce(tf.add,
                               [self.left_att_in[i] * left_att[i] for i in range(self.window)])
        right_weighted = reduce(tf.add,
                                [self.right_att_in[i] * right_att[i] for i in range(self.window)])

        left_all = reduce(tf.add, [left_att[i] for i in range(self.window)])
        right_all = reduce(tf.add, [right_att[i] for i in range(self.window)])

        return tf.concat([left_weighted / left_all, right_weighted / right_all], 1)

    def predict(self):
        # this method will be overrided by derived classes
        pass

    def fdict(self, now, size, interval, _entity, _context, _label):
        # this method will be overrided by derived classes
        pass

    def mag(self, matrix):
        return tf.reduce_sum(tf.pow(matrix, 2))

    def cross_entropy(self, predicted, truth):
        return -tf.reduce_sum(truth * tf.log(predicted + 1e-10)) \
               - tf.reduce_sum((1 - truth) * tf.log(1 - predicted + 1e-10))


class SA(Model):
    def attention(self):
        W1 = tf.Variable(tf.random_normal([self.lstm_size * 2, self.hidden_layer], stddev=self.dev))
        W2 = tf.Variable(tf.random_normal([self.hidden_layer, 1], stddev=self.dev))

        left_att = [tf.exp(tf.matmul(tf.tanh(tf.matmul(self.left_att_in[i], W1)), W2)) \
                    for i in range(self.window)]
        right_att = [tf.exp(tf.matmul(tf.tanh(tf.matmul(self.right_att_in[i], W1)), W2)) \
                     for i in range(self.window)]

        return (left_att, right_att)

    def predict(self):
        x = tf.concat([self.entity(), self.context()], 1)

        W = tf.Variable(tf.random_normal([self.word_size + self.lstm_size * 4, self.type_size],
                                         stddev=self.dev))
        self.t = tf.nn.sigmoid(tf.matmul(x, W))
        self.t_ = tf.placeholder(tf.float32, [None, self.type_size])

        self.loss = self.cross_entropy(self.t, self.t_)
        self.train = tf.train.AdamOptimizer(0.005).minimize(self.loss)

    def fdict(self, now, size, interval, _entity, _context, _label):
        fd = {}
        new_size = int(size / interval)

        ent = np.zeros([new_size, self.word_size])
        lab = np.zeros([new_size, self.type_size])
        for i in range(new_size):
            vec = _entity[now + i * interval]
            ent[i] = vec
            lab[i] = _label[now + i * interval]
        fd[self.entity_in] = ent
        fd[self.t_] = lab

        for j in range(self.window):
            left_con = np.zeros([new_size, self.word_size])
            right_con = np.zeros([new_size, self.word_size])
            for i in range(new_size):
                left_con[i, :] = _context[now + i * interval][2 * j]
                right_con[i, :] = _context[now + i * interval][2 * j + 1]
            fd[self.left_in[j]] = left_con
            fd[self.right_in[j]] = right_con

        return fd


batch_size = 500
iter_num = 2000
check_freq = 100


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


def delete_word():
    words = Context.objects.all()
    model = gensim.models.Word2Vec.load(word2vec_model_path)
    for word in words:
        entity = word.entity
        up1 = word.up1
        up2 = word.up2
        up3 = word.up3
        up4 = word.up4
        up5 = word.up5
        down1 = word.down1
        down2 = word.down2
        down3 = word.down3
        down4 = word.down4
        down5 = word.down5
        word_list = [entity, up1, up2, up3, up4, up5, down1, down2, down3, down4, down5]
        for x in word_list:
            if x == '#':
                continue
            try:
                y = model[x]
            except  KeyError:
                print(x)
                Context.objects.filter(entity=x).update(entity='#')
                Context.objects.filter(up1=x).update(up1='#')
                Context.objects.filter(up2=x).update(up2='#')
                Context.objects.filter(up3=x).update(up3='#')
                Context.objects.filter(up4=x).update(up4='#')
                Context.objects.filter(up5=x).update(up5='#')
                Context.objects.filter(down1=x).update(down1='#')
                Context.objects.filter(down2=x).update(down2='#')
                Context.objects.filter(down3=x).update(down3='#')
                Context.objects.filter(down4=x).update(down4='#')
                Context.objects.filter(down5=x).update(down5='#')


def delete_word2():
    words = ContextTest.objects.all()
    model = gensim.models.Word2Vec.load(word2vec_model_path)
    for word in words:
        entity = word.entity
        up1 = word.up1
        up2 = word.up2
        up3 = word.up3
        up4 = word.up4
        up5 = word.up5
        down1 = word.down1
        down2 = word.down2
        down3 = word.down3
        down4 = word.down4
        down5 = word.down5
        word_list = [entity, up1, up2, up3, up4, up5, down1, down2, down3, down4, down5]
        for x in word_list:
            if x == '#':
                continue
            try:
                y = model[x]
            except  KeyError:
                print(x)
                ContextTest.objects.filter(entity=x).update(entity='#')
                ContextTest.objects.filter(up1=x).update(up1='#')
                ContextTest.objects.filter(up2=x).update(up2='#')
                ContextTest.objects.filter(up3=x).update(up3='#')
                ContextTest.objects.filter(up4=x).update(up4='#')
                ContextTest.objects.filter(up5=x).update(up5='#')
                ContextTest.objects.filter(down1=x).update(down1='#')
                ContextTest.objects.filter(down2=x).update(down2='#')
                ContextTest.objects.filter(down3=x).update(down3='#')
                ContextTest.objects.filter(down4=x).update(down4='#')
                ContextTest.objects.filter(down5=x).update(down5='#')


def delete_word3():
    words = Article.objects.all()
    model = gensim.models.Word2Vec.load(word2vec_model_path)
    for word in words:
        entity = word.entity
        up1 = word.up1
        up2 = word.up2
        up3 = word.up3
        up4 = word.up4
        up5 = word.up5
        down1 = word.down1
        down2 = word.down2
        down3 = word.down3
        down4 = word.down4
        down5 = word.down5
        word_list = [entity, up1, up2, up3, up4, up5, down1, down2, down3, down4, down5]
        for x in word_list:
            if x == '#':
                continue
            try:
                y = model[x]
            except  KeyError:
                print(x)
                Article.objects.filter(entity=x).update(entity='#')
                Article.objects.filter(up1=x).update(up1='#')
                Article.objects.filter(up2=x).update(up2='#')
                Article.objects.filter(up3=x).update(up3='#')
                Article.objects.filter(up4=x).update(up4='#')
                Article.objects.filter(up5=x).update(up5='#')
                Article.objects.filter(down1=x).update(down1='#')
                Article.objects.filter(down2=x).update(down2='#')
                Article.objects.filter(down3=x).update(down3='#')
                Article.objects.filter(down4=x).update(down4='#')
                Article.objects.filter(down5=x).update(down5='#')


def get_train_data():
    entities = []
    context = []
    label = []
    model = gensim.models.Word2Vec.load(word2vec_model_path)
    words = Context.objects.all()
    for word in words:
        entities.append(dic(model, word.entity))
        up1 = dic(model, word.up1)
        up2 = dic(model, word.up2)
        up3 = dic(model, word.up3)
        up4 = dic(model, word.up4)
        up5 = dic(model, word.up5)
        down1 = dic(model, word.down1)
        down2 = dic(model, word.down2)
        down3 = dic(model, word.down3)
        down4 = dic(model, word.down4)
        down5 = dic(model, word.down5)
        sentence = [up1, down1, up2, down2, up3, down3, up4, down4, up5, down5]
        context.append(sentence)
        if word.div_type == 0:
            label.append([1, 0])
        elif word.div_type == 1:
            label.append([0, 1])
        else:
            label.append([0, 1])
    return np.array(entities[0:4500]), np.array(context[0:4500]), np.array(label[0:4500])


def get_test_data():
    entities = []
    context = []
    label = []
    model = gensim.models.Word2Vec.load(word2vec_model_path)
    words = ContextTest.objects.all()
    for word in words:
        entities.append(dic(model, word.entity))
        up1 = dic(model, word.up1)
        up2 = dic(model, word.up2)
        up3 = dic(model, word.up3)
        up4 = dic(model, word.up4)
        up5 = dic(model, word.up5)
        down1 = dic(model, word.down1)
        down2 = dic(model, word.down2)
        down3 = dic(model, word.down3)
        down4 = dic(model, word.down4)
        down5 = dic(model, word.down5)
        sentence = [up1, down1, up2, down2, up3, down3, up4, down4, up5, down5]
        context.append(sentence)
        if word.div_type == 0:
            label.append([1, 0])
        elif word.div_type == 1:
            label.append([0, 1])
        else:
            label.append([0, 1])
    return np.array(entities), np.array(context), np.array(label)


def get_article_data():
    entities = []
    context = []
    label = []
    model = gensim.models.Word2Vec.load(word2vec_model_path)
    words = Article.objects.all()
    for word in words:
        entities.append(dic(model, word.entity))
        up1 = dic(model, word.up1)
        up2 = dic(model, word.up2)
        up3 = dic(model, word.up3)
        up4 = dic(model, word.up4)
        up5 = dic(model, word.up5)
        down1 = dic(model, word.down1)
        down2 = dic(model, word.down2)
        down3 = dic(model, word.down3)
        down4 = dic(model, word.down4)
        down5 = dic(model, word.down5)
        sentence = [up1, down1, up2, down2, up3, down3, up4, down4, up5, down5]
        context.append(sentence)
        if word.div_type == 0:
            label.append([1, 0])
        elif word.div_type == 1:
            label.append([0, 1])
        else:
            label.append([0, 1])
    return np.array(entities), np.array(context), np.array(label)


def analysis():
    name = 'F:\\tmp_data\\temp_data\\doc.txt'
    f = open(name, errors='ignore')
    st = f.read()
    for j in range(0, len(rules)):
        compile_name = re.compile(rules[j], re.M)
        res_name = compile_name.finditer(st)
        for m in res_name:
            seg_list = jieba.lcut(m.group(), cut_all=False)
            if len(seg_list) < 6:
                continue
            target_list = []
            for w in seg_list:
                if len(w) > 1:
                    target_list.append(w)
            print(target_list)
            if len(target_list) < 5:
                break
            for x in range(1, 6):
                try:
                    target = target_list.index(keywords[j]) + x
                except ValueError:
                    continue
                try:
                    tmp_data = target_list[target]
                except IndexError:
                    break
                words_in_sentence = []
                index = target - 5
                for i in range(11):
                    if index == target:
                        index += 1
                        continue
                    if index >= 0:
                        try:
                            words_in_sentence.append(target_list[index])
                        except IndexError:
                            words_in_sentence.append("#")
                    else:
                        words_in_sentence.append("#")
                    index += 1
                ContextTest.objects.create(entity=target_list[target], up1=words_in_sentence[0],
                                           up2=words_in_sentence[1], up3=words_in_sentence[2],
                                           up4=words_in_sentence[3], up5=words_in_sentence[4],
                                           down1=words_in_sentence[5], down2=words_in_sentence[6],
                                           down3=words_in_sentence[7], down4=words_in_sentence[8],
                                           down5=words_in_sentence[9])


def all_predict():
    words = ContextTest.objects.all()
    model = SA('SA')
    sess = tf.Session()
    sess.run(model.initializer)
    model.saver.restore(sess, 'model/test2_model')
    valid_entity, valid_context, valid_label = get_test_data()
    fd = model.fdict(0, len(valid_entity), 1, valid_entity, valid_context, valid_label)
    fd[model.kprob] = 1.0
    showy = sess.run(model.t, feed_dict=fd)
    showy = np.argmax(showy, 1)
    print(len(words))
    print(len(showy))
    i = 0
    for word in words:
        if word.div_type == 1:
            print("行业词：" + word.entity)
        elif word.div_type == 2:
            print("行业词：" + word.entity)
    print("****************")
    for x in showy:
        if x == 1:
            print("行业词：" + words[i].entity)
        elif x == 2:
            print("行业词：" + words[i].entity)
        i += 1


def delete_non_sense_word2():
    words = Article.objects.all()
    for word in words:
        is_sense = True
        word_list = pseg.cut(word.entity)
        # words类别为：generator
        for x, flag in word_list:
            if flag != 'n' and flag != 'v' and flag != 'vn' and flag != 'j':
                is_sense = False
        if not is_sense:
            Article.objects.filter(entity=word.entity).delete()


def article_predict(doc_name_include_path):
    f = open(doc_name_include_path, errors='ignore')
    st = f.read()
    company_rule = re.compile(rule_company, re.M)
    company_name = company_rule.finditer(st)
    sentence_list = re.split(r"；|，|？|。", st)
    for sentence in sentence_list:
        for j in range(0, len(keywords)):
            if keywords[j] in sentence:
                seg_list = jieba.lcut(sentence, cut_all=False)
                if len(seg_list) < 6:
                    continue
                target_list = []
                for w in seg_list:
                    if len(w) > 1:
                        target_list.append(w)
                print(target_list)
                if len(target_list) < 5:
                    break
                for x in range(1, 6):
                    try:
                        target = target_list.index(keywords[j]) + x
                    except ValueError:
                        continue
                    try:
                        tmp_data = target_list[target]
                    except IndexError:
                        break
                    words_in_sentence = []
                    index = target - 5
                    for i in range(11):
                        if index == target:
                            index += 1
                            continue
                        if index >= 0:
                            try:
                                words_in_sentence.append(target_list[index])
                            except IndexError:
                                words_in_sentence.append("#")
                        else:
                            words_in_sentence.append("#")
                        index += 1
                    if keywords[j] == '上游':
                        position = '上游'
                    elif keywords[j] == '下游':
                        position = '下游'
                    elif keywords[j] == '应用于':
                        position = '下游'
                    else:
                        position = '中游'
                    Article.objects.create(entity=target_list[target], position=position, up1=words_in_sentence[0],
                                           up2=words_in_sentence[1], up3=words_in_sentence[2],
                                           up4=words_in_sentence[3], up5=words_in_sentence[4],
                                           down1=words_in_sentence[5], down2=words_in_sentence[6],
                                           down3=words_in_sentence[7], down4=words_in_sentence[8],
                                           down5=words_in_sentence[9])

    delete_non_sense_word2()
    delete_word3()
    words = Article.objects.all()
    model = SA('SA')
    sess = tf.Session()
    sess.run(model.initializer)
    model.saver.restore(sess, 'model/bi_model')
    valid_entity, valid_context, valid_label = get_article_data()
    fd = model.fdict(0, len(valid_entity), 1, valid_entity, valid_context, valid_label)
    fd[model.kprob] = 1.0
    showy = sess.run(model.t, feed_dict=fd)
    showy = np.argmax(showy, 1)
    print(len(words))
    print(len(showy))
    i = 0
    print(company_name)
    for x in showy:
        print("行业词：" + words[i].entity + "  " + words[i].position)
        i += 1
        Article.objects.all().delete()


def view_all_industry_words_SVM():
    result = []
    words = Dictionary.objects.all()
    for word in words:
        if word.is_industry:
            result.append(word.name)
    for i in range(0,len(result)):
        print(result[i]+" ",end='')
        if i%10==0:
            print("")


def view_all_industry_words_lstm():
    words = Context.objects.all()
    for word in words:
        if word.div_type==1 or word.div_type==2:
            print(word.entity)
