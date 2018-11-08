# Company Upstream and Downstream Analysis


## Background

+ Listed companies regularly disclose annual reports in each quarter to disclose the company's operating conditions. But these reports are too long for humans to read and extract effective information. So using some technology of data mining is very popular nowadays for people who want to have better knowledge of listed companies.
+ Most listed companies have their upstream industry and downstream industry. What’s more, their industries can have a huge effect on these companies, so it’s important to know upstream and downstream industries of each company. When we have that information, we can have a better analysis of the factors that influence the company’s profits and help us predict business revenue of certain companies.

## Requirement Analysis

+ First, we need to build a dictionary that contains vocabularies that represents certain kinds of industries. With the dictionary, we can get words of industries from annual reports, this is the prerequisite of our analysis.
+ Then, we need to identify whether these words are actually words that represent the company’s upstream and downstream industries.
+ Finally, show results in a file or on a website.

## Solution
+ First, we use word2vec as our basic technology to map words to vectors in order to do classification and other calculation, more will be illustrated later.
+ We will use some corpus to collect words that represent industry. I’ll do manual annotation and then use SVM to classify words.
+ Then, I will use LSTM to judge whether the word represents upstream or downstream industries.

## Running Environment And Depenencies
+ Python 3.6: Python is a popular computer language in machine learning and NLP for its easy grammar and rich packages. All the code are written via python.
+ Django: Django is a free and open-source web framework, written in Python, which follows the model-view-template (MVT) architectural pattern. It is maintained by the Django Software Foundation (DSF), an independent organization established as a non-profit. Actually, this project does not necessarily use Django. But considering Django provides us very convenient api to operate database and build a web site, we use it to improve our efficiency.
+ Numpy: Numpy is the fundamental package for scientific computing with Python.
+ jieba:
+ Tensorflow: TensorFlow is an open-source software library for dataflow programming across a range of tasks. It is a symbolic math library, and is also used for machine learning applications such as neural networks.

## Technology Used
+ Word2vec:
+ SVM:
+ LSTM:

## Data Set
+ Sohu news data: The data can be downloaded in [this website](http://www.sogou.com/labs/resource/cs.php)
+ Listed companies reports: All the reports are collected in [EastMoney](http://www.eastmoney.com/) This work is done by my classmates.
## Experiment
1. First, we need to train word to vector, the core code can be seen below
	

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

2. We use keyword "行业" to find words that represent industry. We find all the words that appear before the keyword. 

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
4. Then we annotate data artificially. Finally, we use SVM to classify the words.Then we annotate data artificially. Finally, we use SVM to classify the words.

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
## To Be Finished

## Code in [BusinessInteligenceFinalWork](https://github.com/caozixuan/BusinessInteligenceFinalWork)