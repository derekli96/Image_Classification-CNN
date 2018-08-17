# coding: utf-8

import tensorflow as tf


class CNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 64  # 词向量维度
    seq_length = 600  # 序列长度
    num_classes = 2  # 类别数
    num_filters = 256  # 卷积核数目
    kernel_size = 5  # 卷积核尺寸
    vocab_size = 5000  # 词汇表达小

    hidden_dim = 128  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 8  # 每批训练大小
    num_epochs = 50  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard
    
    image_width = 50
    image_height = 50


class CNN(object):

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x1 = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x1')
        self.input_x2 = tf.placeholder(tf.float32, [8,50,50,3], name='input_x2')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.cnn()

    def cnn(self):
        """CNN模型"""
        # conv1, shape = [kernel_size, kernel_size, channels, kernel_numbers]
        with tf.variable_scope("conv_i1",reuse=tf.AUTO_REUSE) as scope:
            weights = tf.get_variable("weights",
                                      shape=[5, 5, 3, 8],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
            biases = tf.get_variable("biases",
                                     shape=[8],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(self.input_x2, weights, strides=[1, 1, 1, 1], padding="SAME")
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name="conv1")
        # pool1 && norm1
        with tf.variable_scope("pool_i1",reuse=tf.AUTO_REUSE) as scope:
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                   padding="SAME", name="pooling1")
            norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                              beta=0.75, name='norm1')

        # conv2
        with tf.variable_scope("conv_i2",reuse=tf.AUTO_REUSE) as scope:
            weights = tf.get_variable("weights",
                                      shape=[5, 5, 8, 8],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
            biases = tf.get_variable("biases",
                                     shape=[8],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding="SAME")
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_activation, name="conv2")

        # pool2 && norm2
        with tf.variable_scope("pool_i2",reuse=tf.AUTO_REUSE) as scope:
            pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                   padding="SAME", name="pooling2")
            norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                              beta=0.75, name='norm2')

        # full_connect1
        with tf.variable_scope("fc_i1",reuse=tf.AUTO_REUSE) as scope:
            reshape = tf.reshape(norm2, shape=[8, -1])
            dim = reshape.get_shape()[1].value
            weights = tf.get_variable("weights",
                                      shape=[dim, 128],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
            biases = tf.get_variable("biases",
                                     shape=[128],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name="fc1")
            fc1 = tf.contrib.layers.dropout(fc1, self.keep_prob)

        # full_connect2
        with tf.variable_scope("fc_i2",reuse=tf.AUTO_REUSE) as scope:
            weights = tf.get_variable("weights",
                                      shape=[128, 64],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
            biases = tf.get_variable("biases",
                                     shape=[64],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name="fc2")
            fc2 = tf.contrib.layers.dropout(fc2, self.keep_prob)

        # full_connect3 & softmax
        with tf.variable_scope("fc_i3",reuse=tf.AUTO_REUSE) as scope:
            weights = tf.get_variable("weights",
                                      shape=[64, 2],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
            biases = tf.get_variable("biases",
                                     shape=[2],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            self.logits = tf.add(tf.matmul(fc2, weights), biases, name="softmax_linear")
        
        with tf.variable_scope("softmax_i",reuse=tf.AUTO_REUSE) as scope:
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别
        
        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
