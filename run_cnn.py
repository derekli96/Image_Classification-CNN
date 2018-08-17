#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
from sklearn import metrics

import cnn_model
from data.data_process import read_vocab, read_category, batch_iter, process_file, build_vocab, get_image

base_dir = 'D:/data/JP/'
vocab_dir = os.path.join(base_dir, 'vocab.txt')
save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径
image_add = 'D:/data/profile_photo/'


def get_time_dif(start_time):
    #"""获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x1_batch, x2_batch, y_batch, keep_prob):
    
    feed_dict = {
        #model.input_x1: x1_batch,
        model.input_x2: x2_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, x1_, x2_, y_):
    #"""评估在某一数据上的准确率和损失"""
    data_len = len(x1_)
    batch_eval = batch_iter(x1_, x2_, y_, 8)
    total_loss = 0.0
    total_acc = 0.0
    for x1_batch, x2_batch, y_batch in batch_eval:
        batch_len = len(x1_batch)
        #print (batch_len)
        if batch_len == 8:
            feed_dict = feed_data(x1_batch, x2_batch, y_batch, 1.0)
            loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
            total_loss += loss * batch_len
            total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def train(val_set, test_set):
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'tensorboard/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()
    x1_train, x2_train, y_train = process_file(val_set, test_set, image_add, 0, word_to_id, cat_to_id, config.seq_length)
    x1_val, x2_val, y_val = process_file(val_set, test_set, image_add, 1, word_to_id, cat_to_id, config.seq_length)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x1_train, x2_train, y_train, config.batch_size)
        
        for x1_batch, x2_batch, y_batch in batch_train:
            #x2_batch = session.run(x2_batch)
            feed_dict = feed_data(x1_batch, x2_batch, y_batch, config.dropout_keep_prob)
            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x1_val, x2_val, y_val)

                if acc_val >= best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break

def test(val_set, test_set):
    print("Loading test data...")
    start_time = time.time()
    x1_test, x2_test, y_test = process_file(val_set, test_set, image_add, 2, word_to_id, cat_to_id, config.seq_length)
    x2_test=np.array(x2_test)
    images = []
    for i in x2_test:
        image = get_image(i)
        images.append(image)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(session, x1_test, x2_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = 8
    data_len = len(x1_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x1_test), dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x1: x1_test[start_id:end_id],
            model.input_x2: images[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)

    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        raise ValueError("""usage: python run_cnn.py [train / test]""")

    print('Configuring CNN model...')
    config = cnn_model.CNNConfig()
    test_set = 9
    val_set = 0
    
    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        build_vocab(val_set, test_set, image_add, vocab_dir, config.vocab_size)
   
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)
    config.vocab_size = len(words)
    model = cnn_model.CNN(config)
   
    if sys.argv[1] == 'train':
        train(val_set, test_set)
    else:
        test(val_set, test_set)
   