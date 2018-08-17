# coding: utf-8

import os
import sys
from collections import Counter
import numpy as np
import tensorflow.contrib.keras as kr
import tensorflow as tf
from PIL import Image


base_dir = 'D:/data/JP/'

if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False


def native_word(word, encoding='utf-8'):
    """如果在python2下面使用python3训练的模型，可考虑调用此函数转化一下字符编码"""
    if not is_py3:
        return word.encode(encoding)
    else:
        return word


def native_content(content):
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content


def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)


def read_file(val_set, test_set, image_dir, set_type):
    """读取文件数据"""
    contents, images_address, labels = [], [], []
    if set_type == 0:
        for i in range(0, 10):
            if i != val_set and i != test_set:
                text_set = os.path.join(base_dir, 'textset_'+ str(i) +'.txt')
                image_set = os.path.join(base_dir, 'imageset_'+ str(i) +'.txt')
                with open_file(text_set) as f:
                    for line in f:
                        try:
                            label, content = line.strip().split('\t')
                            if content:
                                contents.append(list(native_content(content)))
                                labels.append(native_content(label))
                        except:
                            pass
                with open_file(image_set) as u:
                    for line in u:
                        try:
                            user = line.strip()
                            images_address.append(image_dir+user+'.png')
                        except:
                            pass
    
    elif set_type == 1:
        text_set = os.path.join(base_dir, 'textset_'+ str(val_set) +'.txt')
        image_set = os.path.join(base_dir, 'imageset_'+ str(val_set) +'.txt')
        with open_file(text_set) as f:
            for line in f:
                try:
                    label, content = line.strip().split('\t')
                    if content:
                        contents.append(list(native_content(content)))
                        labels.append(native_content(label))
                except:
                    pass
        with open_file(image_set) as u:
            for line in u:
                try:
                    user = line.strip()
                    images_address.append(image_dir+user+'.png')
                except:
                    pass
    
    elif set_type == 2:
        text_set = os.path.join(base_dir, 'textset_'+ str(test_set) +'.txt')
        image_set = os.path.join(base_dir, 'imageset_'+ str(test_set) +'.txt')
        with open_file(text_set) as f:
            for line in f:
                try:
                    label, content = line.strip().split('\t')
                    if content:
                        contents.append(list(native_content(content)))
                        labels.append(native_content(label))
                except:
                    pass
        with open_file(image_set) as u:
            for line in u:
                try:
                    user = line.strip()
                    images_address.append(image_dir+user+'.png')
                except:
                    pass
    
                 
    return contents, images_address, labels


def build_vocab(val_set, test_set, image_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    data_train, _, _ = read_file(val_set, test_set, image_dir, 0)
    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    words = ['<PAD>'] + list(words)
    #添加<PAD>将所有文本pad为同一长度
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open_file(vocab_dir) as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [native_content(_.strip()) for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category():
    """读取分类目录，固定"""
    categories = ['J', 'P']
    categories = [native_content(x) for x in categories]
    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)


def process_file(val_set, test_set, image_dir, process_type, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, images, labels = read_file(val_set, test_set, image_dir, process_type)

    text_id, image_id, label_id = [], [], []
    for i in range(len(contents)):
        text_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(text_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示
    
    data_len = len(labels)
    batch_num = int(data_len / 8)
    return x_pad[:batch_num*8], images[:batch_num*8], y_pad[:batch_num*8]

def get_image(img_dir):
    image = Image.open(img_dir)
    # Image.open()
    if image.mode == 'P':
        image = image.convert("RGB")
    elif image.mode == 'L':
        image = image.convert("RGB")
    image = image.resize([50, 50])
    image_arr = np.array(image)
    return image_arr

def batch_iter(x1, x2, y, batch_size=8):
    """生成批次数据"""
    data_len = len(x1)
    num_batch = int((data_len - 1) / batch_size) + 1
    indices = np.random.permutation(np.arange(data_len))
    x1_shuffle = x1[indices]
    x2_shuffle = np.array(x2)[indices]
    y_shuffle = y[indices]
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        
        images = []
        for i in x2_shuffle[start_id:end_id]:
            image = get_image(i)
            images.append(image)
                                    
        #input_queue = tf.train.slice_input_producer([x2_shuffle,y])
        #images_contents = tf.read_file(input_queue[0])
        #images = tf.image.decode_png(images_contents,channels=3)
        #images = tf.image.resize_image_with_crop_or_pad(images,50,50)
        #x2_shuffle = tf.image.per_image_standardization(images)
        yield x1_shuffle[start_id:end_id], images, y_shuffle[start_id:end_id]
