# -*- coding: utf-8 -*-
import cv2
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import traceback


"""
数据集：RVL-CDIP（http://www.cs.cmu.edu/~aharley/rvl-cdip/）
输入图像尺寸：224x224x3

train dataset  32w
test dataset  4w
validation  4w

0 letter
1 form
2 email
3 handwritten
4 advertisement
5 scientific report
6 scientific publication
7 specification
8 file folder
9 news article
10 budget
11 invoice
12 presentation
13 questionnaire
14 resume
15 memo
"""


# 图像存储目录
IMG_PATH = r'E:\pycharm\tensorflow-learn\paper\datasets\RVL-CDIP\images'

# 图像信息
WIDTH = 224
HEIGHT = 224
CHANNEL = 3
IMG_TYPE = 'RGB'  # 与CHANNEL同步

# 与训练相关联参数
CATEGORY = 16
BATCH_SIZE = 32


# 对cv2读取的图像进行裁剪
# header:1 footer:2 leftbody:3 rightbody:4 all:0
def get_dest_body(img, body_id=0):
    sp = img.shape
    if body_id == 1:  # header
        up = 0
        down = sp[0] // 2
        left = 0
        right = sp[1]
    elif body_id == 2:  # footer
        up = sp[0] // 2
        down = sp[0]
        left = 0
        right = sp[1]
    elif body_id == 3:  # leftbody
        up = 0
        down = sp[0]
        left = 0
        right = sp[1] // 2
    elif body_id == 4:  # rightbody
        up = 0
        down = sp[0]
        left = sp[1] // 2
        right = sp[1]
    else:
        return img
    return img[up: down, left: right]


# 生成tfrecords格式数据集
def make_tfrecords_datasets(save_datasets_path, datasets_path, body_id=0):
    def _bytes_feature(value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    def _image_example(img_raw, label):
        feature = {
            'img_raw': _bytes_feature(img_raw),
            'label': _int64_feature(label),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    print('=============  正在生成tfrecords格式数据集  ==============')
    with tf.io.TFRecordWriter(save_datasets_path) as writer:
        with open(datasets_path, 'r') as datasets_read:
            while True:
                lines = datasets_read.readline()
                if not lines:
                    break
                try:
                    img_path, label = [i for i in lines.split(' ')]
                    img_path = os.path.join(IMG_PATH, img_path)
                    label = int(label)

                    img = cv2.imread(img_path)
                    img = get_dest_body(img, body_id)
                    img = cv2.resize(img, (WIDTH, HEIGHT))[:, :, ::-1]
                    image_string = cv2.imencode('.jpg', img)[1].tostring()

                    tf_example = _image_example(image_string, label)
                    writer.write(tf_example.SerializeToString())
                except Exception:
                    print(traceback.format_exc())
                    print('图片读取失败: ', img_path)
                    continue
    print('=============  成功生成tfrecords格式数据集  ==============')


# 获取数据集（tfrecords）
def get_dataset_by_tfrecords(tfrecords_path):
    def _argment_helper(img):
        img = tf.cast(img, tf.float32)
        img = tf.reshape(img, [WIDTH, HEIGHT, CHANNEL])
        img = tf.math.divide(img, tf.constant(255.0)) - 0.5  # 使之灰度值在 -0.5 到 0.5 之间
        return img
    def _parse_image_function(example_proto):
        feature = {
            'img_raw': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        }
        feature_dict = tf.io.parse_single_example(example_proto, feature)

        img = tf.image.decode_jpeg(feature_dict['img_raw'], CHANNEL)
        img = _argment_helper(img)
        label = tf.cast(feature_dict['label'], tf.int64)
        label = tf.one_hot(label, CATEGORY)
        return img, label

    dataset = tf.data.TFRecordDataset(tfrecords_path)
    dataset = dataset.repeat()
    # dataset = dataset.shuffle(buffer_size=512)

    import multiprocessing
    n_map_threads = multiprocessing.cpu_count()
    dataset = dataset.map(map_func=_parse_image_function, num_parallel_calls=n_map_threads)
    dataset = dataset.prefetch(buffer_size=BATCH_SIZE * 2)
    dataset = dataset.batch(batch_size=BATCH_SIZE)
    return dataset

# # 数据集少时可直接读取图片
# # 获取图像矩阵列表以及对应标签列表
# def read_img(datasets_path):
#     with open(datasets_path, 'r') as datasets_read:
#         imgs = []
#         labels = []
#         while True:
#             lines = datasets_read.readline()
#             if not lines:
#                 break
#             img_path, label = [i for i in lines.split(' ')]
#             img_path = os.path.join(IMG_PATH, img_path)
#             # print(img_path, label)
#             im = cv2.imread(img_path)
#             try:
#                 img = cv2.resize(im, (WIDTH, HEIGHT)) / 255.
#             except Exception:
#                 print('读取图片失败 +1')
#                 continue
#             imgs.append(img)
#             labels.append(label)
#     return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)
#
#
# # 数据增强及打乱数据
# def datasets_inhence(data, label):
#     image_gen_train = ImageDataGenerator(
#         rescale=1. / 1.,  # 如为图像，分母为255时，可归至0～1
#         rotation_range=45,  # 随机45度旋转
#         width_shift_range=.15,  # 宽度偏移
#         height_shift_range=.15,  # 高度偏移
#         horizontal_flip=False,  # 水平翻转
#         zoom_range=0.2  # 将图像随机缩放阈量50％
#     )
#     image_gen_train.fit(data)
#
#     num_example = data.shape[0]  # data.shape是(N, 100, 100, 3)
#     arr = np.arange(num_example)  # 创建等差数组 0，1，...,N
#     np.random.shuffle(arr)  # 打乱顺序
#     data = data[arr]
#     label = label[arr]
#     return data, label
#
#
# # 将所有标签转化为one_hot形式
# def to_one_hot(label):
#     return tf.one_hot(label, CATEGORY)
#
#
# # 获取数据集（直接将原图像存入list中）
# def get_dataset(datasets_path):
#     data, label = read_img(datasets_path)
#
#     data, label = datasets_inhence(data, label)
#
#     # 对标签进行处理
#     label = to_one_hot(label)
#     label = label.numpy()
#     return data, label
#
#
# # 返回训练集以及测试集
# def get_train_and_test_and_val_datasets(TRAIN_DATA_PATH, TEST_DATA_PATH, VAL_DATA_PATH):
#     print('===========  开始数据集读取  ==============')
#     if TRAIN_DATA_PATH != None:
#         train_x, train_y = get_dataset(TRAIN_DATA_PATH)
#     if TEST_DATA_PATH != None:
#         test_x, test_y = get_dataset(TEST_DATA_PATH)
#     if VAL_DATA_PATH != None:
#         val_x, val_y = get_dataset(VAL_DATA_PATH)
#     print('===========  完成数据集读取  ==============')
#     return train_x, train_y, test_x, test_y, val_x, val_y


#########################
if __name__ == '__main__':
    pass
    # # 生成tfrecords格式数据集
    # save_datasets_path = r'./datasets/rvl_cdip_train_header.tfrecords'
    # datasets_path = r'E:\pycharm\tensorflow-learn\paper\datasets\RVL-CDIP\labels\train.txt'
    # make_tfrecords_datasets(save_datasets_path, datasets_path, 1)
    #
    # save_datasets_path = r'./datasets/rvl_cdip_train_footer.tfrecords'
    # datasets_path = r'E:\pycharm\tensorflow-learn\paper\datasets\RVL-CDIP\labels\train.txt'
    # make_tfrecords_datasets(save_datasets_path, datasets_path, 2)
    # save_datasets_path = r'./datasets/rvl_cdip_test_footer.tfrecords'
    # datasets_path = r'E:\pycharm\tensorflow-learn\paper\datasets\RVL-CDIP\labels\test.txt'
    # make_tfrecords_datasets(save_datasets_path, datasets_path, 2)
    # save_datasets_path = r'./datasets/rvl_cdip_val_footer.tfrecords'
    # datasets_path = r'E:\pycharm\tensorflow-learn\paper\datasets\RVL-CDIP\labels\val.txt'
    # make_tfrecords_datasets(save_datasets_path, datasets_path, 2)
    #
    # save_datasets_path = r'./datasets/rvl_cdip_train_leftbody.tfrecords'
    # datasets_path = r'E:\pycharm\tensorflow-learn\paper\datasets\RVL-CDIP\labels\train.txt'
    # make_tfrecords_datasets(save_datasets_path, datasets_path, 3)
    # save_datasets_path = r'./datasets/rvl_cdip_test_leftbody.tfrecords'
    # datasets_path = r'E:\pycharm\tensorflow-learn\paper\datasets\RVL-CDIP\labels\test.txt'
    # make_tfrecords_datasets(save_datasets_path, datasets_path, 3)
    # save_datasets_path = r'./datasets/rvl_cdip_val_leftbody.tfrecords'
    # datasets_path = r'E:\pycharm\tensorflow-learn\paper\datasets\RVL-CDIP\labels\val.txt'
    # make_tfrecords_datasets(save_datasets_path, datasets_path, 3)
    #
    # save_datasets_path = r'./datasets/rvl_cdip_train_rightbody.tfrecords'
    # datasets_path = r'E:\pycharm\tensorflow-learn\paper\datasets\RVL-CDIP\labels\train.txt'
    # make_tfrecords_datasets(save_datasets_path, datasets_path, 4)
    # save_datasets_path = r'./datasets/rvl_cdip_test_rightbody.tfrecords'
    # datasets_path = r'E:\pycharm\tensorflow-learn\paper\datasets\RVL-CDIP\labels\test.txt'
    # make_tfrecords_datasets(save_datasets_path, datasets_path, 4)
    # save_datasets_path = r'./datasets/rvl_cdip_val_rightbody.tfrecords'
    # datasets_path = r'E:\pycharm\tensorflow-learn\paper\datasets\RVL-CDIP\labels\val.txt'
    # make_tfrecords_datasets(save_datasets_path, datasets_path, 4)
