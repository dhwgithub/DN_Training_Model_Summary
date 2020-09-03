import cv2
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model

# 解决：Function call stack:
# distributed_function
gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

"""
数据集：RVL-CDIP（http://www.cs.cmu.edu/~aharley/rvl-cdip/）
输入图像尺寸：224x224x3
VGG16-Holistic：在RVL-CDIP数据集的完整图像进行训练（25 epoch）得到，其初始权重来自于从ImageNet数据集训练的VGG16模型
"""

# DATA_PATH = r'E:\pycharm\tensorflow-learn\paper\datasets\RVL-CDIP'
# 暂时用Tobacco3482数据集替代RVL-CDIP数据集
DATA_PATH = r'E:\pycharm\tensorflow-learn\cnn_learn\make_my_dataset\open_datasets\Tobacco3482'

WIDTH = 224
HEIGHT = 224
CHANNEL = 3
CATEGORY = 16

def read_img(path):
    imgs = []
    labels = []
    cate = [os.path.join(path, x) for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]  # 获取每类目录

    for idx, i in enumerate(cate):
        for j in os.listdir(i):  # 遍历每类目录文件
            im = cv2.imread(os.path.join(i, j))
            img = cv2.resize(im, (WIDTH, HEIGHT)) / 255.
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


def to_one_hot(label):
    return tf.one_hot(label, CATEGORY)


################  准备数据集  #####################
data, label = read_img(DATA_PATH)

image_gen_train = ImageDataGenerator(
    rescale=1. / 1.,  # 如为图像，分母为255时，可归至0～1
    rotation_range=45,  # 随机45度旋转
    width_shift_range=.15,  # 宽度偏移
    height_shift_range=.15,  # 高度偏移
    horizontal_flip=False,  # 水平翻转
    zoom_range=0.2  # 将图像随机缩放阈量50％
)
image_gen_train.fit(data)

num_example = data.shape[0]  # data.shape是(3029, 100, 100, 3)
arr = np.arange(num_example)  # 创建等差数组 0，1，...,3028
np.random.shuffle(arr)  # 打乱顺序
data = data[arr]
label = label[arr]

label_oh = to_one_hot(label)

ratio = 0.8
s = np.int(num_example * ratio)
x_train = data[:s]
y_train = label_oh.numpy()[:s]
x_val = data[s:]
y_val = label_oh.numpy()[s:]

################  准备预处理模型  #####################
img_shape = (WIDTH, HEIGHT, CHANNEL)
# base_model = tf.keras.applications.ResNet50(input_shape=img_shape, include_top=False, weights='imagenet')
# base_model.trainable = False

################  准备模型  #####################
class VGG16_holistic(Model):
    def __init__(self):
        super(VGG16_holistic, self).__init__()
        self.m_holistic = tf.keras.applications.VGG19(input_shape=img_shape, include_top=False, weights='imagenet')
        self.m_holistic.trainable = False

        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.f1 = tf.keras.layers.Dense(64, activation="relu")
        self.f2 = tf.keras.layers.Dense(CATEGORY, activation="softmax")
    def call(self, x):
        x = self.m_holistic(x)
        x = self.gap(x)
        x = self.f1(x)
        x = self.f2(x)
        return x

model = VGG16_holistic()

exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=0.01, decay_steps=20, decay_rate=0.96)
model.compile(optimizer=tf.keras.optimizers.Adam(exponential_decay),
               loss=tf.keras.losses.categorical_crossentropy,
               metrics=[tf.keras.metrics.categorical_accuracy])

################  保存模型以及可视化  ###################
checkpoint_save_path = r"./model/VGG16-Holistic-model/VGG16-Holistic.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)
tensorBoard_callback = TensorBoard(
    histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq='epoch',
    embeddings_freq=0,
)
# 设置存储路径可能导致程序报错
# tensorboard --logdir=E:\pycharm\tensorflow-learn\paper\Intra-Domain_Transfer_Learning_and_Stacked_Generalization\logs

################  开始训练（50 epoch）  #####################
history = model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.1,
                    callbacks=[cp_callback, tensorBoard_callback])

model.summary()

################  测试  #####################
model.evaluate(x_val, y_val, verbose=2)
