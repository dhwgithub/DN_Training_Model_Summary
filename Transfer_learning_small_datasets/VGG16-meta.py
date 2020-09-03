import cv2
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 解决：Function call stack:
# distributed_function
gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

"""
数据集：RVL-CDIP（http://www.cs.cmu.edu/~aharley/rvl-cdip/）
输入图像尺寸：224x224x3
VGG16-meta：在RVL-CDIP数据集上进行最终训练得到，组合于其他五类模型
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

class Holistic(Model):
    def __init__(self):
        super(Holistic, self).__init__()
        self.m_holistic = tf.keras.applications.VGG19(input_shape=img_shape, include_top=False, weights='imagenet')
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.f2 = tf.keras.layers.Dense(64, activation="relu")
        self.f3 = tf.keras.layers.Dense(CATEGORY, activation="softmax")
    def call(self, x):
        x = self.m_holistic(x)
        x = self.gap(x)
        x = self.f2(x)
        x = self.f3(x)
        return x
VGG16_Holistic = Holistic()
VGG16_Holistic.trainable = False
# print(len(VGG16_Holistic.layers))  # 4
# fine_tune_at = 2
# for layer in VGG16_Holistic.layers[:fine_tune_at]:
#     layer.trainable = False

checkpoint_VGG16_Holistic_path = r"./model/VGG16-Holistic-model/VGG16-Holistic.ckpt"
if os.path.exists(checkpoint_VGG16_Holistic_path + '.index'):
    print('-------------load VGG16_Holistic model-----------------')
    VGG16_Holistic.load_weights(checkpoint_VGG16_Holistic_path)


# temp = tf.keras.Sequential([
#         tf.keras.applications.VGG16(input_shape=img_shape, include_top=False, weights='imagenet'),
#         tf.keras.layers.GlobalAveragePooling2D(),
#         tf.keras.layers.Dense(64, activation="relu")
#     ])
class Holistic(Model):
    def __init__(self):
        super(Holistic, self).__init__()
        self.m_holistic = tf.keras.applications.VGG19(input_shape=img_shape, include_top=False, weights='imagenet')
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.f2 = tf.keras.layers.Dense(64, activation="relu")
        # self.f3 = tf.keras.layers.Dense(CATEGORY, activation="softmax")
    def call(self, x):
        x = self.m_holistic(x)
        x = self.gap(x)
        x = self.f2(x)
        # x = self.f3(x)
        return x
temp = Holistic()
temp.trainable = False
path = r"./model/VGG16-Holistic-model/VGG16-Holistic.ckpt"
if os.path.exists(path + '.index'):
    temp.load_weights(path)

def get_pre_model(model_path, model_name):
    """
    :param model_path: .ckpt路径
    :return: 预训练模型
    """
    VGG16_model = tf.keras.Sequential([
        temp,
        tf.keras.layers.Dense(64, activation="relu"),
        # tf.keras.layers.Dropout(ratio),
        tf.keras.layers.Dense(CATEGORY, activation="softmax")
    ])

    # print(len(VGG16_model.layers))  # 3
    VGG16_model.trainable = False
    # fine_tune_at = 1
    # for layer in VGG16_model.layers[:fine_tune_at]:
    #     layer.trainable = False

    if os.path.exists(model_path + '.index'):
        print('-------------load ', model_name, ' model-----------------')
        VGG16_model.load_weights(model_path)
    return VGG16_model


VGG16_Header = get_pre_model(r"./model/VGG16-Header-model/VGG16-Header-model.ckpt", 'header')
VGG16_Footer = get_pre_model(r"./model/VGG16-Footer-model/VGG16-Footer.ckpt", 'footer')
VGG16_LeftBody = get_pre_model(r"./model/VGG16-LeftBody-model/VGG16-LeftBody.ckpt", 'leftbody')
VGG16_RightBody = get_pre_model(r"./model/VGG16-RightBody-model/VGG16-RightBody.ckpt", 'rightbody')

################  准备模型  #####################
class VGG16_meta(Model):
    def __init__(self, m_holistic, m_header, m_footer, m_leftbody, m_rightbody):
        super(VGG16_meta, self).__init__(m_holistic, m_header, m_footer, m_leftbody, m_rightbody)
        self.m_holistic = m_holistic
        self.m_header = m_header
        self.m_footer = m_footer
        self.m_leftbody = m_leftbody
        self.m_rightbody = m_rightbody

        self.d1 = tf.keras.layers.Dropout(0.2)
        self.f2 = tf.keras.layers.Dense(32, activation='relu')
        self.d2 = tf.keras.layers.Dropout(0.1)
        self.f3 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        x1 = self.m_holistic(x)
        x2 = self.m_header(x)
        x3 = self.m_footer(x)
        x4 = self.m_rightbody(x)
        x5 = self.m_leftbody(x)

        x = tf.concat([x1, x2, x3, x4, x5], 1)
        # print(x.shape)

        x = self.d1(x)
        x = self.f2(x)
        x = self.d2(x)
        x = self.f3(x)
        return x

model = VGG16_meta(VGG16_Holistic, VGG16_Header, VGG16_Footer, VGG16_LeftBody, VGG16_RightBody)

# 创建指数衰减学习率优化器,将其放入优化器即可
exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=0.001, decay_steps=70, decay_rate=0.96)
model.compile(optimizer=tf.keras.optimizers.Adam(exponential_decay),  # 学习率
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),  # 学习率
#               loss=tf.keras.losses.categorical_crossentropy,
#               metrics=[tf.keras.metrics.categorical_accuracy])

################  保存模型以及可视化  ###################
checkpoint_save_path = r"./model/VGG16-meta-model/VGG16-meta.ckpt"
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

################  开始训练  #####################
history = model.fit(x_train, y_train, batch_size=16, epochs=10, validation_split=0.1,
                    callbacks=[cp_callback, tensorBoard_callback])

model.summary()

################  测试  #####################
model.evaluate(x_val, y_val, verbose=2)
