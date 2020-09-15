from tensorflow.keras import Model
import tensorflow as tf

import get_datasets

# 图像设置
IMG_SHAPE = (get_datasets.WIDTH, get_datasets.HEIGHT, get_datasets.CHANNEL)

'''
holistic 无输出层模型
'''
class sub_holistic(Model):
    def __init__(self):
        super(sub_holistic, self).__init__()
        self.m_holistic = tf.keras.applications.VGG16(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
        self.m_holistic.trainable = False

        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.d0 = tf.keras.layers.Dropout(0.2)
        self.f1 = tf.keras.layers.Dense(128, activation="relu")
        self.d1 = tf.keras.layers.Dropout(0.2)
        self.out = tf.keras.layers.Dense(128, activation="relu")

    def call(self, x):
        x = self.m_holistic(x)
        x = self.gap(x)
        x = self.d0(x)
        x = self.f1(x)
        x = self.d1(x)
        x = self.out(x)
        return x
