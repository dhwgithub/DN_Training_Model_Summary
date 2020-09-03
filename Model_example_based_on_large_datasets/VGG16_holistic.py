from tensorflow.keras import Model
import tensorflow as tf

import get_datasets as get_datasets


IMG_SHAPE = (get_datasets.WIDTH, get_datasets.HEIGHT, get_datasets.CHANNEL)

class VGG16_holistic(Model):
    def __init__(self):
        super(VGG16_holistic, self).__init__()
        self.m_holistic = tf.keras.applications.VGG16(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
        self.m_holistic.trainable = False

        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.f1 = tf.keras.layers.Dense(128, activation="relu")
        self.d1 = tf.keras.layers.Dropout(0.5)
        self.f2 = tf.keras.layers.Dense(get_datasets.CATEGORY, activation="softmax")
    def call(self, x):
        x = self.m_holistic(x)
        x = self.gap(x)
        x = self.f1(x)
        x = self.d1(x)
        x = self.f2(x)
        return x
