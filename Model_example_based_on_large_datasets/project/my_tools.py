# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from matplotlib import pyplot as plt


'''
加载模型

checkpoint_save_path  模型加载路径
log_file  log文件对象
model  需要保存及加载的模型
'''
def save_and_load_models(checkpoint_save_path, log_file, model, name='当前模型'):
    if os.path.exists(checkpoint_save_path + '.index'):
        info = '加载' + name + '完成\n'
        log_file.write(info)
        model.load_weights(checkpoint_save_path)
        print('------------- 加载' + name + '完成 -----------------')


'''
编译模型

LEARNING_RATE  初始学习率
DECAY_STEPS  衰减步数
DECAY_RATE  衰减率
model  编译模型对象
'''
def compile_categorical_models(LEARNING_RATE, DECAY_STEPS, DECAY_RATE, model):
    exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LEARNING_RATE, decay_steps=DECAY_STEPS, decay_rate=DECAY_RATE)
    model.compile(optimizer=tf.keras.optimizers.Adam(exponential_decay),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=[tf.keras.metrics.categorical_accuracy])


'''
保存模型训练数据

checkpoint_save_path  保存路径
'''
def save_model_checkpoint(checkpoint_save_path):
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                     save_weights_only=True,
                                                     save_best_only=True)
    return cp_callback


'''
可视化配置参数

update_freq  更新频率
'''
def vision_by_tensorboard(update_freq='epoch'):
    # 设置存储路径可能导致程序报错
    tensorBoard_callback = TensorBoard(
        histogram_freq=1,
        write_graph=True,
        write_images=False,
        update_freq=update_freq,
        embeddings_freq=0,
    )
    return tensorBoard_callback


'''
可视化并保存为图片格式

'''
def vision_by_plt(history, loss__and_acc_img_path):
    # print("H.histroy keys：", history.history.keys())
    # H.histroy keys： dict_keys(['loss', 'categorical_accuracy', 'val_loss', 'val_categorical_accuracy'])

    # 显示训练集和验证集的acc和loss曲线
    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # print(acc)
    # print(val_acc)

    # print(loss)
    # print(val_loss)

    plt.close('all')

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()  # 为图像加上图例

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # loss__and_acc_img_path = './test_loss_and_acc_model.jpg'
    plt.savefig(loss__and_acc_img_path)
    # plt.show()

