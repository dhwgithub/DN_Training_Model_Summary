# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from matplotlib import pyplot as plt
import time

import get_datasets as get_datasets
import VGG16_holistic as holistic_model

# # 解决：Function call stack:
# # distributed_function
# gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
# tf.config.experimental.set_memory_growth(gpu[0], True)

# 训练设置
TRAIN_NUM = 25000  # 训练集数量
VAL_NUM = 5000  # 验证集数量
TEST_NUM = 5000  # 测试集数量
BATCH_SIZE = get_datasets.BATCH_SIZE  # 批处理大小
EPOCH_SIZE = 1  # 时期数
LOG_FILE_PATH = './test_train_info.txt'  # 训练记录文件
# 优化器设置
LEARNING_RATE = 0.01
DECAY_STEPS = TRAIN_NUM // BATCH_SIZE
DECAY_RATE = 0.96

print('===============================================  数据集准备  ===================================================')
# 该路径存储图像位置及标签类别
# TRAIN_DATA_PATH = r'E:\pycharm\tensorflow-learn\paper\datasets\RVL-CDIP\labels\train.txt'
# TEST_DATA_PATH = r'E:\pycharm\tensorflow-learn\paper\datasets\RVL-CDIP\labels\test.txt'
# VAL_DATA_PATH = r'E:\pycharm\tensorflow-learn\paper\datasets\RVL-CDIP\labels\val.txt'
#
# train_x, train_y, test_x, test_y, val_x, val_y = get_datasets.get_train_and_test_and_val_datasets(TRAIN_DATA_PATH, TEST_DATA_PATH, VAL_DATA_PATH)

TRAIN_DATA_PATH = r'E:\pycharm\tensorflow-learn\paper\Intra-Domain_Transfer_Learning_and_Stacked_Generalization\project\datasets\rvl_cdip_train.tfrecords'
TEST_DATA_PATH = r'E:\pycharm\tensorflow-learn\paper\Intra-Domain_Transfer_Learning_and_Stacked_Generalization\project\datasets\rvl_cdip_test.tfrecords'
VAL_DATA_PATH = r'E:\pycharm\tensorflow-learn\paper\Intra-Domain_Transfer_Learning_and_Stacked_Generalization\project\datasets\rvl_cdip_val.tfrecords'
MINI_VAL_DATA_PATH = r'E:\pycharm\tensorflow-learn\paper\Intra-Domain_Transfer_Learning_and_Stacked_Generalization\project\datasets\rvl_cdip_mini.tfrecords'
MINI_TEST_DATA_PATH = r'E:\pycharm\tensorflow-learn\paper\Intra-Domain_Transfer_Learning_and_Stacked_Generalization\project\datasets\rvl_cdip_mini_te.tfrecords'
MINI_TRAIN_DATA_PATH = r'E:\pycharm\tensorflow-learn\paper\Intra-Domain_Transfer_Learning_and_Stacked_Generalization\project\datasets\rvl_cdip_mini_tr.tfrecords'

train_data = get_datasets.get_dataset_by_tfrecords(MINI_TRAIN_DATA_PATH)
val_data = get_datasets.get_dataset_by_tfrecords(MINI_VAL_DATA_PATH)
test_data = get_datasets.get_dataset_by_tfrecords(MINI_TEST_DATA_PATH)

print('===============================================  加载模型  =====================================================')
log_file = open(LOG_FILE_PATH, 'a', encoding='utf-8')
info = '=========================================================================================================\n'\
       + str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) + '\n'
log_file.write(info)

model = holistic_model.VGG16_holistic()

# 保存与加载模型
checkpoint_save_path = r"./model/VGG16-Holistic-model/VGG16-Holistic.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    info = '加载上次保存的模型，'
    log_file.write(info)
    model.load_weights(checkpoint_save_path)

info = '当前模型（.ckpt）保存位置：' + checkpoint_save_path + '\n'
log_file.write(info)

print('=============================================  配置训练指标  ===================================================')
exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=LEARNING_RATE, decay_steps=DECAY_STEPS, decay_rate=DECAY_RATE)
model.compile(optimizer=tf.keras.optimizers.Adam(exponential_decay),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])

info = '采用衰减学习率，各参数为（初始学习率、衰减步数和衰减率）：' + str(LEARNING_RATE) + ' ' + str(DECAY_STEPS) + ' ' \
       + str(DECAY_RATE) + "\n"
log_file.write(info)

# 可视化
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)
tensorBoard_callback = TensorBoard(
    histogram_freq=1,
    write_graph=True,
    write_images=False,
    update_freq=BATCH_SIZE,
    embeddings_freq=0,
)
# 设置存储路径可能导致程序报错
# tensorboard --logdir=E:\pycharm\tensorflow-learn\paper\Intra-Domain_Transfer_Learning_and_Stacked_Generalization\project\logs

print('================================================  训练模型  ====================================================')
# validation_data=val_data
# initial_epoch 整数，开始训练的时期（用于恢复以前的训练运行）。1是初始状态，但是不能对tensorboard续线
history = model.fit(train_data, epochs=EPOCH_SIZE, steps_per_epoch=TRAIN_NUM // BATCH_SIZE,
                    validation_data=val_data, validation_steps=VAL_NUM // BATCH_SIZE,
                    callbacks=[cp_callback, tensorBoard_callback])


def add_log(d):
    print(d, file=log_file)


model.summary(print_fn=add_log)

info = '模型训练参数为（批次大小、训练轮数）：' + str(BATCH_SIZE) + ' ' + str(EPOCH_SIZE) + '\n'
log_file.write(info)

print('===============================================  可视化模型  ===================================================')
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

loss__and_acc_img_path = './test_loss_and_acc_model.jpg'
plt.savefig(loss__and_acc_img_path)
# plt.show()

info = '训练准确率和损失曲线保存于：' + loss__and_acc_img_path + "\n"
log_file.write(info)

print('================================================  测试模型  ====================================================')
res = model.evaluate(test_data, verbose=2, steps=TEST_NUM // BATCH_SIZE)  # steps 样本批次（非批次大小）

info = '测试集结果(loss and acc)：' + str(res) + "\n"
log_file.write(info)
info = '\n=========================================================================================================\n\n'
log_file.write(info)
log_file.close()
