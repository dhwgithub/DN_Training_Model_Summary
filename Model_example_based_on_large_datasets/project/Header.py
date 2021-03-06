# -*- coding: utf-8 -*-
import time
from tensorflow.keras import Model
import tensorflow as tf

import my_tools
import get_datasets as get_datasets
import Holistic
import sub_holistic


class header(Model):
    def __init__(self, holistic):
        super(header, self).__init__()
        self.holistic = holistic.layers[0]
        self.d1 = tf.keras.layers.Dropout(0.2)
        self.f1 = tf.keras.layers.Dense(64, activation="relu")
        self.d2 = tf.keras.layers.Dropout(0.2)
        self.out = tf.keras.layers.Dense(get_datasets.CATEGORY, activation="softmax")

    def call(self, x):
        x = self.holistic(x)
        x = self.d1(x)
        x = self.f1(x)
        x = self.d2(x)
        x = self.out(x)
        return x


def train_header_model():
    # # 解决：Function call stack:
    # # distributed_function
    # gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
    # tf.config.experimental.set_memory_growth(gpu[0], True)

    # 训练设置
    TRAIN_NUM = 320000  # 训练集数量
    VAL_NUM = 40000  # 验证集数量
    TEST_NUM = 40000  # 测试集数量
    BATCH_SIZE = get_datasets.BATCH_SIZE  # 批处理大小
    EPOCH_SIZE = 5  # 时期数
    LOG_FILE_PATH = './train_info.txt'  # 训练记录文件

    # 优化器设置
    LEARNING_RATE = 0.001
    DECAY_STEPS = TRAIN_NUM // BATCH_SIZE
    DECAY_RATE = 0.96


    print('===============================================  数据集准备  ===============================================')
    TRAIN_HEADER_DATA_PATH = r'.\datasets\header\rvl_cdip_train_header.tfrecords'
    TEST_HEADER_DATA_PATH = r'.\datasets\header\rvl_cdip_test_header.tfrecords'
    VAL_HEADER_DATA_PATH = r'.\datasets\header\rvl_cdip_val_header.tfrecords'

    train_data = get_datasets.get_dataset_by_tfrecords(TRAIN_HEADER_DATA_PATH)
    val_data = get_datasets.get_dataset_by_tfrecords(VAL_HEADER_DATA_PATH)
    test_data = get_datasets.get_dataset_by_tfrecords(TEST_HEADER_DATA_PATH)


    print('===============================================  加载模型  =================================================')
    log_file = open(LOG_FILE_PATH, 'a', encoding='utf-8')
    info = '========================================================================================================\n'\
           + str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) + '\n'
    log_file.write(info)

    holistic = Holistic.holistic(sub_holistic.sub_holistic())
    checkpoint_path = r"./model/Holistic-model/Holistic.ckpt"
    my_tools.save_and_load_models(checkpoint_path, log_file, holistic, 'holistic')

    model = header(holistic)
    checkpoint_save_path = r"./model/Header-model/Header.ckpt"
    my_tools.save_and_load_models(checkpoint_save_path, log_file, model, 'header')

    info = '当前模型（.ckpt）保存位置：' + checkpoint_save_path + '\n'
    log_file.write(info)


    print('=============================================  配置训练指标  ===============================================')
    # 编译模型
    my_tools.compile_categorical_models(LEARNING_RATE, DECAY_STEPS, DECAY_RATE, model)

    info = '采用衰减学习率，各参数为（初始学习率、衰减步数和衰减率）：' + str(LEARNING_RATE) + ' ' + str(DECAY_STEPS) + ' ' \
           + str(DECAY_RATE) + "\n"
    log_file.write(info)

    cp_callback = my_tools.save_model_checkpoint(checkpoint_save_path)

    # 可视化
    tensorBoard_callback = my_tools.vision_by_tensorboard()
    # tensorboard --logdir=.\logs


    print('================================================  训练模型  ================================================')
    # initial_epoch 整数，开始训练的时期（用于恢复以前的训练运行）。1是初始状态，但是不能对tensorboard续线
    history = model.fit(train_data, epochs=EPOCH_SIZE, steps_per_epoch=TRAIN_NUM // BATCH_SIZE,
                        validation_data=val_data, validation_steps=VAL_NUM // BATCH_SIZE,
                        callbacks=[cp_callback, tensorBoard_callback])

    # # 保存模型，与cp_callback二选一即可
    # model.save_weights(checkpoint_save_path)


    def add_log(d):
        print(d, file=log_file)


    model.summary(print_fn=add_log)

    info = '模型训练参数为（批次大小、训练轮数）：' + str(BATCH_SIZE) + ' ' + str(EPOCH_SIZE) + '\n'
    log_file.write(info)


    print('===============================================  可视化模型  ===============================================')
    loss__and_acc_img_path = r'./header_loss_and_acc_model.jpg'
    # 可视化并保存图片
    my_tools.vision_by_plt(history, loss__and_acc_img_path)

    info = '训练准确率和损失曲线保存于：' + loss__and_acc_img_path + "\n"
    log_file.write(info)


    print('================================================  测试模型  ================================================')
    res = model.evaluate(test_data, verbose=2, steps=TEST_NUM // BATCH_SIZE)  # steps 样本批次（非批次大小）

    info = '测试集结果(loss and acc)：' + str(res) + "\n"
    log_file.write(info)
    info = '\n=====================================================================================================\n\n'
    log_file.write(info)
    log_file.close()


if __name__ == '__main__':
    train_header_model()
