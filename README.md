# DN_Training_Model_Summary
Tensorflow_2.0.0 深度网络模型训练小结，含数据集制作与增强、模型创建与保存、可视化和模型训练及记录等

## Transfer_learning_small_datasets
> **运行环境：** win10 -- tensorflow 2.0.0 -- python 3.6

> **训练方法：** 使用小数据集的迁移学习模型，首先训练Holistic模型，接着训练Footer、Header、LeftBody和RightBody模型，最后训练综合以上5个模型的meta模型（模型实际是基于VGG19）

> **数据集：** 数据集采用直接读入模型的方式，不加载为其他格式。进行了数据增强处理

> **参考：** 模型设计思路依据论文《Intra-Domain Transfer Learning and Stacked Generalization》

## Model_example_based_on_large_datasets
> **运行环境：** win10 -- tensorflow 2.0.0 -- python 3.6

> **训练方法：** 使用大数据集RVL-CDIP的单个迁移学习模型，首先使用get_datasets.py制作自己的tfrecord格式数据集，然后运行test.py进行训练，训练的模型是VGG16_holistic.py。与*Transfer_learning_small_datasets*不同的是，该模型制作了属于自己的tfrecord数据集，同时读取的数据是迭代器，对于模型训练的方法稍有不同

> **注：** 训练结束后，不仅可以使用tensorboard进行可视化查看，而且会利用plt模型生成对应的acc和loss曲线图，以及记录训练参数信息的自定义txt文件




*初学者，如有理解偏差欢迎指正，有疑问可以采用主页联系方式交流学习！*
