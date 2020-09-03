# DN_Training_Model_Summary
Tensorflow_2.0.0 深度网络模型训练小结，含数据集制作与增强、模型创建与保存、可视化和模型训练及记录等

## Transfer_learning_small_datasets
> **运行环境：** win10 -- tensorflow 2.0.0 -- python 3.6

> **训练方法：** 使用小数据集的迁移学习模型，首先训练Holistic模型，接着训练Footer、Header、LeftBody和RightBody模型

> **数据集：** 数据集采用直接读入模型的方式，不加载为其他格式。进行了数据增强处理

> **参考：** 模型设计思路依据论文《Intra-Domain Transfer Learning and Stacked Generalization》
