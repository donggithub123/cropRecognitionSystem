# -*- coding: utf-8 -*-
# @Time : 2024-10-2024/10/16 23:42
# @Author : 林枫
# @File : dataPartitioner.py
import os  # 导入os模块，用于文件和目录操作
import random  # 导入random模块，用于随机操作
import shutil  # 导入shutil模块，用于文件复制等操作

"""
将初始数据集划分成train、val和test的数据集
"""


class DataPartitioner:
    def __init__(self, dataset_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """初始化数据划分器类
        :param dataset_path: 数据集路径
        :param train_ratio: 训练集占比
        :param val_ratio: 验证集占比
        :param test_ratio: 测试集占比
        """
        self.dataset_path = dataset_path  # 设置数据集路径
        self.train_ratio = train_ratio  # 设置训练集比例
        self.val_ratio = val_ratio  # 设置验证集比例
        self.test_ratio = test_ratio  # 设置测试集比例

    def mkfile(self, file):
        """创建目录
        :param file: 需要创建的目录路径
        """
        if not os.path.exists(file):  # 检查目录是否存在
            os.makedirs(file)  # 如果不存在，创建该目录

    def get_flower_classes(self):
        """获取数据集中的类别
        :return: 类别名称列表
        """
        return [cla for cla in os.listdir(self.dataset_path) if
                os.path.isdir(os.path.join(self.dataset_path, cla))]  # 返回数据集目录下的所有子目录（类别名）

    def partition_data(self):
        """划分数据集为训练集、验证集和测试集"""
        flower_classes = self.get_flower_classes()  # 获取所有类别

        # 创建训练集、验证集和测试集目录
        self.mkfile('../data/train')  # 创建训练集目录
        self.mkfile('../data/val')  # 创建验证集目录
        self.mkfile('../data/test')  # 创建测试集目录

        # 遍历每个类别，划分图像
        for cla in flower_classes:  # 遍历每个类别
            cla_path = os.path.join(self.dataset_path, cla)  # 获取当前类别的路径
            images = os.listdir(cla_path)  # 获取当前类别下的所有图像
            num = len(images)  # 统计当前类别下图像的数量

            # 随机抽样，划分数据
            random.shuffle(images)  # 随机打乱图像列表
            train_end = int(num * self.train_ratio)  # 训练集结束索引
            val_end = train_end + int(num * self.val_ratio)  # 验证集结束索引

            train_images = images[:train_end]  # 训练集图像列表
            val_images = images[train_end:val_end]  # 验证集图像列表
            test_images = images[val_end:]  # 测试集图像列表

            # 复制图像到对应目录
            for image in train_images:  # 遍历训练集图像
                target_dir = os.path.join('../data/train', cla)  # 目标路径
                self.mkfile(target_dir)  # 确保目标目录存在
                shutil.copy(os.path.join(cla_path, image), os.path.join(target_dir, image))  # 复制图像

            for image in val_images:  # 遍历验证集图像
                target_dir = os.path.join('../data/val', cla)  # 目标路径
                self.mkfile(target_dir)  # 确保目标目录存在
                shutil.copy(os.path.join(cla_path, image), os.path.join(target_dir, image))  # 复制图像

            for image in test_images:  # 遍历测试集图像
                target_dir = os.path.join('../data/test', cla)  # 目标路径
                self.mkfile(target_dir)  # 确保目标目录存在
                shutil.copy(os.path.join(cla_path, image), os.path.join(target_dir, image))  # 复制图像

            print(f"Class [{cla}] processing done! ({num} images)")  # 打印当前类别处理完成的信息

        print("All processing done!")  # 打印所有处理完成的信息


if __name__ == "__main__":
    dataset_path = '../Rice_Image_Dataset'  # 数据集目录
    partitioner = DataPartitioner(dataset_path)  # 创建数据划分器实例
    partitioner.partition_data()  # 开始划分数据
