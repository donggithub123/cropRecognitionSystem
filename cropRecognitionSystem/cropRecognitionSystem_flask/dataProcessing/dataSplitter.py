# -*- coding: utf-8 -*-
# @Time : 2024-10-2024/10/16 23:42
# @Author : 林枫
# @File : dataSplitter.py
import os
import random
import shutil

"""
将已化分为train和test的数据集划分成train、val和test的数据集
"""


class DataSplitter:
    def __init__(self, base_path, val_ratio=0.1):
        """初始化数据划分器
        :param base_path: 数据集的根目录
        :param val_ratio: 验证集占比（0 < val_ratio < 1）
        """
        self.base_path = base_path  # 设置根目录路径
        self.train_path = os.path.join(base_path, 'train')  # 训练集路径
        self.val_path = os.path.join(base_path, 'val')  # 验证集路径
        self.val_ratio = val_ratio  # 设置验证集比例

    def mkfile(self, directory):
        """创建目录"""
        if not os.path.exists(directory):  # 如果目录不存在
            os.makedirs(directory)  # 创建目录

    def split_data(self):
        """将训练集数据划分为训练集和验证集"""
        self.mkfile(self.val_path)  # 创建验证集目录

        # 遍历训练集的所有子目录（类别）
        for class_dir in os.listdir(self.train_path):
            class_path = os.path.join(self.train_path, class_dir)  # 当前类别路径
            if os.path.isdir(class_path):  # 检查是否为目录
                images = os.listdir(class_path)  # 获取当前类别下的所有图像
                num_images = len(images)  # 统计图像数量
                val_size = int(num_images * self.val_ratio)  # 计算验证集大小

                random.shuffle(images)  # 随机打乱图像列表

                val_images = images[:val_size]  # 选取前部分作为验证集
                train_images = images[val_size:]  # 剩余部分作为新的训练集

                # 创建当前类别的验证集目录
                self.mkfile(os.path.join(self.val_path, class_dir))

                # 复制验证集图像到新目录
                for image in val_images:
                    shutil.copy(os.path.join(class_path, image), os.path.join(self.val_path, class_dir, image))

                print(f"Moved {val_size} images to {self.val_path}/{class_dir}.")  # 打印每个类别的信息

        print("All processing done!")  # 打印所有处理完成的信息


if __name__ == "__main__":
    base_path = '../data'  # 设置数据根目录
    val_ratio = 0.1  # 设置验证集占比（可根据需要修改）

    splitter = DataSplitter(base_path, val_ratio)  # 创建数据划分器实例
    splitter.split_data()  # 开始划分数据
