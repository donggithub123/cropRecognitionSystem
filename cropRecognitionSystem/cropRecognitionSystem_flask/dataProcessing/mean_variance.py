# -*- coding: utf-8 -*-
# @Time : 2024-10-2024/10/16 23:44
# @Author : 林枫
# @File : mean_variance.py

import os
import numpy as np
from PIL import Image, UnidentifiedImageError
from pathlib import Path


class ImageStatistics:
    def __init__(self, folder_path):
        """
        初始化 ImageStatistics 类。

        :param folder_path: 包含图像的文件夹路径
        """
        self.folder_path = Path(folder_path)  # 使用 Path 对象
        self.total_pixels = 0  # 统计总像素数
        self.sum_normalized_pixel_values = np.zeros(3)  # RGB通道的归一化像素值和
        self.sum_squared_diff = np.zeros(3)  # 每个通道的平方差和

    def calculate_mean_and_variance(self):
        """
        计算指定文件夹中所有图像的均值和方差。

        :return: 均值和方差的元组
        """
        # 遍历文件夹中的图像文件
        for image_path in self.folder_path.rglob('*'):
            if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                print(f"Processing image: {image_path}")  # 打印正在处理的图像路径
                try:
                    image = Image.open(image_path)  # 打开图像文件
                    image_array = np.array(image)  # 将图像转换为numpy数组
                    # 归一化像素值到0-1之间
                    normalized_image_array = image_array / 255.0

                    # 计算像素总数
                    num_pixels = normalized_image_array.size  # 获取当前图像的总像素数
                    self.total_pixels += num_pixels  # 更新总像素数

                    # 累加归一化后的像素值和
                    self.sum_normalized_pixel_values += np.sum(normalized_image_array, axis=(0, 1))

                    # 计算当前均值与当前图像的平方差
                    current_mean = self.sum_normalized_pixel_values / self.total_pixels  # 当前均值
                    diff = normalized_image_array - current_mean  # 计算当前图像与均值的差值
                    self.sum_squared_diff += np.sum(diff ** 2, axis=(0, 1))  # 累加平方差
                except (UnidentifiedImageError, OSError) as e:
                    print(f"无法打开或识别图片文件：{image_path} - 错误：{e}")
                except Exception as e:
                    print(f"发生了其他错误：{e}")

        # 计算均值和方差
        if self.total_pixels > 0:
            mean = self.sum_normalized_pixel_values / self.total_pixels  # 计算均值
            variance = self.sum_squared_diff / self.total_pixels  # 计算方差
        else:
            print("没有有效图像文件，无法计算均值和方差。")
            mean = np.zeros(3)  # 返回全零均值
            variance = np.zeros(3)  # 返回全零方差

        return mean, variance  # 返回均值和方差


if __name__ == "__main__":
    # 创建 ImageStatistics 实例并计算训练集的均值和方差
    image_stats = ImageStatistics('../data/train/')  # 替换为你的绝对路径
    mean_train, variance_train = image_stats.calculate_mean_and_variance()

    # 打印结果，转换为列表格式
    print("Mean (Training):", mean_train.tolist())  # 打印训练集均值，转换为列表格式
    print("Variance (Training):", variance_train.tolist())  # 打印训练集方差，转换为列表格式
