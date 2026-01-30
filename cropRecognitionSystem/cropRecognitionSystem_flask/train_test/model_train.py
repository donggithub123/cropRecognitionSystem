# -*- coding: utf-8 -*-
# @Time : 2024-08-2024/8/18 20:55
# @Author : 林枫
# @File : model_train.py

from datetime import datetime, timezone
import pandas as pd
import pytz
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch
import torch.utils.data as Data
import matplotlib.pyplot as plt
import signal  # 在文件开头引入信号模块
from model.GoogLeNet_model import GoogLeNet, Inception
from model.MobileNetV2_model import MobileNetV2
from model.ResNet18_model import ResNet18, Residual
from model.VGG16Net_model import VGG16Net
from model.LeNet_model import LeNet
from model.AlexNet_model import AlexNet
import torch.nn as nn
import time
import copy

MODEL_CLASSES = {
    "GoogLeNet": {"model": GoogLeNet, "other": Inception},
    "ResNet18": {"model": ResNet18, "other": Residual},
    "MobileNetV2": {"model": MobileNetV2},
    "VGG16Net": {"model": VGG16Net},
    "LeNet": {"model": LeNet},
    "AlexNet": {"model": AlexNet},
}
# 输出对齐信息
device_label = "当前训练设备信息："
device_info_label = "训练设备："
device_name_label = "训练设备名称："
cuda_version_label = "CUDA版本："
# 获取设备名称，确保设备存在
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无可用GPU"
cuda_version = torch.version.cuda if torch.version.cuda else "未安装"


class ModelTrain:
    def __init__(self, model_name, num_epochs=5, pretrained_weights=None):
        """初始化训练类"""
        pretrained_weights = "../weights/" + pretrained_weights if pretrained_weights else None
        model_name = model_name.split('_')[0]
        model_info = MODEL_CLASSES.get(model_name)
        if not model_info:
            raise ValueError(f"模型名 '{model_name}' 不在可用模型列表中")  # 检查模型是否存在

        self.model_class = model_info["model"]
        print(f"\033[32m{"使用训练模型：":^50}\033[0m", f"\033[32m{model_info["model"].__name__:^70}\033[0m")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置训练设备
        # 打印训练设备信息
        print(f"\033[32m{device_label:^50}\033[0m")
        print(f"\033[32m{device_info_label:^50}\033[0m", f"\033[32m{str(self.device):^70}\033[0m")
        print(f"\033[32m{device_name_label:^50}\033[0m", f"\033[32m{device_name:^70}\033[0m")
        print(f"\033[32m{cuda_version_label:^50}\033[0m", f"\033[32m{cuda_version:^70}\033[0m")

        # 检查是否有其他类
        if "other" in model_info:
            self.other_class = model_info["other"]
            self.model = self.model_class(self.other_class).to(self.device)  # 实例化模型并移动到设备
        else:
            self.model = self.model_class().to(self.device)  # 这里需要加括号，实例化模型
        model_path = '../weights/Crop_' + model_info["model"].__name__ + '_model_{:.3f}%.pth'  # 模型保存路径格式
        self.model_path = model_path
        self.num_epochs = num_epochs

        self.best_model_weights = copy.deepcopy(self.model.state_dict())  # 记录当前最佳模型权重
        self.best_accuracy = 0.0  # 初始化最佳准确率
        # 注册信号处理程序
        signal.signal(signal.SIGINT, self.signal_handler)  # 捕获Ctrl+C
        signal.signal(signal.SIGTERM, self.signal_handler)  # 捕获终止信号

        # 加载预训练模型权重
        if pretrained_weights:
            try:
                self.model.load_state_dict(torch.load(pretrained_weights, map_location=self.device, weights_only=True))
                print(f"\033[32m{"加载预训练模型权重：":^50}\033[0m", f"\033[32m{pretrained_weights:^70}\033[0m")
            except Exception as e:
                print(f"加载预训练模型权重失败: {e}")

    def save_best_model(self):
        """信号处理函数，保存当前最佳模型"""
        print("\n接收到终止信号，正在保存最佳模型...")
        torch.save(self.best_model_weights, self.model_path.format(self.best_accuracy * 100))
        print(self.model_path.format(self.best_accuracy * 100) + "模型已保存。")
        exit(0)  # 退出程序

    def signal_handler(self, signum, frame):
        """信号处理程序"""
        self.save_best_model()  # 调用保存最佳模型的函数

    @staticmethod
    def get_current_time():
        """获取当前北京时间，格式为YYYY-MM-DD HH:MM:SS"""
        beijing_tz = pytz.timezone('Asia/Shanghai')  # 创建一个时区对象，代表北京时间
        utc_time = datetime.now(timezone.utc)  # 使用时区感知的UTC时间
        beijing_time = utc_time.astimezone(beijing_tz)  # 将UTC时间转换为北京时间
        return beijing_time.strftime('%Y-%m-%d %H:%M:%S')  # 格式化输出时间

    def load_train_val_data(self):
        """加载MNIST数据集，并进行训练集和验证集划分"""
        # 定义数据集的路径
        TRAN_ROOT_TRAIN = r'..\data\train'
        VAL_ROOT_TRAIN = r'..\data\val'
        # 定义数据集处理方法变量
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.16018272193439712, 0.1755668202902573, 0.10740333164906743],
                                 [0.04689176552845943, 0.05427002046013127, 0.03591450879001498]),  # 归一化处理
            transforms.Resize((224, 224))  # 调整图像大小
        ])
        # 加载数据集
        train_data = ImageFolder(TRAN_ROOT_TRAIN, transform=transform)
        val_data = ImageFolder(VAL_ROOT_TRAIN, transform=transform)

        # 划分训练集和验证集
        train_data_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=0)
        val_data_loader = Data.DataLoader(dataset=val_data, batch_size=64, shuffle=True, num_workers=0)

        return train_data_loader, val_data_loader  # 返回数据加载器

    def train_model(self, train_loader, val_loader):
        """训练模型，并进行验证"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)  # Adam优化器
        criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
        train_losses, val_losses = [], []  # 训练和验证损失列表
        train_accuracies, val_accuracies = [], []  # 训练和验证准确度列表
        total_start_time = time.time()  # 当前时间

        try:
            # 训练过程
            for epoch in range(self.num_epochs):
                epoch_start_time = time.time()  # 当前时间
                print('  ' + "-" * 70 + f"第 {epoch + 1} 轮训练开始（共 {self.num_epochs} 轮）" + "-" * 70)
                print(f"\033[31m第 {epoch + 1} 轮开始时间: {self.get_current_time()}\033[0m")  # 输出每轮开始时间

                # 初始化参数
                train_loss, train_corrects, val_loss, val_corrects = 0.0, 0, 0.0, 0  # 损失和正确数初始化
                train_count, val_count = 0, 0  # 训练和验证样本数初始化

                # 训练过程
                for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)  # 移动数据到设备
                    self.model.train()  # 设置模型为训练模式
                    outputs = self.model(batch_x)  # 前向传播
                    predicted_labels = torch.argmax(outputs, dim=1)  # 预测标签
                    loss = criterion(outputs, batch_y)  # 计算损失
                    optimizer.zero_grad()  # 清零梯度
                    loss.backward()  # 反向传播
                    optimizer.step()  # 更新参数

                    # 累计损失和正确数
                    train_loss += loss.item() * batch_x.size(0)
                    train_corrects += torch.sum(predicted_labels == batch_y.data)
                    train_count += batch_x.size(0)

                    # 输出训练进度
                    progress = (batch_idx + 1) / len(train_loader) * 100
                    print(f"\r训练进度: {progress:.2f}%", end='')

                print()  # 换行输出训练进度

                # 验证过程
                for batch_idx, (batch_x, batch_y) in enumerate(val_loader):
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)  # 移动数据到设备
                    self.model.eval()  # 设置模型为评估模式
                    outputs = self.model(batch_x)  # 前向传播
                    predicted_labels = torch.argmax(outputs, dim=1)  # 预测标签
                    loss = criterion(outputs, batch_y)  # 计算损失

                    # 累计损失和正确数
                    val_loss += loss.item() * batch_x.size(0)
                    val_corrects += torch.sum(predicted_labels == batch_y.data)
                    val_count += batch_x.size(0)

                    # 输出验证进度
                    progress = (batch_idx + 1) / len(val_loader) * 100
                    print(f"\r验证进度: {progress:.2f}%", end='')

                print()  # 换行输出验证进度

                # 记录损失和准确率
                train_losses.append(train_loss / train_count)
                train_accuracies.append(train_corrects.double().item() / train_count)
                val_losses.append(val_loss / val_count)
                val_accuracies.append(val_corrects.double().item() / val_count)

                # 输出每轮训练和验证结果
                print(f"\033[31m{epoch + 1} train loss: {train_losses[-1]:.4f} train acc: {train_accuracies[-1]:.4f}\033[0m")
                print(f"\033[31m{epoch + 1} val loss: {val_losses[-1]:.4f} val acc: {val_accuracies[-1]:.4f}\033[0m")
                print(f"\033[31m第 {epoch + 1} 轮结束时间: {self.get_current_time()}\033[0m")  # 输出每轮结束时间

                # 保存最佳模型
                if val_accuracies[-1] > self.best_accuracy:
                    self.best_accuracy = val_accuracies[-1]  # 更新最佳准确率
                    self.best_model_weights = copy.deepcopy(self.model.state_dict())  # 保存最佳模型参数

                # 计算和输出每轮耗时
                elapsed_time = time.time() - epoch_start_time
                hours, minutes, seconds = elapsed_time // 3600, (elapsed_time % 3600) // 60, elapsed_time % 60
                print(f"\033[33m训练和验证耗费的时间： {hours:.0f} h {minutes:.0f} m {seconds:.0f} s\033[0m")

        except Exception as e:
            print(f"\033[31m发生错误: {e}\033[0m")  # 捕获并输出错误信息

        # 选择最优参数并保存模型
        total_time = time.time() - total_start_time  # 计算总耗时
        hours, minutes, seconds = total_time // 3600, (total_time % 3600) // 60, total_time % 60
        print(f"\033[33m训练和验证耗费的总时间： {hours:.0f} h {minutes:.0f} m {seconds:.0f} s\033[0m")  # 输出总耗时

        torch.save(self.best_model_weights, self.model_path.format(self.best_accuracy * 100))
        # 返回训练过程数据
        train_process = pd.DataFrame(data={
            "epoch": range(self.num_epochs),
            "train_loss": train_losses,
            "val_loss": val_losses,
            "train_accuracy": train_accuracies,
            "val_accuracy": val_accuracies,
        })
        self.plot_training_results(train_process)  # 绘制结果

    @staticmethod
    def plot_training_results(train_process):
        """绘制训练和验证的损失及准确率曲线"""
        plt.figure(figsize=(12, 4))
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(train_process["epoch"], train_process.train_loss, 'ro-', label="训练损失")
        plt.plot(train_process["epoch"], train_process.val_loss, 'bs-', label="验证损失")
        plt.legend()
        plt.xlabel("轮次")
        plt.ylabel("损失")

        # 绘制准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(train_process["epoch"], train_process.train_accuracy, 'ro-', label="训练准确率")
        plt.plot(train_process["epoch"], train_process.val_accuracy, 'bs-', label="验证准确率")
        plt.legend()
        plt.xlabel("轮次")
        plt.ylabel("准确率")

        plt.show()  # 显示图形


if __name__ == '__main__':
    pretrained_weights = 'Fruit_ResNet18_model_85.831%.pth'  # 迁移学习，替换为实际的权重路径
    model_trainer = ModelTrain("ResNet18_model", num_epochs=3, pretrained_weights=pretrained_weights)  # 实例化训练类
    train_data_loader, val_data_loader = model_trainer.load_train_val_data()  # 加载数据
    model_trainer.train_model(train_data_loader, val_data_loader)
