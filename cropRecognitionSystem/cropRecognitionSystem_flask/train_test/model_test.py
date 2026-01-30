# -*- coding: utf-8 -*-
# @Time : 2024-08-2024/8/20 11:37
# @Author : 林枫
# @File : model_test.py

from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch
import torch.utils.data as Data
from model.GoogLeNet_model import GoogLeNet, Inception
from model.MobileNetV2_model import MobileNetV2
from model.ResNet18_model import ResNet18, Residual
from model.VGG16Net_model import VGG16Net
from model.LeNet_model import LeNet
from model.AlexNet_model import AlexNet

MODEL_CLASSES = {
    "GoogLeNet": {"model": GoogLeNet, "other": Inception},
    "ResNet18": {"model": ResNet18, "other": Residual},
    "MobileNetV2": {"model": MobileNetV2},
    "VGG16Net": {"model": VGG16Net},
    "LeNet": {"model": LeNet},
    "AlexNet": {"model": AlexNet},
}


class ModelTest:
    def __init__(self, model_name, weights):
        """初始化模型和数据处理方法"""
        if not weights:
            raise ValueError("权重不可为空")  # 检查模型是否存在
        weights = "../weights/" + weights
        model_name = model_name.split('_')[0]
        model_info = MODEL_CLASSES.get(model_name)
        if not model_info:
            raise ValueError(f"模型名 '{model_name}' 不在可用模型列表中")  # 检查模型是否存在

        self.model_class = model_info["model"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置训练设备
        # 检查是否有其他类
        if "other" in model_info:
            self.other_class = model_info["other"]
            self.model = self.model_class(self.other_class).to(self.device)  # 实例化模型并移动到设备
        else:
            self.model = self.model_class().to(self.device)  # 这里需要加括号，实例化模型
        self.model.load_state_dict(torch.load(weights, weights_only=False))
        self.classes = ['小麦', '水稻', '玉米', '甘蔗', '黄麻']

    def test_data_process(self):
        """数据预处理，加载测试数据集"""
        # 定义数据集的路径
        ROOT_TRAIN = r'..\data\test'
        # 定义数据集处理方法变量
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.16018272193439712, 0.1755668202902573, 0.10740333164906743],
                                 [0.04689176552845943, 0.05427002046013127, 0.03591450879001498]),  # 归一化处理
            transforms.Resize((224, 224))  # 调整图像大小
        ])
        # 加载数据集
        data = ImageFolder(ROOT_TRAIN, transform=transform)
        test_dataloader = Data.DataLoader(dataset=data, batch_size=256, shuffle=True, num_workers=0)
        return test_dataloader

    def test_model_process(self, test_dataloader):
        """测试模型，计算准确率并输出预测失败的信息"""
        test_corrects = 0.0
        test_num = 0

        with torch.no_grad():
            self.model.eval()  # 设置模型为评估模式
            for test_data_x, test_data_y in test_dataloader:
                test_data_x = test_data_x.to(self.device)  # 特征放入到测试设备中
                test_data_y = test_data_y.to(self.device)  # 标签放入到测试设备中

                output = self.model(test_data_x)  # 前向传播过程
                pre_lab = torch.argmax(output, dim=1)  # 预测标签
                test_corrects += torch.sum(pre_lab == test_data_y.data).item()  # 更新正确预测数量
                test_num += test_data_x.size(0)  # 更新测试样本总数

                # 输出预测失败的信息
                for i in range(test_data_x.size(0)):
                    if pre_lab[i] != test_data_y[i]:  # 仅当预测失败时输出
                        outputs = f"预测值：{self.classes[pre_lab[i].item()]} ------------ 真实值：{self.classes[test_data_y[i].item()]}"
                        print(outputs)

        test_acc = test_corrects / test_num  # 计算测试准确率
        print(f"测试的准确率为：{test_acc:.2%}")  # 输出百分比格式的准确率


if __name__ == '__main__':
    weights = 'Crop_ResNet18_model_98.750%.pth'  # 迁移学习，替换为实际的权重路径
    model_tester = ModelTest("ResNet18_model", weights)  # 实例化测试类
    test_dataloader = model_tester.test_data_process()  # 加载测试数据
    model_tester.test_model_process(test_dataloader)
