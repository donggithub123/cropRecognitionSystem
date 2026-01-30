# -*- coding: utf-8 -*-
# @Time : 2024-10-11 15:13
# @Author : 林枫
# @File : prediction.py

from torchvision import transforms
import torch
from PIL import Image
from model.GoogLeNet_model import GoogLeNet, Inception
from model.MobileNetV2_model import MobileNetV2
from model.ResNet18_model import ResNet18, Residual
from model.VGG16Net_model import VGG16Net
from model.LeNet_model import LeNet
from model.AlexNet_model import AlexNet
import json
import time
from io import BytesIO
import requests

MODEL_CLASSES = {
    "GoogLeNet": {"model": GoogLeNet, "other": Inception},
    "ResNet18": {"model": ResNet18, "other": Residual},
    "MobileNetV2": {"model": MobileNetV2},
    "VGG16Net": {"model": VGG16Net},
    "LeNet": {"model": LeNet},
    "AlexNet": {"model": AlexNet},
}


class Prediction:
    def __init__(self, model_name, weights, image_path):
        """初始化，加载模型、权重和图片路径"""
        try:
            weights = "../weights/" + weights if weights else None
            self.image_path = image_path

            # 检查参数有效性
            if not model_name or not weights or not image_path:
                raise ValueError("参数缺失: 模型名、权重或图片路径不能为空")

            # 提取模型名称并获取对应模型类
            model_name = model_name.split('_')[0]
            model_info = MODEL_CLASSES.get(model_name)
            if not model_info:
                raise ValueError(f"模型名 '{model_name}' 不在可用模型列表中")

            # 加载模型
            self.model_class = model_info["model"]
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if "other" in model_info:
                self.model = self.model_class(model_info["other"]).to(self.device)  # 传递其他类
            else:
                self.model = self.model_class().to(self.device)  # 无其他类时直接实例化模型

            self.model.load_state_dict(torch.load(weights, map_location=self.device, weights_only=False))  # 加载权重
            self.model.to(self.device)
            self.classes = ['小麦', '水稻', '玉米', '甘蔗', '黄麻']

        except Exception as e:
            raise ValueError(f"初始化时发生错误: {e}")

    def preprocess_image(self):
        """处理图片，准备用于模型预测"""
        try:
            # 从URL获取图像
            response = requests.get(self.image_path)
            response.raise_for_status()  # 检查请求是否成功
            image_data = Image.open(BytesIO(response.content)).convert('RGB')  # 转为RGB图像

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.16018272193439712, 0.1755668202902573, 0.10740333164906743],
                                     [0.04689176552845943, 0.05427002046013127, 0.03591450879001498]),  # 归一化处理
                transforms.Resize((224, 224))  # 调整图像大小
            ])

            data = transform(image_data)
            data = data.unsqueeze(0)  # 增加批次维度
            return data
        except Exception as e:
            raise ValueError(f"图像处理时发生错误: {e}")

    def predict(self, image_tensor):
        """使用模型进行预测"""
        try:
            self.model.eval()  # 设置为评估模式
            with torch.no_grad():
                image_tensor = image_tensor.to(self.device)
                output = self.model(image_tensor)
                prediction = torch.argmax(output, dim=1).item()  # 获取预测类别索引
                confidence = torch.max(torch.softmax(output, dim=1)).item()  # 获取置信度
            return self.classes[prediction], confidence
        except Exception as e:
            raise ValueError(f"预测时发生错误: {e}")

    def run(self):
        """运行预测并返回JSON格式结果"""
        try:
            start_time = time.time()  # 记录开始时间
            image_tensor = self.preprocess_image()  # 预处理图像
            prediction, confidence = self.predict(image_tensor)  # 获取预测结果和置信度
            end_time = time.time()  # 记录结束时间
            total_time = end_time - start_time  # 计算总耗时

            # 返回成功的JSON格式
            result = {
                "status": 200,
                "message": "预测成功",
                "prediction": prediction,
                "confidence": f"{confidence * 100:.2f}%",
                "total_time": f"{total_time:.4f}秒"
            }
            return json.dumps(result, ensure_ascii=False)

        except Exception as e:
            # 返回错误信息的JSON格式
            error_response = {
                "status": 400,
                "message": f"出错: {str(e)}"
            }
            return json.dumps(error_response, ensure_ascii=False)


# 示例：从其他地方调用
if __name__ == '__main__':
    app = Prediction('GoogLeNet_model', 'Fruit_GoogLeNet_model_81.218%.pth',
                          '/data/app/菠萝/38.jpg')
    result = app.run()
    print(result)
