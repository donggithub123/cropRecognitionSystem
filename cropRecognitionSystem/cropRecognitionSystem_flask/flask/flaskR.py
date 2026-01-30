# -*- coding: utf-8 -*-
# @Time : 2024-09-2024/9/28 22:37
# @Author : 林枫
# @File : flaskR.py

import json
import os
from flask import Flask, request, jsonify
from train_test.prediction import Prediction


class ImagePredictorAPI:
    def __init__(self):
        """初始化 Flask 应用和模型"""
        self.app = Flask(__name__)

        # 定义路由
        self.app.add_url_rule('/predict', 'predict', self.predict, methods=['POST'])
        self.app.add_url_rule('/file_names', 'file_names', self.file_names, methods=['GET'])

    def predict(self):
        """处理预测请求"""
        # 检查请求中是否有必要的数据
        data = request.get_json()

        if not data or 'model_name' not in data or 'weights' not in data or 'image_path' not in data:
            return jsonify({'error': 'Model name, weight path, or image not provided'}), 400

        model_name = data['model_name']
        weight_path = data['weights']
        image_path = data['image_path']  # 假设 image 是以某种方式编码的图像数据

        try:
            # 将图像数据处理为张量
            prediction = Prediction(model_name, weight_path, image_path)
            result = prediction.run()
            return result
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def get_file_names(self, directory):
        """获取指定文件夹中的所有文件名"""
        try:
            # 列出目录中的所有文件和文件夹
            files = os.listdir(directory)

            # 过滤出文件（排除文件夹）
            file_names = [file for file in files if os.path.isfile(os.path.join(directory, file))]

            return file_names
        except Exception as e:
            print(f"发生错误: {e}")
            return []

    def file_names(self):
        """测试接口"""
        model_items = [name.split('.')[0] for name in self.get_file_names("../model")]
        weight_items = [name for name in self.get_file_names("../weights")]

        # 转换为所需格式
        formatted_model_items = [{'value': item, 'label': item} for item in model_items]
        formatted_weight_items = [{'value': item, 'label': item} for item in weight_items]

        # 创建字典
        result = {
            'model_items': formatted_model_items,
            'weight_items': formatted_weight_items
        }

        # 转换为 JSON 字符串
        json_result = json.dumps(result, ensure_ascii=False, indent=2)
        return json_result

    def run(self, port=5000):
        """运行 Flask 应用"""
        self.app.run(port=port)


if __name__ == '__main__':
    api = ImagePredictorAPI()
    api.run()
