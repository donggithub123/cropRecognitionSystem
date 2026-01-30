农作物识别系统围绕智慧农业需求构建，采用深度学习与前后端的技术架构。系统基于 PyTorch框架完成 6 类卷积神经网络的搭建与训练，涵盖 LeNet、AlexNet、VGG16Net、ResNet18、GoogLeNet 及 MobileNetV2。
后端基于 Spring Boot 与 OpenJDK 11 开发，实现模型本地推理，彻底消除跨服务网络依赖；同时依托 JPA 完成 MySQL 数据持久化。
前端采用 Vue 结合 Element UI 构建，实现登录页自定义背景、图像上传、固定 6 类模型的下拉选择框、识别结果可视化等交互功能。
