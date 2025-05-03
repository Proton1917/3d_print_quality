# 使用机器学习评估光固化 3D 打印层质量

本项目开发基于多头神经网络的系统，分析光固化 3D 打印中的图像，实现自动质量评估。通过集成先进的计算机视觉技术与深度学习方法，该系统能够同时进行质量分类、缺陷检测和参数预测，为3D打印制造过程提供全面的质量控制方案。

## 项目概述

光固化3D打印技术在医疗器械、精密零件和定制化产品制造中得到广泛应用，但打印质量控制一直是行业面临的挑战。本项目旨在构建一个智能系统，通过分析每一层打印后的图像，实现：

- **质量评估**：自动将打印层质量分为优、良、中、差四个等级
- **缺陷识别**：检测并分类常见缺陷（气泡、表面不均、层分离、翘曲等）
- **参数预测**：反向推导生成该层所用的打印参数（层厚、曝光时间、强度、温度）

## 项目结构

```
3d_print_quality/
├── data/              # 数据文件夹
│   ├── raw/           # 原始图像
│   ├── processed/     # 预处理后图像
│   ├── labeled/       # 带标签数据集
│   └── metadata.csv   # 参数记录
├── models/            # 模型定义
│   ├── backbone/      # 骨干网络（特征提取）
│   │   ├── resnet.py  # ResNet系列模型
│   │   └── vit.py     # Vision Transformer模型
│   ├── attention/     # 注意力机制
│   │   └── attention_modules.py  # 各类注意力模块
│   ├── heads/         # 任务头
│   │   ├── classification_head.py  # 质量分类头
│   │   ├── detection_head.py       # 缺陷检测头
│   │   └── regression_head.py      # 参数回归头
│   └── model.py       # 完整模型定义
├── training/          # 训练相关代码
│   ├── augmentations.py  # 数据增强
│   ├── dataloader.py     # 数据加载
│   ├── losses.py         # 损失函数
│   └── train.py          # 训练循环
├── utils/             # 工具函数
│   └── visualization.py  # 可视化工具
├── inference/         # 模型推理代码
│   └── inference.py      # 推理函数
├── notebooks/         # Jupyter笔记本
│   └── data_exploration.ipynb  # 数据探索与模型演示
├── configs/           # 配置文件
│   └── default.yaml   # 默认配置
├── scripts/           # 脚本文件
│   ├── evaluate_model.py  # 模型评估脚本
│   ├── preprocess.py      # 数据预处理脚本
│   └── train_model.py     # 模型训练脚本
├── requirements.txt   # 项目依赖
└── README.md          # 项目说明
```

## 技术特点

### 1. 多头神经网络架构

本项目采用多任务学习框架，共享特征提取的骨干网络，同时通过专门的任务头进行不同目标的学习：
- 质量评估：4类分类问题（优/良/中/差）
- 缺陷检测：5类分类问题（无缺陷/气泡/表面不均/层分离/翘曲）
- 参数预测：回归问题（层厚/曝光时间/强度/温度）

### 2. 灵活的骨干网络选择

支持两种主流的骨干网络架构：
- **ResNet系列**：经典的卷积神经网络，计算效率高
- **Vision Transformer**：基于自注意力机制，捕获全局依赖关系

### 3. 注意力增强特征提取

集成多种注意力机制提升模型性能：
- **CBAM**：结合通道注意力和空间注意力
- **自注意力机制**：优化特征间的相互关系

### 4. 自适应任务平衡

采用动态权重平均损失函数，自动平衡各任务的学习过程，避免单一任务主导训练。

### 5. 高级数据增强

使用Albumentations库实现丰富的数据增强策略：
- 几何变换：旋转、翻转、缩放、裁剪
- 光学变换：亮度、对比度、色调调整
- 噪声添加：高斯噪声、模糊等

## 安装与环境配置

### 环境要求

- Python 3.7+
- PyTorch 1.12.0+
- CUDA支持（推荐用于训练）

### 安装步骤

1. 克隆项目
```bash
git clone <repository-url>
cd 3d_print_quality
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 准备数据
   - 将原始图像放入 `data/raw` 目录
   - 准备元数据CSV文件（格式参考 `data/metadata.csv`）

## 使用说明

### 1. 数据预处理

预处理原始图像，进行裁剪、去噪和标准化等操作：

```bash
python scripts/preprocess.py --config configs/default.yaml
```

### 2. 模型训练

使用预处理后的数据训练模型：

```bash
python scripts/train_model.py --config configs/default.yaml
```

训练过程中，模型检查点和TensorBoard日志将保存到配置文件指定的输出目录。

### 3. 模型评估

评估模型性能，生成混淆矩阵和性能指标：

```bash
python scripts/evaluate_model.py --model_path models/checkpoints/best_model.pth
```

### 4. 单张图像推理

对单张图像进行质量评估：

```bash
python inference/inference.py --model_path models/checkpoints/best_model.pth --image_path data/processed/img_001.jpg --visualize
```

### 5. 批量推理

批量处理多张图像并输出结果CSV：

```bash
python inference/inference.py --model_path models/checkpoints/best_model.pth --image_dir data/test_images --output_csv results.csv
```

### 6. 配置文件说明

`configs/default.yaml` 包含模型、训练和数据处理的所有配置参数，可根据需要调整：

- **数据配置**：图像路径、批量大小、分割比例等
- **模型配置**：骨干网络类型、注意力模块使用等
- **训练配置**：优化器、学习率、训练周期等

## 数据要求与格式

### 元数据CSV文件格式

元数据文件 `data/metadata.csv` 应包含以下字段：
- `image_id`：图像文件名（不含扩展名）
- `layer_thickness`：层厚（毫米）
- `exposure_time`：曝光时间（秒）
- `intensity`：激光/UV强度（百分比）
- `temperature`：打印温度（摄氏度）
- `quality_score`：质量评分（0-1范围）
- `defect_type`：缺陷类型（文本描述）

### 图像要求

- 格式：JPG或PNG
- 大小：建议统一尺寸（模型默认使用224×224）
- 命名规则：与元数据中的`image_id`字段对应

## 高级使用

### 训练自定义模型

1. 创建新的配置文件：
```bash
cp configs/default.yaml configs/custom.yaml
```

2. 修改配置参数（如使用ViT骨干网络）：
```yaml
model:
  backbone_type: "vit"
  backbone_name: "vit_base_patch16_224"
  use_attention: true
```

3. 使用新配置训练：
```bash
python scripts/train_model.py --config configs/custom.yaml
```

### 数据探索与可视化

项目提供了Jupyter Notebook用于数据探索和模型性能分析：

```bash
jupyter notebook notebooks/data_exploration.ipynb
```

## 项目目标与性能

我们的目标是建立一个可靠的质量评估系统，实现打印误差控制在2微米以内。当前系统在测试集上实现的性能指标：

- 质量分类准确率：> 90%
- 缺陷检测准确率：> 85%
- 参数预测平均相对误差：< 5%

## 未来工作

- 集成实时监控系统，实现在线质量控制
- 拓展支持更多种类的3D打印材料和工艺
- 开发基于检测结果的自动参数矫正系统
- 扩充数据集规模，进一步提高模型鲁棒性

## 贡献指南

欢迎对项目进行贡献！可以通过以下方式参与：
- 提交bug报告和功能需求
- 改进文档和代码注释
- 提供更多数据集和预训练模型
- 开发新的骨干网络或注意力机制

## 许可证

本项目采用 MIT 许可证 - 详细信息请查看 LICENSE 文件