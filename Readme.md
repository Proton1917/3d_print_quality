# 使用机器学习评估光固化 3D 打印层质量的完整项目计划 (改进版)

## 项目目标

我们将开发一个基于多头神经网络的系统，分析光固化 3D 打印中的图像，实现自动质量评估。项目分为三个阶段：

1. **数据采集与处理阶段**：收集高质量、多样化的打印图像
2. **多模型训练阶段**：实现三种不同的模型架构训练与比较
3. **模型部署与应用阶段**：将最佳模型集成到现有工作流

我们的目标是建立一个可靠的质量评估系统，实现打印误差控制在2微米以内。

---

## 1. 数据采集与处理
### 1.1 设备准备
- **相机设置**: 使用120fps高帧率相机，固定在打印机上捕捉打印过程
- **照明控制**: 安装可控LED照明，确保图像曝光一致
- **校准流程**: 建立相机校准流程，确保图像尺寸与实际尺寸对应

### 1.2 数据采集策略
- **采样多样性**: 捕捉不同参数配置下的打印情况（50种参数组合）
- **标注方法**: 基于参数偏离最优值的程度进行自动标注
- **文件组织**:
  ```
  data/
  ├── raw/            # 原始图像
  ├── processed/      # 预处理后图像
  ├── labeled/        # 带标签数据集
  └── metadata.csv    # 参数记录
  ```

### 1.3 数据增强
- **基本增强**: 旋转(±10°)、翻转、亮度变化(±10%)
- **特殊增强**: 模拟不同光照条件、添加噪声模拟干扰
- **合成数据**: 利用物理模型生成额外的合成样本

### 1.4 数据预处理流程
- **去噪处理**: 高斯滤波、中值滤波
- **归一化**: 图像标准化到[0,1]区间
- **区域提取**: 使用图像分割算法自动提取关注区域

---

## 2. 模型架构设计与训练

### 2.1 多头神经网络架构
参考Brion等人的研究，设计包含共享主干网络和多个输出头的网络架构:

- **共享主干**: ResNet50或Vision Transformer
- **注意力机制**: 添加多层注意力模块增强特征提取能力
- **输出头定义**:
  - 头1: 打印质量分类(优/良/中/差)
  - 头2: 打印参数回归(层厚、曝光时间等)
  - 头3: 缺陷类型识别(气泡、不均匀、分层等)

### 2.2 训练策略
- **分阶段训练**: 
  1. 首先使用小型数据集训练主干网络
  2. 冻结主干，训练各输出头
  3. 进行端到端微调
- **损失函数组合**:
  - 质量分类: 交叉熵损失
  - 参数回归: MSE损失
  - 缺陷识别: Focal Loss
- **优化器**: AdamW，学习率1e-4，权重衰减1e-5
- **批量大小**: 32，累积梯度进行4次更新

### 2.3 模型评估指标
- **质量分类**: 准确率、F1分数、混淆矩阵
- **参数回归**: MAE、RMSE、R²、袋外误差(OOB)
- **缺陷识别**: 精确率、召回率、IoU
- **可视化评估**:
  - 使用GradCAM可视化网络注意区域
  - t-SNE可视化特征分布

### 2.4 消融实验设计
- **架构变体**: 比较单头vs多头设计效果
- **特征变体**: 评估注意力机制的贡献
- **参数敏感性**: 分析不同超参数对性能的影响

---

## 3. 项目实施步骤

### 3.1 环境准备
```bash
# 创建虚拟环境
conda create -n 3dprint python=3.9
conda activate 3dprint

# 安装依赖
pip install torch torchvision torchaudio
pip install pandas scikit-learn pillow numpy matplotlib
pip install opencv-python albumentations timm
```

### 3.2 项目结构
```
3d_print_quality/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── labeled/
│   └── metadata.csv
├── models/
│   ├── backbone/
│   │   ├── resnet.py
│   │   └── vit.py
│   ├── attention/
│   │   └── attention_modules.py
│   ├── heads/
│   │   ├── classification_head.py
│   │   ├── regression_head.py
│   │   └── detection_head.py
│   └── model.py
├── training/
│   ├── dataloader.py
│   ├── augmentations.py
│   ├── losses.py
│   ├── train.py
│   └── evaluate.py
├── utils/
│   ├── visualization.py
│   ├── metrics.py
│   └── logging.py
├── inference/
│   ├── predict.py
│   └── deploy.py
├── notebooks/
│   ├── EDA.ipynb
│   ├── ModelComparison.ipynb
│   └── ResultsAnalysis.ipynb
├── configs/
│   ├── default.yaml
│   └── experiment_1.yaml
├── scripts/
│   ├── preprocess.py
│   ├── train_model.py
│   └── evaluate_model.py
├── requirements.txt
└── README.md
```
### 3.3 数据处理实现
以下是`preprocess.py`脚本示例:
[预处理脚本](preprocess-script.py)
### 3.4 多头神经网络模型实现
[模型实现](model-script.py)
### 3.5 数据加载器实现
[数据加载器](dataloader-script.py)
### 3.6 训练脚本实现
[训练脚本](train-script.py)
### 3.7 配置文件示例
[配置文件](config.yaml)
### 3.8 评估和可视化工具
[可视化工具](visualization-script.py)
### 3.9 模型评估工具
[评估指标](metrics-script.py)
### 3.10 损失函数实现
[损失函数](losses-script.py)