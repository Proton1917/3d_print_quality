import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    """分类头，用于打印质量分类任务"""
    
    def __init__(self, in_features, num_classes=4, dropout_rate=0.5):
        """
        初始化分类头
        
        参数:
            in_features: 输入特征维度
            num_classes: 分类类别数量 (优/良/中/差)
            dropout_rate: Dropout比例
        """
        super(ClassificationHead, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        """前向传播"""
        return self.classifier(x)