import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectionHead(nn.Module):
    """检测头，用于缺陷类型识别"""
    
    def __init__(self, in_features, num_defect_types=5, dropout_rate=0.5):
        """
        初始化检测头
        
        参数:
            in_features: 输入特征维度
            num_defect_types: 缺陷类型数量 (气泡、不均匀、分层等)
            dropout_rate: Dropout比例
        """
        super(DetectionHead, self).__init__()
        
        self.detector = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_defect_types)
        )
        
    def forward(self, x):
        """前向传播"""
        return self.detector(x)