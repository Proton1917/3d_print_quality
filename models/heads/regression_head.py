import torch
import torch.nn as nn

class RegressionHead(nn.Module):
    """回归头，用于预测打印参数如层厚、曝光时间等"""
    
    def __init__(self, in_features, num_params=4, dropout_rate=0.5):
        """
        初始化回归头
        
        参数:
            in_features: 输入特征维度
            num_params: 需要预测的参数数量 (层厚、曝光时间、强度、温度等)
            dropout_rate: Dropout比例
        """
        super(RegressionHead, self).__init__()
        
        self.regressor = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_params)
        )
        
    def forward(self, x):
        """前向传播"""
        return self.regressor(x)