import torch
import torch.nn as nn
import torchvision.models as models

class ResNetBackbone(nn.Module):
    """ResNet骨干网络，用于特征提取"""
    
    def __init__(self, model_name='resnet50', pretrained=True, freeze_layers=True):
        """
        初始化ResNet骨干网络
        
        参数:
            model_name: 使用的ResNet变体 ('resnet18', 'resnet34', 'resnet50', 'resnet101')
            pretrained: 是否使用预训练权重
            freeze_layers: 是否冻结主干网络参数
        """
        super(ResNetBackbone, self).__init__()
        
        # 初始化模型
        weights = 'DEFAULT' if pretrained else None
        if model_name == 'resnet18':
            self.backbone = models.resnet18(weights=weights)
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(weights=weights)
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(weights=weights)
        elif model_name == 'resnet101':
            self.backbone = models.resnet101(weights=weights)
        else:
            raise ValueError(f"不支持的模型名称: {model_name}")
        
        # 移除最后的全连接层
        self.features_dim = self.backbone.fc.in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # 冻结backbone参数
        if freeze_layers:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """前向传播"""
        x = self.backbone(x)
        x = torch.flatten(x, 1)  # 展平特征图
        return x

def get_resnet_backbone(model_name='resnet50', pretrained=True, freeze_layers=True):
    """获取ResNet骨干网络实例"""
    return ResNetBackbone(model_name, pretrained, freeze_layers)