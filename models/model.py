import torch
import torch.nn as nn

from models.backbone.resnet import get_resnet_backbone
from models.backbone.vit import get_vit_backbone
from models.attention.attention_modules import CBAM
from models.heads.classification_head import ClassificationHead
from models.heads.regression_head import RegressionHead
from models.heads.detection_head import DetectionHead

class PrintQualityModel(nn.Module):
    """3D打印质量评估多头模型"""
    
    def __init__(self, backbone_type='resnet', backbone_config=None, use_attention=True, 
                 num_classes=4, num_params=4, num_defect_types=5):
        """
        初始化3D打印质量评估多头模型
        
        参数:
            backbone_type: 骨干网络类型 ('resnet' 或 'vit')
            backbone_config: 骨干网络配置参数
            use_attention: 是否使用注意力模块
            num_classes: 质量分类类别数量
            num_params: 需要回归的参数数量
            num_defect_types: 缺陷类型数量
        """
        super(PrintQualityModel, self).__init__()
        
        # 默认配置
        if backbone_config is None:
            if backbone_type == 'resnet':
                backbone_config = {'model_name': 'resnet50', 'pretrained': True, 'freeze_layers': True}
            else:  # vit
                backbone_config = {'model_name': 'vit_base_patch16_224', 'pretrained': True, 'freeze_layers': True}
        
        # 初始化骨干网络
        if backbone_type == 'resnet':
            # 期望的配置键：model_name, pretrained, freeze_layers
            model_name = backbone_config.get('model_name', 'resnet50')
            pretrained = backbone_config.get('pretrained', True)
            freeze_layers = backbone_config.get('freeze_layers', True)

            self.backbone = get_resnet_backbone(
                model_name=model_name,
                pretrained=pretrained,
                freeze_layers=freeze_layers,
            )
            self.features_dim = self.backbone.features_dim

            # 对于ResNet，注意力模块插入到特征提取后
            if use_attention:
                # 假设ResNet最后输出是2048维特征，需要将其重塑为特征图形式以便CBAM处理
                self.reshape_features = True
                # 根据不同ResNet变体自动推断通道数
                in_channels = self.features_dim
                self.attention = CBAM(in_channels=in_channels)
        else:  # vit
            # 允许通过backbone_name进行简单映射（默认vit_base_patch16_224）
            name = backbone_config.get('model_name', 'vit_base_patch16_224')
            pretrained = backbone_config.get('pretrained', True)
            freeze_layers = backbone_config.get('freeze_layers', True)

            vit_cfg = {
                'vit_base_patch16_224': dict(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12),
                'vit_small_patch16_224': dict(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6),
                'vit_tiny_patch16_224':  dict(img_size=224, patch_size=16, embed_dim=192, depth=12, num_heads=3),
            }.get(name, dict(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12))

            self.backbone = get_vit_backbone(
                img_size=vit_cfg['img_size'],
                patch_size=vit_cfg['patch_size'],
                embed_dim=vit_cfg['embed_dim'],
                num_heads=vit_cfg['num_heads'],
                depth=vit_cfg['depth'],
                pretrained=pretrained,
                freeze_layers=freeze_layers,
            )
            self.features_dim = self.backbone.features_dim
            self.reshape_features = False

            # 对于ViT，我们使用不同形式的注意力机制
            if use_attention:
                self.attention = nn.MultiheadAttention(
                    embed_dim=self.features_dim,
                    num_heads=8,
                    dropout=0.1
                )
                
        self.use_attention = use_attention
        
        # 初始化各个输出头
        self.classification_head = ClassificationHead(self.features_dim, num_classes)
        self.regression_head = RegressionHead(self.features_dim, num_params)
        self.detection_head = DetectionHead(self.features_dim, num_defect_types)
    
    def forward(self, x):
        """前向传播"""
        # 骨干网络特征提取
        features = self.backbone(x)
        
        # 应用注意力机制
        if self.use_attention:
            if self.reshape_features:
                # 将特征重塑为特征图形式，假设批大小为B
                B = features.shape[0]
                features_map = features.view(B, self.features_dim, 1, 1)
                features_map = self.attention(features_map)
                features = features_map.view(B, -1)
            else:
                # ViT风格的自注意力
                features = features.unsqueeze(0)  # 添加序列维度
                features, _ = self.attention(features, features, features)
                features = features.squeeze(0)  # 移除序列维度
        
        # 通过各个输出头
        quality_pred = self.classification_head(features)
        params_pred = self.regression_head(features)
        defect_pred = self.detection_head(features)
        
        return {
            'quality': quality_pred,
            'params': params_pred,
            'defects': defect_pred
        }
