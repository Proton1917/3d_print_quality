"""
损失函数模块 - 为多头神经网络模型实现自定义损失函数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class FocalLoss(nn.Module):
    """Focal Loss 用于处理类别不平衡
    
    Args:
        alpha: 类别权重，可以是标量或张量
        gamma: 调制因子，用于减少易分样本的损失
        reduction: 损失聚合方式
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算Focal Loss
        
        Args:
            inputs: 模型预测，形状为[B, C]
            targets: 目标类别，形状为[B]
        
        Returns:
            损失值
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss

class SmoothL1LossWithUncertainty(nn.Module):
    """带不确定性的平滑L1损失
    
    针对参数回归任务，结合不确定性学习
    
    Args:
        beta: L1损失平滑参数
        reduction: 损失聚合方式
    """
    
    def __init__(self, beta=1.0, reduction='mean'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction
        self.smooth_l1 = nn.SmoothL1Loss(beta=beta, reduction='none')
    
    def forward(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor,
        log_vars: torch.Tensor = None
    ) -> torch.Tensor:
        """计算带不确定性的平滑L1损失
        
        Args:
            inputs: 模型预测，形状为[B, C]
            targets: 目标值，形状为[B, C]
            log_vars: 不确定性的对数方差，形状为[C]
        
        Returns:
            损失值
        """
        loss = self.smooth_l1(inputs, targets)
        
        if log_vars is not None:
            # 应用不确定性加权
            precision = torch.exp(-log_vars)
            loss = precision * loss + log_vars
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class MultiHeadLoss(nn.Module):
    """多头网络的组合损失函数
    
    综合质量分类、参数回归和缺陷检测的损失
    
    Args:
        quality_weight: 质量分类损失权重
        parameter_weight: 参数回归损失权重
        defect_weight: 缺陷检测损失权重
        use_focal_loss: 是否使用Focal Loss进行分类
        focal_gamma: Focal Loss中的gamma参数
    """
    
    def __init__(
        self, 
        quality_weight=1.0, 
        parameter_weight=0.5, 
        defect_weight=0.3,
        use_focal_loss=True,
        focal_gamma=2.0
    ):
        super().__init__()
        self.quality_weight = quality_weight
        self.parameter_weight = parameter_weight
        self.defect_weight = defect_weight
        
        # 质量分类损失
        if use_focal_loss:
            self.quality_criterion = FocalLoss(gamma=focal_gamma)
        else:
            self.quality_criterion = nn.CrossEntropyLoss()
        
        # 参数回归损失
        self.parameter_criterion = nn.SmoothL1Loss()
        
        # 缺陷检测损失
        self.defect_criterion = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        quality_outputs: torch.Tensor,
        quality_targets: torch.Tensor,
        parameter_outputs: torch.Tensor,
        parameter_targets: torch.Tensor,
        defect_outputs: torch.Tensor = None,
        defect_targets: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """计算多头损失
        
        Args:
            quality_outputs: 质量分类输出，形状为[B, num_classes]
            quality_targets: 质量分类目标，形状为[B]
            parameter_outputs: 参数回归输出，形状为[B, num_params]
            parameter_targets: 参数回归目标，形状为[B, num_params]
            defect_outputs: 缺陷检测输出，形状为[B, num_defects]
            defect_targets: 缺陷检测目标，形状为[B, num_defects]
        
        Returns:
            包含各损失组件和总损失的字典
        """
        # 计算质量分类损失
        quality_loss = self.quality_criterion(quality_outputs, quality_targets)
        
        # 计算参数回归损失
        parameter_loss = self.parameter_criterion(parameter_outputs, parameter_targets)
        
        # 计算缺陷检测损失（如果提供）
        if defect_outputs is not None and defect_targets is not None:
            defect_loss = self.defect_criterion(defect_outputs, defect_targets)
        else:
            defect_loss = torch.tensor(0.0, device=quality_loss.device)
        
        # 计算加权总损失
        total_loss = (
            self.quality_weight * quality_loss +
            self.parameter_weight * parameter_loss +
            self.defect_weight * defect_loss
        )
        
        return {
            'quality_loss': quality_loss,
            'parameter_loss': parameter_loss,
            'defect_loss': defect_loss,
            'total_loss': total_loss
        }

class AdaptiveMultiHeadLoss(nn.Module):
    """自适应多头损失函数
    
    使用可学习的权重平衡不同任务的损失
    
    Args:
        initial_quality_weight: 质量分类损失初始权重
        initial_parameter_weight: 参数回归损失初始权重
        initial_defect_weight: 缺陷检测损失初始权重
    """
    
    def __init__(
        self, 
        initial_quality_weight=1.0, 
        initial_parameter_weight=0.5, 
        initial_defect_weight=0.3
    ):
        super().__init__()
        # 初始化可学习的对数权重
        self.log_quality_weight = nn.Parameter(torch.tensor(
            initial_quality_weight).log())
        self.log_parameter_weight = nn.Parameter(torch.tensor(
            initial_parameter_weight).log())
        self.log_defect_weight = nn.Parameter(torch.tensor(
            initial_defect_weight).log())
        
        # 损失函数
        self.quality_criterion = nn.CrossEntropyLoss()
        self.parameter_criterion = nn.SmoothL1Loss()
        self.defect_criterion = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        quality_outputs: torch.Tensor,
        quality_targets: torch.Tensor,
        parameter_outputs: torch.Tensor,
        parameter_targets: torch.Tensor,
        defect_outputs: torch.Tensor = None,
        defect_targets: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """计算自适应多头损失
        
        Args:
            quality_outputs: 质量分类输出
            quality_targets: 质量分类目标
            parameter_outputs: 参数回归输出
            parameter_targets: 参数回归目标
            defect_outputs: 缺陷检测输出
            defect_targets: 缺陷检测目标
        
        Returns:
            包含各损失组件和总损失的字典
        """
        # 获取任务权重
        quality_weight = torch.exp(self.log_quality_weight)
        parameter_weight = torch.exp(self.log_parameter_weight)
        defect_weight = torch.exp(self.log_defect_weight)
        
        # 计算各任务损失
        quality_loss = self.quality_criterion(quality_outputs, quality_targets)
        parameter_loss = self.parameter_criterion(parameter_outputs, parameter_targets)
        
        if defect_outputs is not None and defect_targets is not None:
            defect_loss = self.defect_criterion(defect_outputs, defect_targets)
        else:
            defect_loss = torch.tensor(0.0, device=quality_loss.device)
        
        # 计算加权总损失
        total_loss = (
            quality_weight * quality_loss +
            parameter_weight * parameter_loss +
            defect_weight * defect_loss
        )
        
        # 添加正则化项防止权重失控
        total_loss += 0.01 * (
            self.log_quality_weight**2 +
            self.log_parameter_weight**2 +
            self.log_defect_weight**2
        )
        
        return {
            'quality_loss': quality_loss,
            'parameter_loss': parameter_loss,
            'defect_loss': defect_loss,
            'total_loss': total_loss,
            'quality_weight': quality_weight.item(),
            'parameter_weight': parameter_weight.item(),
            'defect_weight': defect_weight.item()
        }