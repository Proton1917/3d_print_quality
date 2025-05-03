import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss，用于处理类别不平衡问题"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        初始化Focal Loss
        
        参数:
            alpha: 正样本的权重系数
            gamma: 聚焦参数，降低易分类样本的权重
            reduction: 损失计算方式，'mean', 'sum' 或 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """计算Focal Loss"""
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # 预测概率
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class MultiTaskLoss(nn.Module):
    """多任务损失函数，适用于多头模型"""
    
    def __init__(self, task_weights=None):
        """
        初始化多任务损失函数
        
        参数:
            task_weights: 各任务权重字典，默认各任务权重相等
        """
        super(MultiTaskLoss, self).__init__()
        
        # 默认任务权重
        if task_weights is None:
            self.task_weights = {
                'quality': 1.0,
                'params': 1.0,
                'defects': 1.0
            }
        else:
            self.task_weights = task_weights
            
        # 定义各任务损失函数
        self.quality_loss_fn = FocalLoss(gamma=2)
        self.params_loss_fn = nn.MSELoss()
        self.defects_loss_fn = FocalLoss(gamma=2)
        
    def forward(self, predictions, targets):
        """
        计算多任务损失
        
        参数:
            predictions: 模型预测结果字典，包含'quality', 'params', 'defects'
            targets: 目标值字典，包含'quality_class', 'params', 'defect_type'
            
        返回:
            总损失和各任务损失字典
        """
        # 计算各任务损失
        quality_loss = self.quality_loss_fn(predictions['quality'], targets['quality_class'])
        params_loss = self.params_loss_fn(predictions['params'], targets['params'])
        defects_loss = self.defects_loss_fn(predictions['defects'], targets['defect_type'])
        
        # 计算加权总损失
        total_loss = (
            self.task_weights['quality'] * quality_loss +
            self.task_weights['params'] * params_loss +
            self.task_weights['defects'] * defects_loss
        )
        
        # 返回总损失和各任务损失
        losses = {
            'total': total_loss,
            'quality': quality_loss.item(),
            'params': params_loss.item(),
            'defects': defects_loss.item()
        }
        
        return losses

class DynamicWeightAverageLoss(nn.Module):
    """动态权重平均损失函数，自动平衡多任务权重"""
    
    def __init__(self, num_tasks=3, temp=2.0):
        """
        初始化动态权重平均损失函数
        
        参数:
            num_tasks: 任务数量
            temp: 温度参数，控制权重分布的软硬程度
        """
        super(DynamicWeightAverageLoss, self).__init__()
        self.num_tasks = num_tasks
        self.temp = temp
        
        # 初始化任务损失
        self.task_losses = torch.zeros(num_tasks)
        
        # 定义各任务损失函数
        self.quality_loss_fn = FocalLoss(gamma=2)
        self.params_loss_fn = nn.MSELoss()
        self.defects_loss_fn = FocalLoss(gamma=2)
    
    def forward(self, predictions, targets):
        """
        计算动态权重平均损失
        
        参数:
            predictions: 模型预测结果字典，包含'quality', 'params', 'defects'
            targets: 目标值字典，包含'quality_class', 'params', 'defect_type'
            
        返回:
            总损失和各任务损失字典
        """
        # 计算各任务损失
        quality_loss = self.quality_loss_fn(predictions['quality'], targets['quality_class'])
        params_loss = self.params_loss_fn(predictions['params'], targets['params'])
        defects_loss = self.defects_loss_fn(predictions['defects'], targets['defect_type'])
        
        # 当前批次损失
        current_losses = torch.tensor([quality_loss.item(), params_loss.item(), defects_loss.item()])
        
        # 更新任务损失历史
        if self.task_losses.sum() == 0:
            # 首次运行时初始化
            self.task_losses = current_losses
        else:
            # 指数移动平均更新
            self.task_losses = 0.9 * self.task_losses + 0.1 * current_losses
            
        # 计算权重，损失越大权重越大
        weights = self.task_losses / self.task_losses.sum()
        weights = weights ** self.temp
        weights = weights / weights.sum()
        
        # 计算加权总损失
        total_loss = weights[0] * quality_loss + weights[1] * params_loss + weights[2] * defects_loss
        
        # 返回总损失和各任务损失
        losses = {
            'total': total_loss,
            'quality': quality_loss.item(),
            'params': params_loss.item(),
            'defects': defects_loss.item(),
            'weights': weights.detach().cpu().numpy()
        }
        
        return losses