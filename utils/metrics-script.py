"""
模型评估指标计算模块
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
)
from typing import Dict, List, Union, Optional

def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """计算分类指标
    
    Args:
        predictions: 预测张量 [batch_size, num_classes] 或 [batch_size] (如果已经是类别索引)
        targets: 目标张量 [batch_size]
        threshold: 二分类阈值 (对于多类别分类没有使用)
    
    Returns:
        包含各种指标的字典
    """
    # 确保在CPU上
    if torch.is_tensor(predictions):
        if predictions.dim() > 1 and predictions.size(1) > 1:
            # 多类别分类，获取类别索引
            predictions = torch.argmax(predictions, dim=1).cpu().numpy()
        else:
            # 二分类，应用阈值
            predictions = (predictions.squeeze() > threshold).float().cpu().numpy()
    
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    
    # 计算各种指标
    try:
        # 多类别指标
        accuracy = accuracy_score(targets, predictions)
        
        # 仅当有多个类别时才计算这些指标
        n_classes = len(np.unique(np.concatenate([targets, predictions])))
        if n_classes > 1:
            # 对于多类别，要指定average参数
            if n_classes == 2:
                # 二分类
                precision = precision_score(targets, predictions)
                recall = recall_score(targets, predictions)
                f1 = f1_score(targets, predictions)
            else:
                # 多类别
                precision = precision_score(targets, predictions, average='macro')
                recall = recall_score(targets, predictions, average='macro')
                f1 = f1_score(targets, predictions, average='macro')
            
            # 计算混淆矩阵
            cm = confusion_matrix(targets, predictions)
        else:
            # 只有一个类别时，所有样本都应该是这个类别
            precision = 1.0
            recall = 1.0
            f1 = 1.0
            cm = np.array([[len(targets)]])
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }
    except Exception as e:
        # 如果出错，返回空字典
        print(f"计算指标时出错: {e}")
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'confusion_matrix': np.zeros((1, 1))
        }

def compute_regression_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> Dict[str, float]:
    """计算回归指标
    
    Args:
        predictions: 预测张量 [batch_size, num_params]
        targets: 目标张量 [batch_size, num_params]
    
    Returns:
        包含各种指标的字典
    """
    # 确保在CPU上
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    
    # 计算各种指标
    mse = mean_squared_error(targets, predictions, multioutput='raw_values')
    mae = mean_absolute_error(targets, predictions, multioutput='raw_values')
    
    # 计算每个参数的R²
    r2 = []
    for i in range(predictions.shape[1]):
        r2.append(r2_score(targets[:, i], predictions[:, i]))
    
    return {
        'mse': mse.tolist(),
        'mae': mae.tolist(),
        'r2': r2
    }

def compute_oob_score(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> float:
    """计算袋外(OOB)得分
    
    袋外得分类似于随机森林中的评估方法，测试模型在未用于训练的数据上的表现
    
    Args:
        model: 训练好的模型
        data_loader: 包含未参与训练的数据的DataLoader
        device: 计算设备
    
    Returns:
        袋外得分 (0-1)，越高越好
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in data_loader:
            images = batch['image'].to(device)
            quality_targets = batch['quality_class'].to(device)
            
            # 前向传播
            outputs = model(images)
            if isinstance(outputs, dict):
                quality_outputs = outputs['quality']
            else:
                quality_outputs = outputs
            
            # 计算准确率
            _, predicted = torch.max(quality_outputs.data, 1)
            total += quality_targets.size(0)
            correct += (predicted == quality_targets).sum().item()
    
    return correct / total

def compute_print_parameter_importance(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    param_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """计算打印参数的相对重要性
    
    通过分析模型对参数预测的敏感度来估计参数重要性
    
    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 计算设备
        param_names: 参数名称列表
    
    Returns:
        参数重要性字典
    """
    if param_names is None:
        param_names = ['层高', '曝光时间', '光强']
    
    model.eval()
    param_gradients = []
    
    # 收集参数梯度
    for batch in data_loader:
        images = batch['image'].to(device)
        param_targets = batch['parameters'].to(device)
        
        # 计算梯度
        images.requires_grad_()
        outputs = model(images)
        if isinstance(outputs, dict):
            param_outputs = outputs['parameters']
        else:
            param_outputs = outputs
        
        for i in range(param_outputs.shape[1]):
            model.zero_grad()
            if param_outputs.shape[1] == 1:
                param_outputs.backward(retain_graph=(i < param_outputs.shape[1] - 1))
            else:
                param_loss = torch.mean(param_outputs[:, i])
                param_loss.backward(retain_graph=(i < param_outputs.shape[1] - 1))
            
            # 计算输入图像的梯度范数
            grad_norm = torch.norm(images.grad, dim=(1, 2, 3)).mean().item()
            param_gradients.append(grad_norm)
    
    # 归一化梯度以获得相对重要性
    if len(param_gradients) > 0:
        param_gradients = np.array(param_gradients)
        param_importance = param_gradients / np.sum(param_gradients)
        
        return {name: importance for name, importance in zip(param_names, param_importance)}
    else:
        return {name: 0.0 for name in param_names}

def estimate_optimal_parameters(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    param_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """估计最优打印参数
    
    通过搜索质量预测为"优"的样本来估计最优参数
    
    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 计算设备
        param_names: 参数名称列表
    
    Returns:
        最优参数字典
    """
    if param_names is None:
        param_names = ['层高', '曝光时间', '光强']
    
    model.eval()
    optimal_params = []
    optimal_confidence = []
    
    # 收集预测为"优"的样本的参数
    with torch.no_grad():
        for batch in data_loader:
            images = batch['image'].to(device)
            params = batch['parameters'].cpu().numpy()
            
            # 前向传播
            outputs = model(images)
            if isinstance(outputs, dict):
                quality_outputs = outputs['quality']
            else:
                quality_outputs = outputs
            
            # 获取预测类别和置信度
            probabilities = F.softmax(quality_outputs, dim=1)
            top_class = torch.argmax(quality_outputs, dim=1)
            top_prob = torch.gather(probabilities, 1, top_class.unsqueeze(1))
            
            # 找到预测为"优"的样本
            excellent_mask = (top_class == 0)  # 假设"优"对应索引0
            if excellent_mask.sum() > 0:
                excellent_params = params[excellent_mask.cpu().numpy()]
                excellent_conf = top_prob[excellent_mask].cpu().numpy()
                
                optimal_params.append(excellent_params)
                optimal_confidence.append(excellent_conf)
    
    # 计算加权平均值作为最优参数
    if len(optimal_params) > 0:
        all_params = np.vstack(optimal_params)
        all_conf = np.vstack(optimal_confidence)
        
        # 使用置信度作为权重计算加权平均值
        weighted_params = np.sum(all_params * all_conf, axis=0) / np.sum(all_conf)
        
        return {name: value for name, value in zip(param_names, weighted_params.flatten())}
    else:
        return {name: 0.0 for name in param_names}
