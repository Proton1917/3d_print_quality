"""
可视化工具模块 - 用于模型解释和结果可视化
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import cv2

def denormalize_image(img: torch.Tensor) -> torch.Tensor:
    """逆归一化图像用于可视化
    
    Args:
        img: 归一化的图像张量[C, H, W]，ImageNet归一化
    
    Returns:
        逆归一化的图像张量，像素值在[0, 1]范围内
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    return img * std + mean

def visualize_attention_maps(
    images: torch.Tensor,
    attention_maps: torch.Tensor,
) -> torch.Tensor:
    """可视化注意力图
    
    Args:
        images: 批量图像张量[B, C, H, W]
        attention_maps: 批量注意力图张量[B, C, H, W]
    
    Returns:
        可视化结果，张量[B, 3, H, W]
    """
    batch_size = images.shape[0]
    results = []
    
    for i in range(batch_size):
        # 处理单个图像
        img = denormalize_image(images[i])
        img_np = img.permute(1, 2, 0).numpy()  # [H, W, C]
        
        # 处理单个注意力图
        # 使用第一个通道的平均值，可以根据需要修改
        att_map = attention_maps[i].mean(dim=0)  # [H, W]
        att_map = F.interpolate(
            att_map.unsqueeze(0).unsqueeze(0),
            size=img.shape[1:],
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()
        
        # 归一化注意力图到[0, 1]
        att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)
        
        # 创建热图
        heatmap = cv2.applyColorMap(
            (att_map * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        
        # 叠加热图
        alpha = 0.6
        overlayed = img_np * (1 - alpha) + heatmap * alpha
        overlayed = np.clip(overlayed, 0, 1)
        
        # 转换回PyTorch张量
        result = torch.from_numpy(overlayed).permute(2, 0, 1).float()  # [C, H, W]
        results.append(result)
    
    # 堆叠所有结果
    return torch.stack(results)

def visualize_gradcam(
    model: nn.Module,
    image: torch.Tensor,
    target_layer: nn.Module,
    class_idx: Optional[int] = None
) -> np.ndarray:
    """使用GradCAM生成可视化
    
    Args:
        model: 神经网络模型
        image: 输入图像张量[C, H, W]
        target_layer: 目标层，通常是最后一个卷积层
        class_idx: 类别索引，如果为None则使用预测类别
    
    Returns:
        GradCAM可视化结果，numpy数组
    """
    model.eval()
    
    # 注册梯度钩子
    gradients = []
    def save_gradient(grad):
        gradients.append(grad)
    
    # 注册激活钩子
    activations = []
    def get_activation(mod, input, output):
        activations.append(output)
    
    # 注册钩子
    handle_activation = target_layer.register_forward_hook(get_activation)
    
    # 确保图像是批量形式
    if len(image.shape) == 3:
        image = image.unsqueeze(0)  # [1, C, H, W]
    
    # 获取输出和激活
    image.requires_grad_()
    outputs = model(image)
    
    # 对于多头网络，获取质量头输出
    if isinstance(outputs, dict):
        outputs = outputs['quality']
    
    # 如果没有指定类别，则使用预测类别
    if class_idx is None:
        class_idx = outputs.argmax(dim=1).item()
    
    # 计算梯度
    model.zero_grad()
    one_hot = torch.zeros_like(outputs)
    one_hot[0, class_idx] = 1
    outputs.backward(gradient=one_hot, retain_graph=True)
    
    # 移除钩子
    handle_activation.remove()
    
    # 获取梯度和激活
    gradients = gradients[0]
    activations = activations[0]
    
    # 计算权重
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    
    # 生成CAM
    cam = (weights * activations).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    
    # 归一化
    cam = F.interpolate(
        cam,
        size=image.shape[2:],
        mode='bilinear',
        align_corners=False
    )
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    
    # 转换为Numpy数组
    cam = cam.detach().cpu().numpy()[0, 0]
    
    # 生成热图
    heatmap = cv2.applyColorMap(
        (cam * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    
    # 转换图像为numpy数组
    img_np = denormalize_image(image[0]).permute(1, 2, 0).detach().cpu().numpy()
    
    # 叠加热图
    alpha = 0.5
    result = img_np * (1 - alpha) + heatmap / 255.0 * alpha
    result = np.clip(result, 0, 1)
    
    return result

def plot_quality_distribution(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    class_names: Optional[List[str]] = None
) -> plt.Figure:
    """绘制质量分类分布
    
    Args:
        predictions: 预测类别索引
        targets: 真实类别索引
        class_names: 类别名称列表
    
    Returns:
        matplotlib图形对象
    """
    if class_names is None:
        class_names = ['优', '良', '中', '差']
    
    # 计算每个类别的样本数量
    num_classes = len(class_names)
    pred_counts = torch.zeros(num_classes)
    target_counts = torch.zeros(num_classes)
    
    for i in range(num_classes):
        pred_counts[i] = (predictions == i).sum().item()
        target_counts[i] = (targets == i).sum().item()
    
    # 转换为百分比
    pred_percent = 100 * pred_counts / pred_counts.sum()
    target_percent = 100 * target_counts / target_counts.sum()
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 设置柱形图
    x = np.arange(num_classes)
    width = 0.35
    
    rects1 = ax.bar(x - width/2, target_percent.numpy(), width, label='实际')
    rects2 = ax.bar(x + width/2, pred_percent.numpy(), width, label='预测')
    
    # 添加标签和标题
    ax.set_xlabel('质量类别')
    ax.set_ylabel('百分比 (%)')
    ax.set_title('质量类别分布')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    
    # 添加数值标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3点垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    
    return fig

def plot_parameter_correlation(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    param_names: Optional[List[str]] = None
) -> plt.Figure:
    """绘制参数回归相关性散点图
    
    Args:
        predictions: 预测参数值
        targets: 真实参数值
        param_names: 参数名称列表
    
    Returns:
        matplotlib图形对象
    """
    if param_names is None:
        param_names = ['层高', '曝光时间', '光强']
    
    num_params = predictions.shape[1]
    
    # 创建网格布局
    rows = int(np.ceil(num_params / 2))
    cols = min(2, num_params)
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i in range(num_params):
        # 提取单个参数的预测和真实值
        pred = predictions[:, i].numpy()
        target = targets[:, i].numpy()
        
        # 绘制散点图
        axes[i].scatter(target, pred, alpha=0.3)
        
        # 添加对角线
        min_val = min(target.min(), pred.min())
        max_val = max(target.max(), pred.max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # 计算R²
        correlation = np.corrcoef(target, pred)[0, 1]
        r2 = correlation ** 2
        
        # 添加标签和标题
        axes[i].set_xlabel('真实值')
        axes[i].set_ylabel('预测值')
        axes[i].set_title(f'{param_names[i]} (R² = {r2:.4f})')
        axes[i].grid(True, alpha=0.3)
    
    # 如果有多余的子图，隐藏它们
    for i in range(num_params, len(axes)):
        axes[i].axis('off')
    
    fig.tight_layout()
    
    return fig

def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Optional[List[str]] = None
) -> plt.Figure:
    """绘制混淆矩阵
    
    Args:
        confusion_matrix: 混淆矩阵，形状为[num_classes, num_classes]
        class_names: 类别名称列表
    
    Returns:
        matplotlib图形对象
    """
    if class_names is None:
        class_names = ['优', '良', '中', '差']
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 显示混淆矩阵
    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # 设置坐标轴
    ax.set(
        xticks=np.arange(confusion_matrix.shape[1]),
        yticks=np.arange(confusion_matrix.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='真实类别',
        xlabel='预测类别',
        title='混淆矩阵'
    )
    
    # 旋转x轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 添加文本注释
    thresh = confusion_matrix.max() / 2.0
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(
                j, i, f"{confusion_matrix[i, j]}",
                ha="center", va="center",
                color="white" if confusion_matrix[i, j] > thresh else "black"
            )
    
    fig.tight_layout()
    
    return fig

def create_visualization_report(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    indices: List[int],
    device: torch.device,
    target_layer: Optional[nn.Module] = None
) -> plt.Figure:
    """创建可视化报告
    
    Args:
        model: 神经网络模型
        dataset: 数据集
        indices: 要可视化的样本索引列表
        device: 计算设备
        target_layer: 用于GradCAM的目标层
    
    Returns:
        matplotlib图形对象
    """
    model.eval()
    num_samples = len(indices)
    
    # 创建网格布局
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        # 获取样本
        sample = dataset[idx]
        image = sample['image']
        quality_class = sample['quality_class'].item()
        
        # 获取预测
        with torch.no_grad():
            image_tensor = image.unsqueeze(0).to(device)
            outputs = model(image_tensor)
            if isinstance(outputs, dict):
                quality_output = outputs['quality']
                parameter_output = outputs['parameters']
            else:
                quality_output = outputs
                parameter_output = None
            
            # 获取预测类别
            predicted_class = quality_output.argmax(dim=1).item()
        
        # 显示原始图像
        img_np = denormalize_image(image).permute(1, 2, 0).numpy()
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f'真实类别: {quality_class}')
        axes[i, 0].axis('off')
        
        # 显示GradCAM可视化结果
        if target_layer is not None:
            gradcam_vis = visualize_gradcam(model, image, target_layer, predicted_class)
            axes[i, 1].imshow(gradcam_vis)
            axes[i, 1].set_title(f'预测类别: {predicted_class}')
            axes[i, 1].axis('off')
        else:
            axes[i, 1].axis('off')
        
        # 显示预测概率
        if quality_output is not None:
            probabilities = F.softmax(quality_output[0], dim=0).cpu().numpy()
            class_names = ['优', '良', '中', '差']
            axes[i, 2].barh(class_names, probabilities, color='skyblue')
            axes[i, 2].set_xlim([0, 1])
            axes[i, 2].set_title('预测概率')
            
            # 添加参数预测
            if parameter_output is not None:
                param_values = parameter_output[0].cpu().numpy()
                param_names = ['层高', '曝光时间', '光强']
                param_text = '\n'.join([f'{name}: {value:.4f}' for name, value in zip(param_names, param_values)])
                axes[i, 2].text(0.5, -0.2, param_text, transform=axes[i, 2].transAxes, ha='center')
    
    fig.tight_layout()
    
    return fig
