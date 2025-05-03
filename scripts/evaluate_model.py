import os
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from models.model import PrintQualityModel
from training.dataloader import PrintQualityDataset
from utils.visualization import plot_confusion_matrix

def evaluate_model(model, dataloader, device):
    """
    评估模型性能
    
    参数:
        model: 模型实例
        dataloader: 数据加载器
        device: 计算设备
        
    返回:
        包含各项评估指标的字典
    """
    model.eval()
    
    # 初始化指标收集器
    quality_preds = []
    quality_targets = []
    defect_preds = []
    defect_targets = []
    param_preds = []
    param_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            # 将数据移动到设备
            images = batch['image'].to(device)
            quality_class = batch['quality_class'].to(device)
            params = batch['params'].to(device)
            defect_type = batch['defect_type'].to(device)
            
            # 前向传播
            predictions = model(images)
            
            # 收集预测结果和目标
            _, quality_pred = torch.max(predictions['quality'], 1)
            quality_preds.append(quality_pred.cpu().numpy())
            quality_targets.append(quality_class.cpu().numpy())
            
            _, defect_pred = torch.max(predictions['defects'], 1)
            defect_preds.append(defect_pred.cpu().numpy())
            defect_targets.append(defect_type.cpu().numpy())
            
            param_preds.append(predictions['params'].cpu().numpy())
            param_targets.append(params.cpu().numpy())
    
    # 合并批次结果
    quality_preds = np.concatenate(quality_preds)
    quality_targets = np.concatenate(quality_targets)
    defect_preds = np.concatenate(defect_preds)
    defect_targets = np.concatenate(defect_targets)
    param_preds = np.concatenate(param_preds)
    param_targets = np.concatenate(param_targets)
    
    # 计算质量分类指标
    quality_cm = confusion_matrix(quality_targets, quality_preds)
    quality_acc = np.mean(quality_preds == quality_targets)
    quality_report = classification_report(quality_targets, quality_preds, output_dict=True)
    
    # 计算缺陷分类指标
    defect_cm = confusion_matrix(defect_targets, defect_preds)
    defect_acc = np.mean(defect_preds == defect_targets)
    defect_report = classification_report(defect_targets, defect_preds, output_dict=True)
    
    # 计算参数回归指标
    param_mse = np.mean((param_preds - param_targets) ** 2, axis=0)
    param_mae = np.mean(np.abs(param_preds - param_targets), axis=0)
    
    # 组合评估结果
    metrics = {
        'quality': {
            'accuracy': quality_acc,
            'confusion_matrix': quality_cm,
            'report': quality_report
        },
        'defect': {
            'accuracy': defect_acc,
            'confusion_matrix': defect_cm,
            'report': defect_report
        },
        'params': {
            'mse': param_mse,
            'mae': param_mae
        },
        'predictions': {
            'quality': quality_preds,
            'defect': defect_preds,
            'params': param_preds
        },
        'targets': {
            'quality': quality_targets,
            'defect': defect_targets,
            'params': param_targets
        }
    }
    
    return metrics

def visualize_results(metrics, output_dir, quality_labels, defect_labels):
    """
    可视化评估结果
    
    参数:
        metrics: 评估指标字典
        output_dir: 输出目录
        quality_labels: 质量标签列表
        defect_labels: 缺陷标签列表
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制质量分类混淆矩阵
    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(
        metrics['quality']['confusion_matrix'],
        classes=quality_labels,
        title='质量分类混淆矩阵'
    )
    plt.savefig(os.path.join(output_dir, 'quality_confusion_matrix.png'))
    plt.close()
    
    # 绘制缺陷分类混淆矩阵
    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(
        metrics['defect']['confusion_matrix'],
        classes=defect_labels,
        title='缺陷分类混淆矩阵'
    )
    plt.savefig(os.path.join(output_dir, 'defect_confusion_matrix.png'))
    plt.close()
    
    # 绘制参数预测散点图
    param_names = ['层厚', '曝光时间', '强度', '温度']
    
    for i, param_name in enumerate(param_names):
        plt.figure(figsize=(8, 8))
        plt.scatter(
            metrics['targets']['params'][:, i],
            metrics['predictions']['params'][:, i],
            alpha=0.5
        )
        
        # 添加理想预测线(y=x)
        min_val = min(
            metrics['targets']['params'][:, i].min(),
            metrics['predictions']['params'][:, i].min()
        )
        max_val = max(
            metrics['targets']['params'][:, i].max(),
            metrics['predictions']['params'][:, i].max()
        )
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel(f'真实{param_name}')
        plt.ylabel(f'预测{param_name}')
        plt.title(f'参数预测: {param_name} (MSE: {metrics["params"]["mse"][i]:.6f})')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'param_prediction_{param_name}.png'))
        plt.close()
    
    # 保存评估指标到CSV
    quality_df = pd.DataFrame(metrics['quality']['report']).transpose()
    quality_df.to_csv(os.path.join(output_dir, 'quality_metrics.csv'))
    
    defect_df = pd.DataFrame(metrics['defect']['report']).transpose()
    defect_df.to_csv(os.path.join(output_dir, 'defect_metrics.csv'))
    
    params_df = pd.DataFrame({
        'Parameter': param_names,
        'MSE': metrics['params']['mse'],
        'MAE': metrics['params']['mae']
    })
    params_df.to_csv(os.path.join(output_dir, 'params_metrics.csv'), index=False)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="评估3D打印质量模型")
    parser.add_argument('--model_path', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径，若不提供则使用模型目录下的best_config.yaml')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录路径，若不提供则使用模型目录下的evaluation')
    args = parser.parse_args()
    
    # 确定配置文件路径
    if args.config is None:
        model_dir = os.path.dirname(args.model_path)
        config_path = os.path.join(model_dir, 'best_config.yaml')
        if not os.path.exists(config_path):
            config_path = os.path.join(os.path.dirname(model_dir), '..', 'configs', 'default.yaml')
    else:
        config_path = args.config
        
    # 确定输出目录
    if args.output_dir is None:
        output_dir = os.path.join(os.path.dirname(args.model_path), 'evaluation')
    else:
        output_dir = args.output_dir
        
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device('cuda:0' if torch.cuda.is_available() and config['use_gpu'] else 'cpu')
    print(f"使用设备：{device}")
    
    # 获取标签映射
    quality_labels = ['优', '良', '中', '差']
    defect_labels = ['无缺陷', '气泡', '表面不均', '层分离', '翘曲']
    
    # 创建测试数据集和数据加载器
    test_dataset = PrintQualityDataset(
        metadata_path=os.path.join(os.path.dirname(config['data']['metadata_path']), 'test_metadata.csv'),
        image_dir=config['data']['image_dir'],
        split='test',
        target_size=tuple(config['data']['target_size'])
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    # 创建并加载模型
    model_config = config['model']
    model = PrintQualityModel(
        backbone_type=model_config['backbone_type'],
        backbone_config={
            'model_name': model_config['backbone_name'],
            'pretrained': False,  # 评估时不需要预训练
            'freeze_layers': False  # 评估时不需要冻结层
        },
        use_attention=model_config['use_attention'],
        num_classes=model_config['num_classes'],
        num_params=model_config['num_params'],
        num_defect_types=model_config['num_defect_types']
    ).to(device)
    
    # 加载检查点
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"成功加载模型检查点：{args.model_path}")
    
    # 评估模型
    print("开始评估模型...")
    metrics = evaluate_model(model, test_loader, device)
    
    # 打印主要指标
    print("\n=== 评估结果 ===")
    print(f"质量分类准确率: {metrics['quality']['accuracy']:.4f}")
    print(f"缺陷分类准确率: {metrics['defect']['accuracy']:.4f}")
    print(f"参数预测MSE: {metrics['params']['mse'].mean():.6f}")
    print(f"参数预测MAE: {metrics['params']['mae'].mean():.6f}")
    
    # 可视化结果
    print("\n生成可视化结果...")
    visualize_results(metrics, output_dir, quality_labels, defect_labels)
    
    print(f"\n评估完成！结果已保存到：{output_dir}")

if __name__ == '__main__':
    main()