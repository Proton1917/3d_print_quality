import os
import time
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from models.model import PrintQualityModel
from training.dataloader import get_dataloaders
from training.losses import MultiTaskLoss, DynamicWeightAverageLoss
from utils.device import get_device, is_mps_device

class Trainer:
    """模型训练器"""
    
    def __init__(self, config):
        """
        初始化训练器
        
        参数:
            config: 配置字典
        """
        self.config = config
        self.device = self._get_device()
        
        # 创建模型输出目录
        os.makedirs(config['output_dir'], exist_ok=True)
        
        # 初始化tensorboard
        self.writer = SummaryWriter(os.path.join(config['output_dir'], 'tensorboard'))
        
        # 创建模型
        self.model = self._create_model()
        
        # 设置优化器
        self.optimizer = self._create_optimizer()
        
        # 设置学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 设置损失函数
        self.criterion = self._create_criterion()
        
        # 获取数据加载器
        self.dataloaders = self._get_dataloaders()
        
        # 初始化混合精度训练
        # MPS设备可能不完全支持混合精度训练，需要检查
        use_amp = config.get('use_amp', False)
        if is_mps_device(self.device):
            if use_amp:
                print("警告：MPS设备可能不完全支持混合精度训练，建议设置use_amp=false")
            # 为安全起见，在MPS上禁用混合精度
            use_amp = False
        
        self.scaler = GradScaler() if use_amp else None
        
        # 最佳模型指标
        self.best_metric = float('inf')
        
    def _get_device(self):
        """获取训练设备"""
        return get_device(use_gpu=self.config.get('use_gpu', True))
    
    def _create_model(self):
        """创建模型"""
        model_config = self.config['model']
        model = PrintQualityModel(
            backbone_type=model_config['backbone_type'],
            backbone_config={
                'model_name': model_config['backbone_name'],
                'pretrained': model_config['pretrained'],
                'freeze_layers': model_config['freeze_layers']
            },
            use_attention=model_config['use_attention'],
            num_classes=model_config['num_classes'],
            num_params=model_config['num_params'],
            num_defect_types=model_config['num_defect_types']
        )
        
        if self.config['checkpoint_path'] and os.path.exists(self.config['checkpoint_path']):
            print(f"加载检查点：{self.config['checkpoint_path']}")
            checkpoint = torch.load(self.config['checkpoint_path'], map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
        return model.to(self.device)
    
    def _create_optimizer(self):
        """创建优化器"""
        optim_config = self.config['optimizer']
        if optim_config['name'].lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=optim_config['learning_rate'],
                weight_decay=optim_config['weight_decay']
            )
        elif optim_config['name'].lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=optim_config['learning_rate'],
                weight_decay=optim_config['weight_decay']
            )
        elif optim_config['name'].lower() == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=optim_config['learning_rate'],
                momentum=optim_config.get('momentum', 0.9),
                weight_decay=optim_config['weight_decay']
            )
        else:
            raise ValueError(f"不支持的优化器：{optim_config['name']}")
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        sched_config = self.config['scheduler']
        if sched_config['name'].lower() == 'steplr':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config['step_size'],
                gamma=sched_config['gamma']
            )
        elif sched_config['name'].lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sched_config['t_max']
            )
        elif sched_config['name'].lower() == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=sched_config['factor'],
                patience=sched_config['patience'],
                verbose=sched_config.get('verbose', True)
            )
        else:
            raise ValueError(f"不支持的调度器：{sched_config['name']}")
    
    def _create_criterion(self):
        """创建损失函数"""
        loss_config = self.config['loss']
        if loss_config['name'].lower() == 'multitask':
            return MultiTaskLoss(task_weights=loss_config.get('task_weights'))
        elif loss_config['name'].lower() == 'dynamic':
            return DynamicWeightAverageLoss(
                num_tasks=3,
                temp=loss_config.get('temp', 2.0)
            )
        else:
            raise ValueError(f"不支持的损失函数：{loss_config['name']}")
    
    def _get_dataloaders(self):
        """获取数据加载器"""
        data_config = self.config['data']
        return get_dataloaders(
            metadata_path=data_config['metadata_path'],
            image_dir=data_config['image_dir'],
            batch_size=data_config['batch_size'],
            num_workers=data_config['num_workers'],
            val_split=data_config['val_split'],
            test_split=data_config['test_split'],
            target_size=tuple(data_config['target_size'])
        )
    
    def train_one_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        dataloader = self.dataloaders['train']
        running_loss = 0.0
        running_task_losses = {'quality': 0.0, 'params': 0.0, 'defects': 0.0}
        
        start_time = time.time()
        
        for i, batch in enumerate(dataloader):
            # 将数据移动到设备
            images = batch['image'].to(self.device)
            targets = {
                'quality_class': batch['quality_class'].to(self.device),
                'params': batch['params'].to(self.device),
                'defect_type': batch['defect_type'].to(self.device)
            }
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            if self.config.get('use_amp', False) and self.scaler is not None and not is_mps_device(self.device):
                # 混合精度训练
                with autocast():
                    # 前向传播
                    predictions = self.model(images)
                    
                    # 计算损失
                    losses = self.criterion(predictions, targets)
                    loss = losses['total']
                
                # 反向传播
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                if self.config['gradient_clip']:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['gradient_clip_value']
                    )
                
                # 更新参数
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 正常训练
                # 前向传播
                predictions = self.model(images)
                
                # 计算损失
                losses = self.criterion(predictions, targets)
                loss = losses['total']
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                if self.config['gradient_clip']:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['gradient_clip_value']
                    )
                
                # 更新参数
                self.optimizer.step()
            
            # 累积损失
            running_loss += loss.item()
            for task in running_task_losses:
                running_task_losses[task] += losses[task]
            
            # 显示进度
            if i % self.config['print_freq'] == 0:
                print(f"Epoch {epoch+1}/{self.config['num_epochs']} "
                      f"Batch {i}/{len(dataloader)} "
                      f"Loss: {loss.item():.4f}")
        
        # 计算平均损失
        epoch_loss = running_loss / len(dataloader)
        epoch_task_losses = {task: val / len(dataloader) for task, val in running_task_losses.items()}
        
        # 记录tensorboard
        self.writer.add_scalar('Loss/train', epoch_loss, epoch)
        for task, loss_val in epoch_task_losses.items():
            self.writer.add_scalar(f'TaskLoss/train/{task}', loss_val, epoch)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{self.config['num_epochs']} "
              f"Train Loss: {epoch_loss:.4f} "
              f"Time: {epoch_time:.2f}s")
        
        return epoch_loss
    
    def validate(self, epoch):
        """验证模型"""
        self.model.eval()
        dataloader = self.dataloaders['val']
        running_loss = 0.0
        running_task_losses = {'quality': 0.0, 'params': 0.0, 'defects': 0.0}
        
        # 分类指标
        correct_quality = 0
        total_quality = 0
        correct_defects = 0
        total_defects = 0
        
        # 回归指标
        params_mse = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                # 将数据移动到设备
                images = batch['image'].to(self.device)
                targets = {
                    'quality_class': batch['quality_class'].to(self.device),
                    'params': batch['params'].to(self.device),
                    'defect_type': batch['defect_type'].to(self.device)
                }
                
                # 前向传播
                predictions = self.model(images)
                
                # 计算损失
                losses = self.criterion(predictions, targets)
                loss = losses['total']
                
                # 累积损失
                running_loss += loss.item()
                for task in running_task_losses:
                    running_task_losses[task] += losses[task]
                
                # 计算指标
                # 质量分类准确率
                _, predicted = torch.max(predictions['quality'], 1)
                total_quality += targets['quality_class'].size(0)
                correct_quality += (predicted == targets['quality_class']).sum().item()
                
                # 缺陷分类准确率
                _, predicted = torch.max(predictions['defects'], 1)
                total_defects += targets['defect_type'].size(0)
                correct_defects += (predicted == targets['defect_type']).sum().item()
                
                # 参数回归MSE
                params_mse += torch.mean((predictions['params'] - targets['params'])**2).item()
        
        # 计算平均损失
        epoch_loss = running_loss / len(dataloader)
        epoch_task_losses = {task: val / len(dataloader) for task, val in running_task_losses.items()}
        
        # 计算指标
        quality_acc = 100 * correct_quality / total_quality
        defects_acc = 100 * correct_defects / total_defects
        params_mse /= len(dataloader)
        
        # 记录tensorboard
        self.writer.add_scalar('Loss/val', epoch_loss, epoch)
        for task, loss_val in epoch_task_losses.items():
            self.writer.add_scalar(f'TaskLoss/val/{task}', loss_val, epoch)
        self.writer.add_scalar('Accuracy/quality', quality_acc, epoch)
        self.writer.add_scalar('Accuracy/defects', defects_acc, epoch)
        self.writer.add_scalar('MSE/params', params_mse, epoch)
        
        print(f"Validation: "
              f"Loss: {epoch_loss:.4f} "
              f"Quality Acc: {quality_acc:.2f}% "
              f"Defects Acc: {defects_acc:.2f}% "
              f"Params MSE: {params_mse:.6f}")
        
        return epoch_loss, quality_acc, defects_acc, params_mse
    
    def train(self):
        """训练模型"""
        print(f"开始训练，共{self.config['num_epochs']}个epochs")
        
        for epoch in range(self.config['num_epochs']):
            # 训练一个epoch
            train_loss = self.train_one_epoch(epoch)
            
            # 验证
            val_loss, quality_acc, defects_acc, params_mse = self.validate(epoch)
            
            # 更新学习率
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # 记录当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('LearningRate', current_lr, epoch)
            
            # 保存最佳模型
            if val_loss < self.best_metric:
                self.best_metric = val_loss
                self.save_checkpoint(
                    epoch=epoch,
                    is_best=True,
                    metrics={
                        'val_loss': val_loss,
                        'quality_acc': quality_acc,
                        'defects_acc': defects_acc,
                        'params_mse': params_mse
                    }
                )
            
            # 定期保存检查点
            if (epoch + 1) % self.config['save_freq'] == 0:
                self.save_checkpoint(
                    epoch=epoch,
                    is_best=False,
                    metrics={
                        'val_loss': val_loss,
                        'quality_acc': quality_acc,
                        'defects_acc': defects_acc,
                        'params_mse': params_mse
                    }
                )
        
        print("训练完成！")
    
    def save_checkpoint(self, epoch, is_best=False, metrics=None):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics or {}
        }
        
        if is_best:
            checkpoint_path = os.path.join(self.config['output_dir'], 'best_model.pth')
        else:
            checkpoint_path = os.path.join(self.config['output_dir'], f'checkpoint_epoch_{epoch+1}.pth')
        
        torch.save(checkpoint, checkpoint_path)
        print(f"已保存检查点到：{checkpoint_path}")
        
        # 保存最佳模型的配置
        if is_best:
            config_path = os.path.join(self.config['output_dir'], 'best_config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f)

def main(argv=None):
    """主函数"""
    parser = argparse.ArgumentParser(description="训练3D打印质量评估模型")
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    args = parser.parse_args(argv)
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置随机种子
    if 'seed' in config:
        seed = config['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    # 创建并训练模型
    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
