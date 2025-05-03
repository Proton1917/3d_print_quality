import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class PrintQualityDataset(Dataset):
    """3D打印质量评估数据集类"""
    
    def __init__(self, 
                 metadata_path,
                 image_dir, 
                 transform=None, 
                 target_size=(224, 224), 
                 split='train'):
        """
        初始化数据集
        
        参数:
            metadata_path: 元数据CSV文件路径
            image_dir: 图像目录路径
            transform: 图像变换
            target_size: 目标图像大小
            split: 'train', 'val' 或 'test'
        """
        self.metadata = pd.read_csv(metadata_path)
        self.image_dir = image_dir
        self.split = split
        
        # 如果没有提供变换，定义默认变换
        if transform is None:
            if split == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize(target_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(target_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform
            
        # 定义质量分类标签映射
        self.quality_mapping = {
            'excellent': 0,
            'good': 1,
            'fair': 2,
            'poor': 3
        }
        
        # 定义缺陷类型映射
        self.defect_mapping = {
            'none': 0,
            'bubbles': 1,
            'uneven_surface': 2,
            'layer_separation': 3,
            'warping': 4
        }
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """获取单个数据样本"""
        # 获取元数据行
        row = self.metadata.iloc[idx]
        
        # 构建图像路径
        image_path = os.path.join(self.image_dir, f"{row['image_id']}.jpg")
        
        # 加载并变换图像
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        # 提取标签和参数
        # 分类标签: 根据质量分数映射到类别
        quality_score = row['quality_score']
        if quality_score >= 0.9:
            quality_class = self.quality_mapping['excellent']
        elif quality_score >= 0.8:
            quality_class = self.quality_mapping['good']
        elif quality_score >= 0.6:
            quality_class = self.quality_mapping['fair']
        else:
            quality_class = self.quality_mapping['poor']
            
        # 回归参数: 层厚、曝光时间等
        params = torch.tensor([
            row['layer_thickness'], 
            row['exposure_time'], 
            row['intensity'], 
            row['temperature']
        ], dtype=torch.float32)
        
        # 缺陷类型
        defect_type = self.defect_mapping.get(row['defect_type'], 0)
        
        return {
            'image': image,
            'quality_class': torch.tensor(quality_class, dtype=torch.long),
            'params': params,
            'defect_type': torch.tensor(defect_type, dtype=torch.long),
            'image_id': row['image_id']
        }

def get_dataloaders(metadata_path, 
                    image_dir, 
                    batch_size=32, 
                    num_workers=4, 
                    val_split=0.15, 
                    test_split=0.15,
                    target_size=(224, 224)):
    """
    创建训练、验证和测试数据加载器
    
    参数:
        metadata_path: 元数据CSV文件路径
        image_dir: 图像目录路径
        batch_size: 批大小
        num_workers: 数据加载的工作进程数
        val_split: 验证集比例
        test_split: 测试集比例
        target_size: 目标图像大小
        
    返回:
        包含训练、验证和测试数据加载器的字典
    """
    # 加载元数据
    metadata = pd.read_csv(metadata_path)
    
    # 随机打乱并分割数据
    metadata = metadata.sample(frac=1, random_state=42).reset_index(drop=True)
    total_size = len(metadata)
    val_size = int(total_size * val_split)
    test_size = int(total_size * test_split)
    train_size = total_size - val_size - test_size
    
    train_metadata = metadata.iloc[:train_size]
    val_metadata = metadata.iloc[train_size:train_size+val_size]
    test_metadata = metadata.iloc[train_size+val_size:]
    
    # 保存分割后的元数据
    train_metadata.to_csv(os.path.join(os.path.dirname(metadata_path), 'train_metadata.csv'), index=False)
    val_metadata.to_csv(os.path.join(os.path.dirname(metadata_path), 'val_metadata.csv'), index=False)
    test_metadata.to_csv(os.path.join(os.path.dirname(metadata_path), 'test_metadata.csv'), index=False)
    
    # 创建数据集
    train_dataset = PrintQualityDataset(
        os.path.join(os.path.dirname(metadata_path), 'train_metadata.csv'),
        image_dir,
        split='train',
        target_size=target_size
    )
    
    val_dataset = PrintQualityDataset(
        os.path.join(os.path.dirname(metadata_path), 'val_metadata.csv'),
        image_dir,
        split='val',
        target_size=target_size
    )
    
    test_dataset = PrintQualityDataset(
        os.path.join(os.path.dirname(metadata_path), 'test_metadata.csv'),
        image_dir,
        split='test',
        target_size=target_size
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }