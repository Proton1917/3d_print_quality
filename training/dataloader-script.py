"""
数据加载器 - 用于加载处理后的3D打印图像数据
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2

class PrintQualityDataset(Dataset):
    """3D打印质量评估数据集"""
    
    def __init__(
        self,
        metadata_path: Union[str, Path],
        data_dir: Union[str, Path],
        transform: Optional[A.Compose] = None,
        mode: str = 'train',
        quality_label_map: Optional[Dict[int, int]] = None
    ):
        """
        初始化数据集
        
        Args:
            metadata_path: 元数据CSV文件路径
            data_dir: 处理后图像目录
            transform: albumentations转换
            mode: 'train', 'val', 或 'test'
            quality_label_map: 质量标签映射字典（可用于平衡类别）
        """
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.transform = transform
        
        # 读取元数据
        self.metadata = pd.read_csv(metadata_path)
        
        # 质量标签映射（可选）
        self.quality_label_map = quality_label_map
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 获取元数据行
        row = self.metadata.iloc[idx]
        
        # 加载图像
        img_path = self.data_dir / row['processed_filename']
        image = np.load(img_path.as_posix())
        
        # 应用变换
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # 确保图像是 PyTorch 张量
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        # 准备标签
        quality_class = row['quality_class']
        if self.quality_label_map and quality_class in self.quality_label_map:
            quality_class = self.quality_label_map[quality_class]
        
        # 解析参数 - 默认为列中的数值
        parameters = np.array([
            row['layer_height'],
            row['exposure_time'],
            row['light_intensity']
        ], dtype=np.float32)
        
        # 解析缺陷 - 如果有的话
        if 'defect_bubble' in row:
            defects = np.array([
                row['defect_bubble'],
                row['defect_layer_shift'],
                row['defect_uneven'],
                row['defect_warping'],
                row['defect_stringing']
            ], dtype=np.float32)
        else:
            # 没有缺陷标签，使用零向量
            defects = np.zeros(5, dtype=np.float32)
        
        return {
            'image': image,
            'quality_class': torch.tensor(quality_class, dtype=torch.long),
            'parameters': torch.tensor(parameters, dtype=torch.float),
            'defects': torch.tensor(defects, dtype=torch.float)
        }

def get_transforms(config: Dict, mode: str = 'train') -> A.Compose:
    """获取数据转换"""
    if mode == 'train':
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.GaussNoise(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

def create_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建训练、验证和测试数据加载器"""
    # 读取配置
    data_dir = config['data']['processed_dir']
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']
    
    # 创建变换
    train_transform = get_transforms(config, 'train')
    val_transform = get_transforms(config, 'val')
    test_transform = get_transforms(config, 'test')
    
    # 创建数据集
    train_dataset = PrintQualityDataset(
        metadata_path=os.path.join(data_dir, 'train_metadata.csv'),
        data_dir=data_dir,
        transform=train_transform,
        mode='train'
    )
    
    val_dataset = PrintQualityDataset(
        metadata_path=os.path.join(data_dir, 'val_metadata.csv'),
        data_dir=data_dir,
        transform=val_transform,
        mode='val'
    )
    
    test_dataset = PrintQualityDataset(
        metadata_path=os.path.join(data_dir, 'test_metadata.csv'),
        data_dir=data_dir,
        transform=test_transform,
        mode='test'
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
        batch_size=batch_size*2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size*2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

# 用于Mac MPS和CUDA的优化加载器创建函数
def create_optimized_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建针对特定硬件优化的数据加载器"""
    # 基本创建
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    # 根据设备进行优化
    if config['training']['device'] == 'mps':
        # MPS优化 - 可能需要更少的worker
        num_workers = min(config['training']['num_workers'], 4)
        for loader in [train_loader, val_loader, test_loader]:
            loader.num_workers = num_workers
            loader.pin_memory = True
    
    elif config['training']['device'] == 'cuda':
        # CUDA优化 - 可以使用更多worker
        for loader in [train_loader, val_loader, test_loader]:
            loader.pin_memory = True
            loader.persistent_workers = True if loader.num_workers > 0 else False
    
    return train_loader, val_loader, test_loader
