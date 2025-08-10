import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(target_size=(224, 224)):
    """
    获取训练数据增强变换
    
    参数:
        target_size: 目标图像大小
        
    返回:
        Albumentations变换对象
    """
    return A.Compose([
        A.Resize(height=target_size[0], width=target_size[1]),
        A.OneOf([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ], p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.MotionBlur(blur_limit=7, p=0.5),
        ], p=0.3),
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.05, p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.05, p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, p=0.5),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2.0, p=0.5),
            A.Equalize(p=0.5),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
        ], p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_test_transforms(target_size=(224, 224)):
    """
    获取验证/测试数据增强变换
    
    参数:
        target_size: 目标图像大小
        
    返回:
        Albumentations变换对象
    """
    return A.Compose([
        A.Resize(height=target_size[0], width=target_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
def get_test_time_augmentations(target_size=(224, 224), n_augmentations=5):
    """
    获取测试时间增强变换列表
    
    参数:
        target_size: 目标图像大小
        n_augmentations: 创建的增强变换数量
        
    返回:
        包含多个Albumentations变换对象的列表
    """
    tta_transforms = []
    
    # 基本变换
    base_transform = A.Compose([
        A.Resize(height=target_size[0], width=target_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    tta_transforms.append(base_transform)
    
    # 水平翻转
    hflip_transform = A.Compose([
        A.Resize(height=target_size[0], width=target_size[1]),
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    tta_transforms.append(hflip_transform)
    
    # 垂直翻转
    vflip_transform = A.Compose([
        A.Resize(height=target_size[0], width=target_size[1]),
        A.VerticalFlip(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    tta_transforms.append(vflip_transform)
    
    # 旋转90度
    rotate90_transform = A.Compose([
        A.Resize(height=target_size[0], width=target_size[1]),
        A.RandomRotate90(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    tta_transforms.append(rotate90_transform)
    
    # 亮度对比度变化
    bright_contrast_transform = A.Compose([
        A.Resize(height=target_size[0], width=target_size[1]),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    tta_transforms.append(bright_contrast_transform)
    
    return tta_transforms[:n_augmentations]