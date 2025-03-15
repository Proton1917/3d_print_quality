"""
预处理脚本：处理原始图像并创建训练数据集
"""
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import yaml
from pathlib import Path
import albumentations as A

def parse_args():
    parser = argparse.ArgumentParser(description='数据预处理脚本')
    parser.add_argument('--config', type=str, default='configs/default.yaml', 
                        help='配置文件路径')
    parser.add_argument('--raw_dir', type=str, default='data/raw',
                        help='原始图像目录')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='输出目录')
    parser.add_argument('--metadata', type=str, default='data/metadata.csv',
                        help='元数据CSV文件')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_augmentations(config):
    """设置数据增强流水线"""
    aug_list = []
    
    # 基本增强
    if config['augmentations']['rotation']:
        aug_list.append(A.Rotate(limit=10, p=0.5))
    
    if config['augmentations']['flip']:
        aug_list.append(A.HorizontalFlip(p=0.5))
    
    if config['augmentations']['brightness']:
        aug_list.append(A.RandomBrightnessContrast(
            brightness_limit=0.1, contrast_limit=0.1, p=0.5))
    
    # 添加特殊增强
    if config['augmentations']['noise']:
        aug_list.append(A.GaussNoise(var_limit=(10, 50), p=0.3))
        
    if config['augmentations']['blur']:
        aug_list.append(A.GaussianBlur(blur_limit=(3, 5), p=0.3))
    
    # 创建完整流水线
    return A.Compose(aug_list)

def extract_roi(image, config):
    """提取感兴趣区域"""
    if config['preprocessing']['auto_roi']:
        # 实现自动ROI提取算法
        # 这里仅作示例，实际可能需要更复杂的算法
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 找到最大轮廓
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            # 稍微扩大ROI
            padding = int(min(w, h) * 0.1)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2*padding)
            h = min(image.shape[0] - y, h + 2*padding)
            
            return image[y:y+h, x:x+w]
    
    # 如果未开启自动ROI或提取失败，使用固定剪裁
    h, w = image.shape[:2]
    crop_h = int(h * config['preprocessing']['crop_ratio'])
    crop_w = int(w * config['preprocessing']['crop_ratio'])
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    
    return image[start_h:start_h+crop_h, start_w:start_w+crop_w]

def preprocess_image(image_path, config):
    """预处理单个图像"""
    # 读取图像
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None
    
    # 转换颜色空间
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 噪声去除
    if config['preprocessing']['denoise']:
        image = cv2.fastNlMeansDenoisingColored(
            image, None, 
            h=config['preprocessing']['denoise_strength'],
            hColor=config['preprocessing']['denoise_strength'],
            templateWindowSize=7, searchWindowSize=21)
    
    # 提取ROI
    image = extract_roi(image, config)
    
    # 统一大小
    target_size = config['preprocessing']['target_size']
    image = cv2.resize(image, (target_size, target_size))
    
    # 归一化
    image = image.astype(np.float32) / 255.0
    
    return image

def create_label(metadata_row, config):
    """根据元数据创建标签"""
    # 读取打印参数
    params = {
        'layer_height': float(metadata_row['layer_height']),
        'exposure_time': float(metadata_row['exposure_time']),
        'light_intensity': float(metadata_row['light_intensity']),
        # 根据需要添加更多参数
    }
    
    # 计算参数偏差
    optimal_params = config['optimal_parameters']
    deviations = {}
    
    for key, value in params.items():
        if key in optimal_params:
            # 计算与最优值的相对偏差
            opt_value = optimal_params[key]
            deviations[f"{key}_deviation"] = (value - opt_value) / opt_value
    
    # 质量分类
    # 基于偏差总和进行简单分类
    total_deviation = sum(abs(v) for v in deviations.values())
    
    if total_deviation < 0.05:
        quality_class = 0  # 优
    elif total_deviation < 0.10:
        quality_class = 1  # 良
    elif total_deviation < 0.20:
        quality_class = 2  # 中
    else:
        quality_class = 3  # 差
    
    # 返回标签字典
    return {
        'quality_class': quality_class,
        'params': params,
        'deviations': deviations
    }

def process_dataset(args, config):
    """处理整个数据集"""
    # 确保输出目录存在
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取元数据
    metadata = pd.read_csv(args.metadata)
    
    # 设置数据增强
    augmentation = setup_augmentations(config)
    
    # 处理所有图像
    processed_data = []
    
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
        image_path = Path(args.raw_dir) / row['image_filename']
        
        if not image_path.exists():
            print(f"找不到图像文件: {image_path}")
            continue
        
        # 预处理图像
        processed_image = preprocess_image(image_path, config)
        if processed_image is None:
            continue
        
        # 创建标签
        label = create_label(row, config)
        
        # 保存处理后的图像
        output_filename = f"{idx:06d}.npy"
        output_path = output_dir / output_filename
        np.save(output_path, processed_image)
        
        # 记录处理信息
        processed_data.append({
            'processed_filename': output_filename,
            'original_filename': row['image_filename'],
            'quality_class': label['quality_class'],
            **label['params'],
            **label['deviations']
        })
        
        # 应用数据增强并保存增强版本
        if config['augmentations']['enabled']:
            for aug_idx in range(config['augmentations']['copies']):
                augmented = augmentation(image=processed_image)
                aug_image = augmented['image']
                
                # 保存增强图像
                aug_filename = f"{idx:06d}_aug{aug_idx:02d}.npy"
                aug_path = output_dir / aug_filename
                np.save(aug_path, aug_image)
                
                # 记录增强信息
                processed_data.append({
                    'processed_filename': aug_filename,
                    'original_filename': row['image_filename'],
                    'quality_class': label['quality_class'],
                    'augmented': True,
                    **label['params'],
                    **label['deviations']
                })
    
    # 保存处理后的元数据
    processed_df = pd.DataFrame(processed_data)
    processed_df.to_csv(output_dir / 'processed_metadata.csv', index=False)
    
    print(f"处理完成! 总共处理了 {len(processed_data)} 个样本")

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    process_dataset(args, config)
