import os
import argparse
import yaml
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def create_directory(directory):
    """创建目录"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def preprocess_image(image_path, output_path, target_size=(224, 224), denoise=True, normalize=False):
    """
    预处理单个图像
    
    参数:
        image_path: 输入图像路径
        output_path: 输出图像路径
        target_size: 目标大小
        denoise: 是否进行去噪
        normalize: 是否归一化
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"警告：无法读取图像 {image_path}")
        return False
    
    # 去噪
    if denoise:
        # 高斯滤波去噪
        image = cv2.GaussianBlur(image, (5, 5), 0)
        
        # 中值滤波去噪 (对椒盐噪声效果好)
        image = cv2.medianBlur(image, 5)
    
    # 调整大小
    image = cv2.resize(image, target_size)
    
    # 归一化
    if normalize:
        image = image.astype(np.float32) / 255.0
    
    # 保存图像
    cv2.imwrite(output_path, image)
    return True

def extract_roi(image_path, output_path, roi_method='adaptive'):
    """
    提取感兴趣区域
    
    参数:
        image_path: 输入图像路径
        output_path: 输出图像路径
        roi_method: ROI提取方法，'fixed' 或 'adaptive'
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"警告：无法读取图像 {image_path}")
        return False
    
    if roi_method == 'fixed':
        # 固定ROI提取（根据预定义的区域）
        h, w = image.shape[:2]
        # 假设ROI在中心位置，并占据图像的60%
        center_x, center_y = w // 2, h // 2
        roi_w, roi_h = int(w * 0.6), int(h * 0.6)
        x1, y1 = center_x - roi_w // 2, center_y - roi_h // 2
        x2, y2 = x1 + roi_w, y1 + roi_h
        
        # 提取ROI
        roi = image[y1:y2, x1:x2]
    else:  # 自适应ROI提取
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 高斯滤波
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 自适应阈值分割
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 寻找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 找到最大的轮廓
            max_contour = max(contours, key=cv2.contourArea)
            
            # 获取边界框
            x, y, w, h = cv2.boundingRect(max_contour)
            
            # 提取ROI，添加一些边距
            padding = 10
            x1, y1 = max(0, x - padding), max(0, y - padding)
            x2, y2 = min(image.shape[1], x + w + padding), min(image.shape[0], y + h + padding)
            
            # 提取ROI
            roi = image[y1:y2, x1:x2]
        else:
            # 如果没有找到有效轮廓，使用原始图像
            roi = image
    
    # 保存ROI图像
    cv2.imwrite(output_path, roi)
    return True

def process_dataset(config):
    """
    处理整个数据集
    
    参数:
        config: 配置字典
    """
    # 获取配置
    raw_dir = config['data']['raw_dir']
    processed_dir = config['data']['processed_dir']
    roi_dir = os.path.join(processed_dir, 'roi')
    metadata_path = config['data']['metadata_path']
    target_size = tuple(config['preprocessing']['target_size'])
    denoise = config['preprocessing']['denoise']
    normalize = config['preprocessing']['normalize']
    roi_method = config['preprocessing']['roi_method']
    
    # 创建目录
    create_directory(processed_dir)
    create_directory(roi_dir)
    
    # 加载元数据
    metadata = pd.read_csv(metadata_path)
    
    print(f"开始处理数据集，共{len(metadata)}个样本")
    
    # 处理每个样本
    for i, row in tqdm(metadata.iterrows(), total=len(metadata)):
        image_id = row['image_id']
        
        # 构建路径
        raw_image_path = os.path.join(raw_dir, f"{image_id}.jpg")
        roi_image_path = os.path.join(roi_dir, f"{image_id}.jpg")
        processed_image_path = os.path.join(processed_dir, f"{image_id}.jpg")
        
        # 提取ROI
        if os.path.exists(raw_image_path):
            success = extract_roi(raw_image_path, roi_image_path, roi_method)
            if not success:
                continue
        else:
            print(f"警告：找不到图像 {raw_image_path}")
            continue
        
        # 预处理图像
        preprocess_image(
            roi_image_path, 
            processed_image_path, 
            target_size=target_size, 
            denoise=denoise, 
            normalize=normalize
        )
    
    print("数据集处理完成！")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="预处理3D打印质量评估数据集")
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 处理数据集
    process_dataset(config)

if __name__ == '__main__':
    main()