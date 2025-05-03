import os
import sys
import argparse
import yaml
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# 添加父目录到系统路径，以便导入项目模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.model import PrintQualityModel
from utils.visualization import plot_prediction_heatmap, visualize_attention_weights

class PrintQualityInference:
    """3D打印质量评估推理类"""
    
    def __init__(self, model_path, config_path=None, device=None):
        """
        初始化推理器
        
        参数:
            model_path: 模型权重路径
            config_path: 配置文件路径，如果为None则尝试从模型目录读取
            device: 计算设备，如果为None则自动选择
        """
        # 确定配置文件路径
        if config_path is None:
            model_dir = os.path.dirname(model_path)
            config_path = os.path.join(model_dir, 'best_config.yaml')
            if not os.path.exists(config_path):
                config_path = os.path.join(os.path.dirname(os.path.dirname(model_path)), 'configs', 'default.yaml')
        
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 设置设备
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() and self.config['use_gpu'] else 'cpu')
        else:
            self.device = device
        
        print(f"使用设备: {self.device}")
        
        # 加载模型
        self._load_model(model_path)
        
        # 设置图像转换
        self.transform = transforms.Compose([
            transforms.Resize(tuple(self.config['data']['target_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 设置标签映射
        self.quality_labels = ['优', '良', '中', '差']
        self.defect_labels = ['无缺陷', '气泡', '表面不均', '层分离', '翘曲']
        
        print("推理器初始化完成")
    
    def _load_model(self, model_path):
        """
        加载模型
        
        参数:
            model_path: 模型权重路径
        """
        model_config = self.config['model']
        self.model = PrintQualityModel(
            backbone_type=model_config['backbone_type'],
            backbone_config={
                'model_name': model_config['backbone_name'],
                'pretrained': False,
                'freeze_layers': False
            },
            use_attention=model_config['use_attention'],
            num_classes=model_config['num_classes'],
            num_params=model_config['num_params'],
            num_defect_types=model_config['num_defect_types']
        ).to(self.device)
        
        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"成功加载模型权重: {model_path}")
    
    def preprocess_image(self, image_path):
        """
        预处理图像
        
        参数:
            image_path: 图像路径
            
        返回:
            预处理后的图像张量
        """
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 应用转换
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        return image_tensor, image
    
    def predict(self, image_path, visualize=False):
        """
        进行预测
        
        参数:
            image_path: 图像路径
            visualize: 是否可视化结果
            
        返回:
            预测结果字典
        """
        # 预处理图像
        image_tensor, original_image = self.preprocess_image(image_path)
        
        # 进行推理
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # 处理质量分类结果
        quality_probs = torch.softmax(predictions['quality'], dim=1)
        quality_scores, quality_class = torch.max(quality_probs, dim=1)
        
        quality_result = {
            'class_id': quality_class.item(),
            'class_name': self.quality_labels[quality_class.item()],
            'confidence': quality_scores.item(),
            'probabilities': quality_probs[0].cpu().numpy().tolist()
        }
        
        # 处理缺陷检测结果
        defect_probs = torch.softmax(predictions['defects'], dim=1)
        defect_scores, defect_class = torch.max(defect_probs, dim=1)
        
        defect_result = {
            'class_id': defect_class.item(),
            'class_name': self.defect_labels[defect_class.item()],
            'confidence': defect_scores.item(),
            'probabilities': defect_probs[0].cpu().numpy().tolist()
        }
        
        # 处理参数预测结果
        params_pred = predictions['params'][0].cpu().numpy().tolist()
        param_names = ['层厚(mm)', '曝光时间(s)', '强度(%)', '温度(°C)']
        
        param_result = {name: value for name, value in zip(param_names, params_pred)}
        
        # 组合结果
        result = {
            'quality': quality_result,
            'defect': defect_result,
            'parameters': param_result
        }
        
        # 可视化
        if visualize:
            self.visualize_prediction(image_path, result)
        
        return result
    
    def visualize_prediction(self, image_path, prediction):
        """
        可视化预测结果
        
        参数:
            image_path: 图像路径
            prediction: 预测结果
        """
        # 读取图像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(15, 10))
        
        # 显示原始图像
        plt.subplot(2, 2, 1)
        plt.imshow(image)
        plt.title("原始图像")
        plt.axis('off')
        
        # 显示质量分类结果
        quality = prediction['quality']
        plt.subplot(2, 2, 2)
        plt.bar(self.quality_labels, quality['probabilities'], color='skyblue')
        plt.title(f"质量评估: {quality['class_name']} (置信度: {quality['confidence']:.2f})")
        plt.ylim(0, 1.0)
        
        # 显示缺陷检测结果
        defect = prediction['defect']
        plt.subplot(2, 2, 3)
        plt.bar(self.defect_labels, defect['probabilities'], color='salmon')
        plt.title(f"缺陷类型: {defect['class_name']} (置信度: {defect['confidence']:.2f})")
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45)
        
        # 显示参数预测结果
        params = prediction['parameters']
        plt.subplot(2, 2, 4)
        plt.bar(list(params.keys()), list(params.values()), color='lightgreen')
        plt.title("参数预测")
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
    def batch_predict(self, image_dir, output_csv=None):
        """
        批量预测
        
        参数:
            image_dir: 图像目录
            output_csv: 输出CSV文件路径
            
        返回:
            预测结果列表
        """
        import glob
        import pandas as pd
        
        # 获取所有图像路径
        image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + \
                     glob.glob(os.path.join(image_dir, "*.png")) + \
                     glob.glob(os.path.join(image_dir, "*.jpeg"))
        
        print(f"找到{len(image_paths)}张图像，开始批量预测...")
        
        results = []
        
        # 处理每张图像
        for image_path in image_paths:
            filename = os.path.basename(image_path)
            print(f"处理图像: {filename}")
            
            # 预测
            prediction = self.predict(image_path)
            
            # 整理结果
            result = {
                'filename': filename,
                'quality_class': prediction['quality']['class_name'],
                'quality_confidence': prediction['quality']['confidence'],
                'defect_type': prediction['defect']['class_name'],
                'defect_confidence': prediction['defect']['confidence']
            }
            
            # 添加参数预测结果
            for param_name, value in prediction['parameters'].items():
                result[param_name] = value
            
            results.append(result)
        
        # 保存结果
        if output_csv:
            df = pd.DataFrame(results)
            df.to_csv(output_csv, index=False)
            print(f"结果已保存到: {output_csv}")
        
        return results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="3D打印质量评估推理")
    parser.add_argument('--model_path', type=str, required=True, help='模型权重路径')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--image_path', type=str, default=None, help='输入图像路径')
    parser.add_argument('--image_dir', type=str, default=None, help='输入图像目录（批量处理）')
    parser.add_argument('--output_csv', type=str, default=None, help='输出CSV文件路径（批量处理）')
    parser.add_argument('--visualize', action='store_true', help='可视化结果')
    
    args = parser.parse_args()
    
    # 创建推理器
    inferencer = PrintQualityInference(args.model_path, args.config)
    
    # 单图像处理
    if args.image_path:
        result = inferencer.predict(args.image_path, visualize=args.visualize)
        print("\n预测结果:")
        print(f"质量等级: {result['quality']['class_name']} (置信度: {result['quality']['confidence']:.2f})")
        print(f"缺陷类型: {result['defect']['class_name']} (置信度: {result['defect']['confidence']:.2f})")
        print("预测参数:")
        for param_name, value in result['parameters'].items():
            print(f"  {param_name}: {value:.4f}")
    
    # 批量处理
    elif args.image_dir:
        results = inferencer.batch_predict(args.image_dir, args.output_csv)
        print(f"共处理{len(results)}张图像")
    
    else:
        parser.error("请提供--image_path或--image_dir参数")

if __name__ == '__main__':
    main()