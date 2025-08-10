import os
import argparse
import yaml
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train import main as train_main

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="训练3D打印质量评估模型")
    # 默认从项目根目录运行脚本时可直接使用默认配置
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='配置文件路径')
    args = parser.parse_args()
    
    # 调用训练模块的主函数
    # 将解析到的配置路径传递给训练入口（training/train.py支持argv参数）
    train_main(["--config", args.config])

if __name__ == '__main__':
    main()
