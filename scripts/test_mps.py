import os
import sys
import torch

# 允许从项目根目录直接运行
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.device import get_device
from models.model import PrintQualityModel


def main():
    print("MPS 自检: 开始")
    has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    print(f"MPS 可用: {has_mps}")

    # 选择设备（优先 CUDA，其次 MPS，最后 CPU）
    device = get_device(use_gpu=True)
    print(f"选择设备: {device}")

    # 构造一个极简模型并执行一次前向（不训练）
    model = PrintQualityModel(
        backbone_type='vit',
        backbone_config={
            'model_name': 'vit_tiny_patch16_224',
            'pretrained': False,
            'freeze_layers': False,
        },
        use_attention=False,
        num_classes=4,
        num_params=4,
        num_defect_types=5,
    ).to(device)

    model.eval()
    x = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        out = model(x)
    print("前向完成:")
    print("  quality:", tuple(out['quality'].shape))
    print("  defects:", tuple(out['defects'].shape))
    print("  params:", tuple(out['params'].shape))
    print("MPS 自检: 结束")


if __name__ == '__main__':
    main()
