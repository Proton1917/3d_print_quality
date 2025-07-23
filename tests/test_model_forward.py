import torch
from models.model import PrintQualityModel

def test_forward_shapes():
    model = PrintQualityModel(
        backbone_type='resnet',
        backbone_config={'model_name': 'resnet18', 'pretrained': False, 'freeze_layers': False},
        use_attention=False,
        num_classes=4,
        num_params=4,
        num_defect_types=5
    )
    x = torch.randn(2, 3, 224, 224)
    outputs = model(x)
    assert outputs['quality'].shape == (2, 4)
    assert outputs['params'].shape == (2, 4)
    assert outputs['defects'].shape == (2, 5)

