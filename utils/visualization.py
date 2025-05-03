import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import cv2
from matplotlib.colors import LinearSegmentedColormap

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=None):
    """
    绘制混淆矩阵
    
    参数:
        cm: 混淆矩阵
        classes: 类别标签
        normalize: 是否归一化
        title: 图表标题
        cmap: 颜色映射
    """
    if cmap is None:
        cmap = plt.cm.Blues
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)
    
    # 添加文本标注
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=14)
    
    plt.ylabel('真实标签', fontsize=14)
    plt.xlabel('预测标签', fontsize=14)
    plt.tight_layout()

def plot_prediction_heatmap(image, model, transform, device, alpha=0.5):
    """
    绘制模型预测的热力图
    
    参数:
        image: 输入图像
        model: 预训练模型
        transform: 图像变换
        device: 计算设备
        alpha: 热力图透明度
    
    返回:
        叠加了热力图的图像
    """
    model.eval()
    
    # 预处理图像
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # 获取模型预测和特征
    with torch.no_grad():
        # 这里假设模型有一个hook可以获取最后一层特征图
        # 具体实现取决于模型架构
        predictions = model(img_tensor)
        
        # 假设我们能够从模型中提取特征图
        # 这里需要根据实际模型调整
        feature_maps = None  # 需要从模型中获取
    
    # 如果无法直接获取特征图，可以使用梯度CAM等技术
    # 这里示例一个简单的热力图生成机制
    if feature_maps is not None:
        # 转换为热力图
        feature_map = feature_maps[0].mean(0).cpu().numpy()
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
        
        # 调整大小为原图尺寸
        heatmap = cv2.resize(feature_map, (image.shape[1], image.shape[0]))
        
        # 转换为伪彩色热力图
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # 叠加到原图
        output = cv2.addWeighted(image, 1-alpha, heatmap_colored, alpha, 0)
    else:
        # 如果无法获取特征图，返回原图
        output = image
    
    return output

def visualize_attention_weights(attention_weights, image=None, alpha=0.7):
    """
    可视化注意力权重
    
    参数:
        attention_weights: 注意力权重矩阵
        image: 原始图像 (可选)
        alpha: 热力图透明度
        
    返回:
        注意力权重热力图
    """
    # 处理注意力权重
    att_map = attention_weights.cpu().numpy()
    att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min())
    
    # 创建热力图
    plt.figure(figsize=(10, 8))
    
    if image is not None:
        # 调整注意力权重大小以匹配图像尺寸
        h, w = image.shape[:2]
        att_map_resized = cv2.resize(att_map, (w, h))
        
        # 创建热力图配色
        cmap = plt.cm.jet
        att_map_colored = cmap(att_map_resized)[:, :, :3]  # 获取RGB部分
        att_map_colored = np.uint8(att_map_colored * 255)
        
        # 混合原图和热力图
        overlay = cv2.addWeighted(
            image, 1-alpha, 
            att_map_colored, alpha, 0
        )
        
        plt.imshow(overlay)
    else:
        # 直接显示热力图
        plt.imshow(att_map, cmap='jet')
        
    plt.colorbar(label='Attention Weight')
    plt.title('Attention Weights Visualization')
    plt.axis('off')
    
    return plt.gcf()  # 返回当前图形

def plot_training_history(history, output_path=None):
    """
    绘制训练历史记录
    
    参数:
        history: 训练历史记录字典
        output_path: 输出文件路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # 绘制损失曲线
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制准确率曲线
    if 'train_acc' in history and 'val_acc' in history:
        ax2.plot(history['train_acc'], label='Train Accuracy')
        ax2.plot(history['val_acc'], label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        
    return fig