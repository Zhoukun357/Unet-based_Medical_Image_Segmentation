import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def mask_to_rgba(mask, color="red", alpha=0.4):
    """
    将二值掩码转换为 RGBA 格式，用于叠加显示。
    color: 'red', 'green', 'blue', 'yellow'
    """
    mask = mask.astype(np.float32)
    h, w = mask.shape
    img_rgba = np.zeros((h, w, 4), dtype=np.float32)
    
    # 定义颜色字典
    colors = {
        "red":    [1.0, 0.0, 0.0],
        "green":  [0.0, 1.0, 0.0],
        "blue":   [0.0, 0.0, 1.0],
        "yellow": [1.0, 1.0, 0.0],
    }
    
    c = colors.get(color, [1.0, 0.0, 0.0])
    
    img_rgba[..., 0] = c[0]
    img_rgba[..., 1] = c[1]
    img_rgba[..., 2] = c[2]
    img_rgba[..., 3] = mask * alpha  # 仅在 mask 为 1 的地方有透明度，其他全透
    
    return img_rgba

def visualize_result(image, gt_mask, pred_mask, pred_prob, save_path=None):
    """
    生成标准的论文四联图：
    1. 原图 (MRI)
    2. 真实标签 (Ground Truth)
    3. 预测概率热力图 (Confidence Map) - 分析模型确定性
    4. 结果叠加 (Overlay) - 红色代表预测肿瘤区域
    """
    # 确保输入转为 numpy 格式
    if torch.is_tensor(image): image = image.cpu().numpy().transpose(1, 2, 0)
    if torch.is_tensor(gt_mask): gt_mask = gt_mask.cpu().numpy().squeeze()
    if torch.is_tensor(pred_mask): pred_mask = pred_mask.cpu().numpy().squeeze()
    if torch.is_tensor(pred_prob): pred_prob = pred_prob.cpu().numpy().squeeze()

    # 反归一化图像以便显示 (假设之前做了标准化，如果没有则根据情况调整)
    # image = (image * std) + mean 
    
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    
    # 1. Original Image
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Input MRI", fontsize=14, fontweight='bold')
    ax[0].axis('off')
    
    # 2. Ground Truth
    ax[1].imshow(gt_mask, cmap='bone')
    ax[1].set_title("Ground Truth", fontsize=14, fontweight='bold')
    ax[1].axis('off')
    
    # 3. Prediction Confidence (Heatmap)
    im = ax[2].imshow(pred_prob, cmap='jet', vmin=0, vmax=1)
    ax[2].set_title("Confidence Map", fontsize=14, fontweight='bold')
    ax[2].axis('off')
    plt.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
    
    # 4. Overlay (Prediction on Image)
    ax[3].imshow(image, cmap='gray')
    # 生成红色半透明掩码
    rgba_pred = mask_to_rgba(pred_mask, color="red", alpha=0.4)
    ax[3].imshow(rgba_pred)
    ax[3].set_title("Prediction Overlay", fontsize=14, fontweight='bold')
    ax[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()
    plt.close()

def visualize_error_analysis(image, gt_mask, pred_mask, save_path=None):
    """
    【高级功能】生成误差分析图 (FP/FN Analysis)。
    这是论文 Discussion 部分非常有用的图表，用于展示模型“哪里错了”。
    
    颜色编码：
    - 黄色 (Yellow): True Positive (正确预测的肿瘤)
    - 红色 (Red): False Positive (误报，模型以为是肿瘤其实是背景)
    - 绿色 (Green): False Negative (漏报，模型没发现的肿瘤)
    """
    if torch.is_tensor(image): image = image.cpu().numpy().transpose(1, 2, 0)
    if torch.is_tensor(gt_mask): gt_mask = gt_mask.cpu().numpy().squeeze()
    if torch.is_tensor(pred_mask): pred_mask = pred_mask.cpu().numpy().squeeze()

    # 计算 TP, FP, FN
    tp = np.logical_and(pred_mask == 1, gt_mask == 1)
    fp = np.logical_and(pred_mask == 1, gt_mask == 0)
    fn = np.logical_and(pred_mask == 0, gt_mask == 1)

    # 创建 RGB 误差图
    h, w = gt_mask.shape
    error_map = np.zeros((h, w, 3))
    
    # 背景设为灰色以便对比 (可选)
    # error_map += 0.1 

    # 填充颜色
    error_map[tp] = [1, 1, 0] # 黄色：预测正确
    error_map[fp] = [1, 0, 0] # 红色：误报 (False Alarm)
    error_map[fn] = [0, 1, 0] # 绿色：漏报 (Missed)

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. 原始图像
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Original MRI", fontsize=15)
    ax[0].axis('off')

    # 2. 误差地图
    ax[1].imshow(error_map)
    # 创建图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='yellow', label='TP (Correct)'),
        Patch(facecolor='red', label='FP (False Alarm)'),
        Patch(facecolor='green', label='FN (Missed Tumor)')
    ]
    ax[1].legend(handles=legend_elements, loc='upper right')
    ax[1].set_title("Error Analysis Map", fontsize=15)
    ax[1].axis('off')

    # 3. 叠加误差到原图
    ax[2].imshow(image, cmap='gray')
    ax[2].imshow(error_map, alpha=0.5) # 半透明叠加
    ax[2].set_title("Error Overlay", fontsize=15)
    ax[2].axis('off')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved error analysis to {save_path}")
    
    plt.show()
    plt.close()