import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm 
import matplotlib.pyplot as plt


from models.unet import UNet
from utils.dataset import CocoTumorDataset
from utils.metrics import dice_coeff, iou_score
import torchvision.transforms as T

# 1. 定义混合损失函数
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets):
        # inputs 经过 sigmoid 处理，或者在 loss 内部处理
        # 这里假设输入是 logits，所以在 BCE 中用 with_logits
        
        # BCE Loss
        bce_loss = nn.BCEWithLogitsLoss()(inputs, targets)
        
        # Dice Loss
        inputs = torch.sigmoid(inputs)
        smooth = 1e-5
        input_flat = inputs.view(-1)
        target_flat = targets.view(-1)
        intersection = (input_flat * target_flat).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(input_flat.sum() + target_flat.sum() + smooth)
        
        # 组合 Loss: 0.5 * BCE + 0.5 * Dice (权重可调)
        return 0.5 * bce_loss + 0.5 * dice_loss

# 2. 配置超参数
HYPERPARAMS = {
    "LR": 1e-4,
    "BATCH_SIZE": 8,
    "EPOCHS": 50,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "TRAIN_ROOT": "data/train",
    "VAL_ROOT": "data/val",
    "ANN_NAME": "_annotations.coco.json", 
    "IMAGE_FOLDER": "images",
    "SAVE_PATH": "checkpoints/best_unet_tumor.pth"
}

def train():
    # 确保保存模型的目录存在
    os.makedirs("checkpoints", exist_ok=True)
    
    # 1. 数据准备
    train_transform = T.Compose([
        T.ToTensor(),
        # 如果你训练时用过 Normalize，这里要加上同样的 Normalize
        # T.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])
    val_transform = T.Compose([T.ToTensor()])

    train_dataset = CocoTumorDataset(
        split_root=HYPERPARAMS["TRAIN_ROOT"],
        ann_file=HYPERPARAMS["ANN_NAME"],
        image_folder=HYPERPARAMS["IMAGE_FOLDER"],
        transform=train_transform,
        target_size=(256, 256)
    )
    val_dataset = CocoTumorDataset(
        split_root=HYPERPARAMS["VAL_ROOT"],
        ann_file=HYPERPARAMS["ANN_NAME"],
        image_folder=HYPERPARAMS["IMAGE_FOLDER"],
        transform=val_transform,
        target_size=(256, 256)
    )

    train_loader = DataLoader(train_dataset, batch_size=HYPERPARAMS["BATCH_SIZE"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=HYPERPARAMS["BATCH_SIZE"], shuffle=False, num_workers=2)

    # 2. 模型、优化器、损失函数
    model = UNet(n_channels=3, n_classes=1).to(HYPERPARAMS["DEVICE"])
    optimizer = optim.Adam(model.parameters(), lr=HYPERPARAMS["LR"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5) # 监控 Dice，不升则降 LR
    criterion = DiceBCELoss()

    # 3. 训练循环
    best_dice = 0.0
    history = {'train_loss': [], 'val_dice': [], 'val_iou': []}

    print(f"Starting training on {HYPERPARAMS['DEVICE']}...")

    for epoch in range(HYPERPARAMS["EPOCHS"]):
        model.train()
        epoch_loss = 0
        
        # 使用 tqdm 显示进度条
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{HYPERPARAMS["EPOCHS"]}', unit='img') as pbar:
            for batch_imgs, batch_masks in train_loader:
                batch_imgs = batch_imgs.to(HYPERPARAMS["DEVICE"])
                batch_masks = batch_masks.to(HYPERPARAMS["DEVICE"])

                optimizer.zero_grad()
                pred_masks = model(batch_imgs)
                
                loss = criterion(pred_masks, batch_masks)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(1)

        # 4. 验证阶段
        model.eval()
        val_dice_score = 0
        val_iou = 0
        with torch.no_grad():
            for val_imgs, val_masks in val_loader:
                val_imgs = val_imgs.to(HYPERPARAMS["DEVICE"])
                val_masks = val_masks.to(HYPERPARAMS["DEVICE"])
                
                val_preds = model(val_imgs)
                val_probs = torch.sigmoid(val_preds) # 转为概率
                
                val_dice_score += dice_coeff(val_probs, val_masks)
                val_iou += iou_score(val_probs, val_masks)

        # 计算平均指标
        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_dice = val_dice_score / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        
        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['val_dice'].append(avg_val_dice)
        history['val_iou'].append(avg_val_iou)

        # 更新学习率
        scheduler.step(avg_val_dice)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"End of Epoch {epoch+1} | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f} | Val Dice: {avg_val_dice:.4f} | Val IoU: {avg_val_iou:.4f}")

        # 5. 保存最佳模型
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save(model.state_dict(), HYPERPARAMS["SAVE_PATH"])
            print(f"★ Model Saved! New Best Dice: {best_dice:.4f}")
    
    # 6. 训练结束，绘制训练曲线
    plot_training_history(history)

def plot_training_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss 曲线
    ax[0].plot(history['train_loss'], label='Train Loss', color='red')
    ax[0].set_title('Training Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    
    # Dice & IoU 曲线
    ax[1].plot(history['val_dice'], label='Val Dice', color='blue')
    ax[1].plot(history['val_iou'], label='Val IoU', color='green')
    ax[1].set_title('Validation Metrics')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Score')
    ax[1].legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300)
    print("Training curves saved as training_curves.png")

if __name__ == "__main__":
    train()