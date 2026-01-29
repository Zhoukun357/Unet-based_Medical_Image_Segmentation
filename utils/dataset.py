
import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

from pycocotools.coco import COCO


class CocoTumorDataset(Dataset):
    """
    直接读取 COCO json，并用 annToMask 动态生成二值 mask (0/1)。
    期望目录结构（每个 split 一套）：
      data/train/
        - coco.json   (或 _annotations.coco.json)
        - images/     (图片都在这里)
      data/val/
      data/test/
    """
    def __init__(self, split_root, ann_file="coco.json", image_folder="images", transform=None, target_size=(256, 256)):
        self.split_root = split_root
        self.ann_path = os.path.join(split_root, ann_file)
        self.image_dir = os.path.join(split_root, image_folder)
        self.transform = transform
        self.target_size = target_size

        if not os.path.exists(self.ann_path):
            raise FileNotFoundError(f"COCO annotation file not found: {self.ann_path}")
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image folder not found: {self.image_dir}")

        self.coco = COCO(self.ann_path)
        self.img_ids = sorted(self.coco.getImgIds())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        file_name = img_info["file_name"]

        img_path = os.path.join(self.image_dir, file_name)
        if not os.path.exists(img_path):
            # 有些 COCO file_name 可能带子目录，这里兜底取 basename
            img_path = os.path.join(self.image_dir, os.path.basename(file_name))
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {file_name}")

        # 1) 读图
        image = Image.open(img_path).convert("RGB")

        # 2) 根据 COCO segmentation 动态生成 mask
        h, w = img_info["height"], img_info["width"]
        mask01 = np.zeros((h, w), dtype=np.uint8)

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)

        for ann in anns:
            m = self.coco.annToMask(ann).astype(np.uint8)  # 0/1
            mask01 = np.maximum(mask01, m)

        mask = Image.fromarray((mask01 * 255).astype(np.uint8))  # PIL L

        # 3) 同步 resize（image 用双线性没问题；mask 必须最近邻）
        image = F.resize(image, list(self.target_size))
        mask = F.resize(mask, list(self.target_size), interpolation=Image.NEAREST)

        # 4) 转 Tensor
        if self.transform is not None:
            # 你的 transform 里一般包含 ToTensor / Normalize 等
            image = self.transform(image)
        else:
            image = F.to_tensor(image)

        mask = F.to_tensor(mask)          # [1,H,W] in [0,1]
        mask = (mask > 0).float()         # 强制二值化成 0/1

        return image, mask
