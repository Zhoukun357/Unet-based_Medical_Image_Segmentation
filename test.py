import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from models.unet import UNet

# Config
MODEL_PATH = 'checkpoints/best_unet_tumor.pth'
TEST_IMAGE_DIR = 'data/test/images'
OUTPUT_MASK_DIR = 'output/masks'
OUTPUT_OVERLAY_DIR = 'output/overlays'
IMG_SIZE = (256, 256)
THRESHOLD = 0.5

os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)
os.makedirs(OUTPUT_OVERLAY_DIR, exist_ok=True)


def run_inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = UNet(n_channels=3, n_classes=1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    transform = T.Compose([
        T.Resize(IMG_SIZE),
        T.ToTensor()
    ])

    image_list = sorted(os.listdir(TEST_IMAGE_DIR))

    with torch.no_grad():
        for img_name in image_list:
            if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            img_path = os.path.join(TEST_IMAGE_DIR, img_name)

            # -------- Load image --------
            original_img = Image.open(img_path).convert("RGB")
            img_tensor = transform(original_img).unsqueeze(0).to(device)

            # -------- Inference --------
            output = model(img_tensor)
            prob = torch.sigmoid(output).squeeze().cpu().numpy()

            pred_mask = (prob > THRESHOLD).astype(np.uint8) * 255

            # -------- Save binary mask --------
            mask_img = Image.fromarray(pred_mask)
            mask_img.save(os.path.join(
                OUTPUT_MASK_DIR,
                img_name.replace('.jpg', '.png')
            ))

            # -------- Create overlay --------
            img_np = np.array(original_img.resize(IMG_SIZE))

            red_mask = np.zeros_like(img_np)
            red_mask[..., 0] = 255  # red channel

            overlay = img_np.copy()
            overlay[pred_mask == 255] = (
                0.6 * img_np[pred_mask == 255] +
                0.4 * red_mask[pred_mask == 255]
            ).astype(np.uint8)

            overlay_img = Image.fromarray(overlay)
            overlay_img.save(os.path.join(
                OUTPUT_OVERLAY_DIR,
                img_name.replace('.jpg', '.png')
            ))

            print(f"[✓] Processed: {img_name}")

    print("Inference finished. Results saved to output/")

if __name__ == '__main__':
    run_inference()
