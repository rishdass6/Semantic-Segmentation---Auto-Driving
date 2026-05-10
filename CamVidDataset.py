import os
import torch.utils.data as data
from PIL import Image
import glob
import numpy as np
import albumentations as A

class CamVidDataset(data.Dataset):
    def __init__(self, image_dir, mask_dir, class_colors, transform):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.class_colors = class_colors
        self.transform = transform

        self.images = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        self.masks = [os.path.join(mask_dir, os.path.basename(img).replace(".png", "_L.png")) for img in self.images]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("RGB"))

        h, w = mask.shape[:2]
        label_map = np.zeros((h,w), dtype=np.int64)

        for class_idx, color in enumerate(self.class_colors):
            match = np.all(mask == np.array(color, dtype=np.uint8), axis= -1)
            label_map[match] = class_idx

        if self.transform is not None:
            augmented = self.transform(image = image, mask = label_map)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask
