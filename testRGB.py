import numpy as np
from PIL import Image
import pandas as pd

df = pd.read_csv("./CamVid/class_dict.csv")
class_colors = [(row.r, row.g, row.b) for _, row in df.iterrows()]

mask = np.array(Image.open('./CamVid/test_labels/0001TP_006690_L.png'))

h, w = mask.shape[:2]
label_map = np.zeros((h, w), dtype = np.int64)

for class_idx, color in enumerate(class_colors):
    match = np.all(mask == np.array(color, dtype =np.uint8), axis = -1)
    label_map[match] = class_idx

print("Unique class indices:", np.unique(label_map))
print("Expected range: 0 –", len(class_colors) - 1)

color_array = np.array(class_colors, dtype = np.uint8)
reconstructed = color_array[label_map]

Image.fromarray(reconstructed).save("reconstructed_mask.png")
Image.fromarray(mask).save("original_mask.png")
