from PIL import Image
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from matplotlib.image import imread
import tensorflow as tf

img_dir = "../../Desktop/MLDB/ground-imagery/images"
mask_dir = "../../Desktop/MLDB/ground-imagery/labels"
img_files = os.listdir(img_dir)
mask_files = os.listdir(mask_dir)

#pixel manipulation for ground truth
def preprocess_ground_masks():
    for i in mask_files:
        image = Image.open(f"../../Desktop/MLDB/ground-imagery/labels/{i}" )
        image = image.point(lambda e: e * 255)
        image.save(f"ground-imagery/labels/{i}")

images = []
masks = []
#prepare images for unet
def preprocess_images_masks():
    for img_name, mask_name in zip(img_files, mask_files):
        img_path = os.path.join(img_dir, img_name)
        mask_path = os.path.join(mask_dir, mask_name)

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        mask = np.expand_dims(mask, axis=-1)

        img = cv2.resize(img, (256, 192))
        mask = cv2.resize(mask, (256, 192))

        images.append(img)
        masks.append(mask)

preprocess_images_masks()
images = np.array(images)
masks = np.array(masks)

X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)