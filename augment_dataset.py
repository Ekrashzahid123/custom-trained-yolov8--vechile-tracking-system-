import os
import glob
import shutil
import cv2
import numpy as np
from collections import Counter

def apply_photometric_aug(img, mode):
    """
    Applies brightness, contrast, HSV jitter, or blur based on mode.
    Mode 1: Brightness boost & contrast boost
    Mode 2: Low-contrast / dim lighting + mild blur
    Mode 3: High contrast + HSV saturation boost
    """
    aug_img = img.copy()
    if mode == 1:
        # Brightness & contrast boost
        aug_img = cv2.convertScaleAbs(aug_img, alpha=1.2, beta=15)
        # HSV saturation boost
        hsv = cv2.cvtColor(aug_img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.25, 0, 255)
        aug_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    elif mode == 2:
        # Low contrast / evening shadow simulation
        aug_img = cv2.convertScaleAbs(aug_img, alpha=0.85, beta=-15)
        # Mild Gaussian blur to simulate low lighting noise
        aug_img = cv2.GaussianBlur(aug_img, (3, 3), 0)
    elif mode == 3:
        # High contrast + subtle hue shift
        aug_img = cv2.convertScaleAbs(aug_img, alpha=1.3, beta=5)
        hsv = cv2.cvtColor(aug_img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + 5) % 180  # slight hue shift
        aug_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return aug_img

def flip_yolo_boxes(boxes):
    """
    Flips YOLO bounding boxes horizontally: x_center -> 1.0 - x_center
    """
    flipped_boxes = []
    for cls_id, x, y, w, h in boxes:
        x_new = round(1.0 - x, 6)
        flipped_boxes.append((cls_id, x_new, y, w, h))
    return flipped_boxes

def generate_augmented_dataset():
    src_dataset = r'd:\Model Imrovement\Vehicle-counting.v1i.yolov8'
    dst_dataset = r'd:\Model Imrovement\Vehicle-counting-augmented'
    
    print(f"Creating augmented dataset at {dst_dataset}...")
    
    # Remove destination dir if exists and recreate
    if os.path.exists(dst_dataset):
        shutil.rmtree(dst_dataset)
        
    os.makedirs(os.path.join(dst_dataset, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(dst_dataset, 'train', 'labels'), exist_ok=True)
    
    # Copy valid and test directly
    shutil.copytree(os.path.join(src_dataset, 'valid'), os.path.join(dst_dataset, 'valid'))
    shutil.copytree(os.path.join(src_dataset, 'test'), os.path.join(dst_dataset, 'test'))
    
    src_train_imgs = glob.glob(os.path.join(src_dataset, 'train', 'images', '*.*'))
    
    augmented_img_count = 0
    class_counts_before = Counter()
    class_counts_after = Counter()
    
    for img_path in src_train_imgs:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        ext = os.path.splitext(img_path)[1]
        lbl_path = os.path.join(src_dataset, 'train', 'labels', f"{base_name}.txt")
        
        if not os.path.exists(lbl_path):
            continue
            
        # Copy original image and label
        dst_img_orig = os.path.join(dst_dataset, 'train', 'images', f"{base_name}{ext}")
        dst_lbl_orig = os.path.join(dst_dataset, 'train', 'labels', f"{base_name}.txt")
        shutil.copy2(img_path, dst_img_orig)
        shutil.copy2(lbl_path, dst_lbl_orig)
        
        # Read boxes
        boxes = []
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:])
                    boxes.append((cls_id, x, y, w, h))
                    class_counts_before[cls_id] += 1
                    class_counts_after[cls_id] += 1
                    
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        cls_ids = [b[0] for b in boxes]
        
        # Determine augmentation multiplier based on minority classes present
        # Class 0: bicycle (severely underrepresented)
        # Class 4: truck, Class 1: bus, Class 2: car
        has_bicycle = (0 in cls_ids)
        has_truck = (4 in cls_ids)
        has_bus = (1 in cls_ids)
        has_car = (2 in cls_ids)
        
        aug_specs = [] # list of tuples: (suffix, do_flip, photo_mode)
        
        if has_bicycle:
            # 4 augmentations for bicycle images
            aug_specs = [
                ('_aug_flip', True, 0),
                ('_aug_bright', False, 1),
                ('_aug_flip_shadow', True, 2),
                ('_aug_contrast_hue', False, 3)
            ]
        elif has_truck or has_bus:
            # 2 augmentations for truck/bus images
            aug_specs = [
                ('_aug_flip', True, 0),
                ('_aug_bright', False, 1)
            ]
        elif has_car:
            # 1 augmentation for car images
            aug_specs = [
                ('_aug_flip', True, 0)
            ]
        elif np.random.rand() < 0.3:
            # 30% chance for motorcycle-only images
            aug_specs = [
                ('_aug_flip', True, 0)
            ]
            
        for suffix, do_flip, photo_mode in aug_specs:
            aug_img = img.copy()
            aug_boxes = list(boxes)
            
            if do_flip:
                aug_img = cv2.flip(aug_img, 1)
                aug_boxes = flip_yolo_boxes(aug_boxes)
                
            if photo_mode > 0:
                aug_img = apply_photometric_aug(aug_img, photo_mode)
                
            aug_img_name = f"{base_name}{suffix}{ext}"
            aug_lbl_name = f"{base_name}{suffix}.txt"
            
            aug_img_path = os.path.join(dst_dataset, 'train', 'images', aug_img_name)
            aug_lbl_path = os.path.join(dst_dataset, 'train', 'labels', aug_lbl_name)
            
            cv2.imwrite(aug_img_path, aug_img)
            with open(aug_lbl_path, 'w') as f:
                for cls_id, x, y, w, h in aug_boxes:
                    f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                    class_counts_after[cls_id] += 1
                    
            augmented_img_count += 1
            
    # Create data.yaml for augmented dataset
    yaml_content = f"""train: d:/Model Imrovement/Vehicle-counting-augmented/train/images
val: d:/Model Imrovement/Vehicle-counting-augmented/valid/images
test: d:/Model Imrovement/Vehicle-counting-augmented/test/images

nc: 5
names: ['bicycle', 'bus', 'car', 'motorcycle', 'truck']
"""
    yaml_path = os.path.join(dst_dataset, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
        
    print(f"=== Dataset Augmentation Complete ===")
    print(f"Generated {augmented_img_count} new augmented images.")
    print("Class distribution BEFORE augmentation:")
    for cls_id, name in enumerate(['bicycle', 'bus', 'car', 'motorcycle', 'truck']):
        print(f"  Class {cls_id} ({name}): {class_counts_before[cls_id]} instances")
    print("Class distribution AFTER augmentation:")
    for cls_id, name in enumerate(['bicycle', 'bus', 'car', 'motorcycle', 'truck']):
        print(f"  Class {cls_id} ({name}): {class_counts_after[cls_id]} instances")

if __name__ == '__main__':
    generate_augmented_dataset()
