"""
YOLOv8 Custom Model Training Script
Train a YOLOv8 model on custom drone detection dataset
"""

# IMPORTANT: torch/YOLO MUST be imported before sklearn to avoid
# an OpenMP DLL conflict on Windows ([WinError 1114]).
try:
    import torch
    _original_load = torch.load
    def _patched_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return _original_load(*args, **kwargs)
    torch.load = _patched_load
except Exception:
    pass

from ultralytics import YOLO

import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml

# Configuration
DATASET_ROOT = Path(r"c:\PINKY\drone detection\captured_datasets")
DETECTED_OBJECTS_DIR = DATASET_ROOT / "detected_objects"
BACKGROUND_FRAMES_DIR = DATASET_ROOT / "background_frames"
TRAINING_DATA_DIR = DATASET_ROOT / "training_data"
YAML_CONFIG = DATASET_ROOT / "dataset_custom.yaml"

# Training parameters
TRAIN_SIZE = 0.7  # 70% training
VAL_SIZE = 0.2    # 20% validation
TEST_SIZE = 0.1   # 10% testing
RANDOM_SEED = 42
EPOCHS = 30
BATCH_SIZE = 16
IMG_SIZE = 640
PATIENCE = 20  # Early stopping patience


def create_training_directories():
    """Create directory structure for training data"""
    print("Creating training directories...")
    
    dirs_to_create = [
        TRAINING_DATA_DIR / "images" / "train",
        TRAINING_DATA_DIR / "images" / "val",
        TRAINING_DATA_DIR / "images" / "test",
        TRAINING_DATA_DIR / "labels" / "train",
        TRAINING_DATA_DIR / "labels" / "val",
        TRAINING_DATA_DIR / "labels" / "test",
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {dir_path}")


def prepare_dataset():
    """Prepare dataset by splitting into train/val/test"""
    print("\n" + "="*60)
    print("PREPARING DATASET")
    print("="*60)
    
    # Collect all image/label pairs
    all_images = []
    
    # Add detected objects
    detected_images = sorted(
        (DETECTED_OBJECTS_DIR / "frames").glob("*.jpg")
    )
    for img_path in detected_images:
        label_path = DETECTED_OBJECTS_DIR / "labels" / f"{img_path.stem}.txt"
        if label_path.exists():
            all_images.append((img_path, label_path, "objects"))
    
    # Add background frames
    background_images = sorted(
        (BACKGROUND_FRAMES_DIR / "frames").glob("*.jpg")
    )
    for img_path in background_images:
        label_path = BACKGROUND_FRAMES_DIR / "labels" / f"{img_path.stem}.txt"
        if label_path.exists():
            all_images.append((img_path, label_path, "background"))
    
    print(f"\nTotal images found: {len(all_images)}")
    print(f"  - Detected objects: {len([x for x in all_images if x[2] == 'objects'])}")
    print(f"  - Background frames: {len([x for x in all_images if x[2] == 'background'])}")
    
    if not all_images:
        raise ValueError("No image/label pairs found in dataset!")
    
    # Split dataset
    random.seed(RANDOM_SEED)
    train_val, test = train_test_split(
        all_images, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    train, val = train_test_split(
        train_val, test_size=VAL_SIZE/(TRAIN_SIZE + VAL_SIZE), random_state=RANDOM_SEED
    )
    
    print(f"\nDataset split:")
    print(f"  - Training: {len(train)} images ({len(train)/len(all_images)*100:.1f}%)")
    print(f"  - Validation: {len(val)} images ({len(val)/len(all_images)*100:.1f}%)")
    print(f"  - Testing: {len(test)} images ({len(test)/len(all_images)*100:.1f}%)")
    
    # Copy files to training directories
    def copy_split(split_images, split_name):
        print(f"\nCopying {split_name} images...")
        for img_path, label_path, category in split_images:
            # Copy image
            dest_img = TRAINING_DATA_DIR / "images" / split_name / img_path.name
            shutil.copy2(img_path, dest_img)

            # Copy label (convert format if from detected_objects)
            dest_label = TRAINING_DATA_DIR / "labels" / split_name / f"{img_path.stem}.txt"

            if category == "objects":
                # Convert custom pixel-coord format to normalized YOLO format
                convert_custom_label_to_yolo(label_path, dest_label, img_path=img_path)
            else:
                # Background: already in YOLO normalized format
                shutil.copy2(label_path, dest_label)

        print(f"✓ Copied {len(split_images)} {split_name} samples")
    
    copy_split(train, "train")
    copy_split(val, "val")
    copy_split(test, "test")
    
    print("\n✓ Dataset preparation complete!")


def convert_custom_label_to_yolo(src_label, dest_label, img_path=None):
    """
    Convert custom label format to standard YOLO format.
    Custom format: class_id cx cy w h class_name confidence  (ALL in absolute pixels)
    YOLO format:   class_id cx cy w h                        (ALL normalized 0-1)
    """
    # Determine image dimensions for normalization
    img_w, img_h = 1.0, 1.0  # fallback: no normalization needed if already 0-1
    if img_path is not None:
        try:
            import cv2
            frame = cv2.imread(str(img_path))
            if frame is not None:
                img_h, img_w = frame.shape[:2]
        except Exception:
            pass

    with open(src_label, 'r') as f:
        lines = f.readlines()

    class_mapping = {
        "drone": 1,
        "airplane": 2,
        "helicopter": 3,
        "bird": 4,
        "background": 0,
    }

    with open(dest_label, 'w') as f:
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            try:
                cx = float(parts[1])
                cy = float(parts[2])
                bw = float(parts[3])
                bh = float(parts[4])

                # If values are > 1 they are in absolute pixels — normalize them
                if cx > 1.0 or cy > 1.0 or bw > 1.0 or bh > 1.0:
                    cx /= img_w
                    cy /= img_h
                    bw /= img_w
                    bh /= img_h

                # Clamp to valid range
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                bw = max(0.0, min(1.0, bw))
                bh = max(0.0, min(1.0, bh))

                # Determine class id
                class_name = parts[5].lower() if len(parts) > 5 else "drone"
                yolo_class_id = class_mapping.get(class_name, 1)

                f.write(f"{yolo_class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            except (ValueError, IndexError):
                continue


def create_yaml_config():
    """Create YAML configuration file for training"""
    print("\nCreating YAML configuration...")
    
    config = {
        'path': str(TRAINING_DATA_DIR),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 5,
        'names': {
            0: 'background',
            1: 'drone',
            2: 'airplane',
            3: 'helicopter',
            4: 'bird'
        }
    }
    
    with open(YAML_CONFIG, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✓ YAML config created: {YAML_CONFIG}")


def train_model():
    """Train YOLOv8 model"""
    print("\n" + "="*60)
    print("TRAINING YOLOV8 MODEL")
    print("="*60)
    
    # Load the base model
    print("\nLoading YOLOv8 nano model...")
    model = YOLO("yolov8n.pt")
    
    # Train the model
    print(f"\nStarting training...")
    print(f"  - Epochs: {EPOCHS}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  - Patience (early stopping): {PATIENCE}")
    
    results = model.train(
        data=str(YAML_CONFIG),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        device='cpu',  # GPU device, change to 0 if GPU available, or 'cpu' for CPU only
        project=str(DATASET_ROOT / "training_results"),
        name="custom_drone_detector",
        exist_ok=True,
        verbose=True,
        save=True,
        pretrained=True,
    )
    
    return model, results


def evaluate_model(model):
    """Evaluate model on test set"""
    print("\n" + "="*60)
    print("EVALUATING MODEL")
    print("="*60)
    
    # Validate on test set
    metrics = model.val(
        data=str(YAML_CONFIG),
        split='test',
    )
    
    print("\nValidation Results:")
    print(f"  - mAP@50: {metrics.box.map50:.4f}")
    print(f"  - mAP@50-95: {metrics.box.map:.4f}")
    
    return metrics


def test_inference(model):
    """Test inference on sample images"""
    print("\n" + "="*60)
    print("TESTING INFERENCE")
    print("="*60)
    
    test_images_dir = TRAINING_DATA_DIR / "images" / "test"
    test_images = list(test_images_dir.glob("*.jpg"))[:3]  # Test on first 3 images
    
    if test_images:
        print(f"\nRunning inference on {len(test_images)} test images...")
        results = model.predict(
            source=test_images,
            conf=0.25,
            iou=0.45,
            device='cpu',
        )
        
        print("✓ Inference complete!")
        for result in results:
            print(f"\n  Image: {result.path}")
            print(f"  Detections: {len(result.boxes)}")
            for box in result.boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                class_name = ["background", "drone"][class_id]
                print(f"    - {class_name}: {confidence:.2%}")


def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("YOLOV8 CUSTOM DRONE DETECTION TRAINING")
    print("="*60)
    
    try:
        # Step 1: Create directories
        create_training_directories()
        
        # Step 2: Prepare dataset
        prepare_dataset()
        
        # Step 3: Create YAML config
        create_yaml_config()
        
        # Step 4: Train model
        model, results = train_model()
        
        # Step 5: Evaluate model
        evaluate_model(model)
        
        # Step 6: Test inference
        test_inference(model)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"\nModel saved to: {DATASET_ROOT / 'training_results/custom_drone_detector'}")
        print("\nTo use the trained model in your detection script:")
        print("  model = YOLO('path/to/training_results/custom_drone_detector/weights/best.pt')")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
