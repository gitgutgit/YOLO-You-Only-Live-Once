"""
Data Augmentation Pipeline for Game Frame Robustness

Author: Minsuk Kim (mk4434)
Purpose: Scale ~1k labeled frames to ~5k effective samples through automated augmentation

Key augmentations for game environment robustness:
- Background randomization (sky, cave textures)
- Motion blur / Gaussian noise (speed simulation)
- Scaling, rotation, hue/saturation jitter
- Partial occlusion / lighting changes
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Tuple, Optional
import random
from pathlib import Path


class GameFrameAugmenter:
    """
    Augmentation pipeline specifically designed for 2D game frames.
    
    Focuses on variations that maintain game semantics while improving
    detector robustness to visual conditions.
    """
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (640, 480),
                 augmentation_factor: int = 5):
        """
        Initialize augmentation pipeline.
        
        Args:
            target_size: Output frame dimensions (width, height)
            augmentation_factor: How many augmented versions per original frame
        """
        self.target_size = target_size
        self.augmentation_factor = augmentation_factor
        
        # Background textures for randomization
        self.background_textures = []
        
        # Define augmentation pipeline
        self.transform = A.Compose([
            # Geometric transformations (preserve bounding boxes)
            A.HorizontalFlip(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.2, 
                rotate_limit=5, 
                p=0.5
            ),
            
            # Color/lighting variations
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.6
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),
            
            # Noise and blur (simulate motion/uncertainty)
            A.OneOf([
                A.MotionBlur(blur_limit=7, p=0.5),
                A.GaussianBlur(blur_limit=5, p=0.3),
                A.MedianBlur(blur_limit=3, p=0.2)
            ], p=0.4),
            
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            
            # Partial occlusion simulation
            A.CoarseDropout(
                max_holes=3,
                max_height=50,
                max_width=50,
                min_holes=1,
                min_height=20,
                min_width=20,
                p=0.2
            ),
            
            # Ensure consistent output size
            A.Resize(height=target_size[1], width=target_size[0]),
            
        ], bbox_params=A.BboxParams(
            format='yolo',  # Normalized [x_center, y_center, width, height]
            label_fields=['class_labels'],
            min_visibility=0.3  # Keep boxes with >30% visibility
        ))
    
    def load_background_textures(self, texture_dir: Path) -> None:
        """
        Load background texture images for randomization.
        
        Args:
            texture_dir: Directory containing background texture images
        """
        if not texture_dir.exists():
            print(f"Warning: Texture directory {texture_dir} not found")
            return
            
        texture_files = list(texture_dir.glob("*.jpg")) + list(texture_dir.glob("*.png"))
        
        for texture_file in texture_files:
            texture = cv2.imread(str(texture_file))
            if texture is not None:
                # Resize to match target dimensions
                texture = cv2.resize(texture, self.target_size)
                self.background_textures.append(texture)
        
        print(f"Loaded {len(self.background_textures)} background textures")
    
    def randomize_background(self, 
                           frame: np.ndarray, 
                           bboxes: List[List[float]],
                           class_labels: List[int]) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """
        Replace background with random texture while preserving game objects.
        
        This is a simplified version - in practice, you'd need semantic segmentation
        or more sophisticated background/foreground separation.
        
        Args:
            frame: Input game frame
            bboxes: Bounding boxes in YOLO format
            class_labels: Class labels for each bbox
            
        Returns:
            Augmented frame with new background
        """
        if not self.background_textures:
            return frame, bboxes, class_labels
        
        # For now, just blend with random texture (simple approach)
        texture = random.choice(self.background_textures)
        alpha = random.uniform(0.1, 0.3)  # Light blending to preserve game objects
        
        augmented_frame = cv2.addWeighted(frame, 1-alpha, texture, alpha, 0)
        
        return augmented_frame, bboxes, class_labels
    
    def augment_frame(self, 
                     frame: np.ndarray,
                     bboxes: List[List[float]] = None,
                     class_labels: List[int] = None) -> Dict:
        """
        Apply augmentation pipeline to a single frame.
        
        Args:
            frame: Input frame as numpy array (H, W, C)
            bboxes: List of bounding boxes in YOLO format [x_center, y_center, w, h]
            class_labels: List of class labels corresponding to bboxes
            
        Returns:
            Dictionary with augmented frame and transformed annotations
        """
        if bboxes is None:
            bboxes = []
        if class_labels is None:
            class_labels = []
        
        # Background randomization (if textures available)
        if random.random() < 0.3 and self.background_textures:
            frame, bboxes, class_labels = self.randomize_background(
                frame, bboxes, class_labels
            )
        
        # Apply albumentations pipeline
        try:
            transformed = self.transform(
                image=frame,
                bboxes=bboxes,
                class_labels=class_labels
            )
            
            return {
                'image': transformed['image'],
                'bboxes': transformed['bboxes'],
                'class_labels': transformed['class_labels']
            }
            
        except Exception as e:
            print(f"Augmentation failed: {e}")
            # Return original if augmentation fails
            resized_frame = cv2.resize(frame, self.target_size)
            return {
                'image': resized_frame,
                'bboxes': bboxes,
                'class_labels': class_labels
            }
    
    def augment_dataset(self, 
                       input_dir: Path,
                       output_dir: Path,
                       annotation_format: str = 'yolo') -> None:
        """
        Augment entire dataset from input directory to output directory.
        
        Args:
            input_dir: Directory with original images and annotations
            output_dir: Directory to save augmented data
            annotation_format: Format of annotations ('yolo', 'coco', etc.)
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(input_dir.glob(ext))
        
        print(f"Found {len(image_files)} images to augment")
        
        for img_path in image_files:
            # Load image
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            
            # Load corresponding annotation file
            ann_path = img_path.with_suffix('.txt')
            bboxes, class_labels = self._load_yolo_annotations(ann_path)
            
            # Generate multiple augmented versions
            for i in range(self.augmentation_factor):
                augmented = self.augment_frame(frame, bboxes, class_labels)
                
                # Save augmented image
                output_name = f"{img_path.stem}_aug_{i:02d}{img_path.suffix}"
                output_img_path = output_dir / output_name
                cv2.imwrite(str(output_img_path), augmented['image'])
                
                # Save augmented annotations
                output_ann_path = output_img_path.with_suffix('.txt')
                self._save_yolo_annotations(
                    output_ann_path, 
                    augmented['bboxes'], 
                    augmented['class_labels']
                )
        
        print(f"Augmentation complete. Generated {len(image_files) * self.augmentation_factor} samples")
    
    def _load_yolo_annotations(self, ann_path: Path) -> Tuple[List[List[float]], List[int]]:
        """Load YOLO format annotations from file."""
        bboxes = []
        class_labels = []
        
        if not ann_path.exists():
            return bboxes, class_labels
        
        with open(ann_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    bbox = [float(x) for x in parts[1:5]]
                    class_labels.append(class_id)
                    bboxes.append(bbox)
        
        return bboxes, class_labels
    
    def _save_yolo_annotations(self, 
                              ann_path: Path, 
                              bboxes: List[List[float]], 
                              class_labels: List[int]) -> None:
        """Save YOLO format annotations to file."""
        with open(ann_path, 'w') as f:
            for bbox, class_id in zip(bboxes, class_labels):
                line = f"{class_id} {' '.join(map(str, bbox))}\n"
                f.write(line)


def create_background_textures(output_dir: Path, num_textures: int = 10) -> None:
    """
    Generate synthetic background textures for augmentation.
    
    Creates simple gradient and noise patterns as placeholder backgrounds.
    In practice, you'd collect real sky/cave/environment textures.
    
    Args:
        output_dir: Directory to save generated textures
        num_textures: Number of textures to generate
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(num_textures):
        # Generate random gradient background
        height, width = 480, 640
        
        # Random colors for gradient
        color1 = np.random.randint(0, 255, 3)
        color2 = np.random.randint(0, 255, 3)
        
        # Create gradient
        texture = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            alpha = y / height
            blended_color = (1 - alpha) * color1 + alpha * color2
            texture[y, :] = blended_color.astype(np.uint8)
        
        # Add some noise
        noise = np.random.normal(0, 10, (height, width, 3))
        texture = np.clip(texture + noise, 0, 255).astype(np.uint8)
        
        # Save texture
        texture_path = output_dir / f"texture_{i:02d}.jpg"
        cv2.imwrite(str(texture_path), texture)
    
    print(f"Generated {num_textures} background textures in {output_dir}")


if __name__ == "__main__":
    # Example usage
    augmenter = GameFrameAugmenter(augmentation_factor=5)
    
    # Generate some background textures
    texture_dir = Path("data/textures")
    create_background_textures(texture_dir, num_textures=10)
    augmenter.load_background_textures(texture_dir)
    
    print("Data augmentation pipeline ready!")
    print(f"Target size: {augmenter.target_size}")
    print(f"Augmentation factor: {augmenter.augmentation_factor}")
