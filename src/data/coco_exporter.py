"""
COCO format exporter.

Exports internal dataset representation to COCO JSON format
compatible with Ultralytics YOLO.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import shutil
import logging

from .schemas import Dataset, DatasetSplits, ImageAnnotation
from ..utils import normalize_path, ensure_dir, get_relative_path

logger = logging.getLogger(__name__)


class COCOExporter:
    """
    Export datasets to COCO format.
    
    Creates:
    - COCO JSON files for train/val/test splits
    - Ultralytics data YAML file
    - Organized image directories
    """
    
    def __init__(self, output_dir: Path, copy_images: bool = True):
        """
        Initialize exporter.
        
        Args:
            output_dir: Output directory for exported dataset
            copy_images: Whether to copy images or use symlinks/references
        """
        self.output_dir = normalize_path(output_dir)
        self.copy_images = copy_images
        
        # Create directory structure
        self.images_dir = self.output_dir / 'images'
        self.labels_dir = self.output_dir / 'labels'
        
    def export(self, dataset: Dataset, splits: DatasetSplits,
              run_id: str = 'dataset') -> Dict[str, Path]:
        """
        Export dataset with splits to COCO format.
        
        Args:
            dataset: Complete dataset
            splits: Train/val/test split assignments
            run_id: Dataset identifier
            
        Returns:
            Dictionary mapping split names to COCO JSON file paths
        """
        logger.info(f"Exporting dataset to COCO format at {self.output_dir}")
        
        # Create directories
        ensure_dir(self.output_dir)
        ensure_dir(self.images_dir)
        
        # Create ID to image mapping
        id_to_img = {img.image_id: img for img in dataset.images}
        
        # Export each split
        coco_files = {}
        
        for split_name in ['train', 'val', 'test']:
            split = splits.get_split(split_name)
            
            # Get images for this split
            split_images = [id_to_img[img_id] for img_id in split.image_ids 
                          if img_id in id_to_img]
            
            if not split_images:
                logger.warning(f"No images in {split_name} split, skipping")
                continue
            
            # Create COCO JSON
            coco_json = self._create_coco_json(
                split_images, 
                dataset.classes,
                split_name
            )
            
            # Save JSON file
            json_path = self.output_dir / f'coco.{split_name}.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(coco_json, f, indent=2)
            
            coco_files[split_name] = json_path
            logger.info(f"Exported {len(split_images)} images to {json_path}")
            
            # Organize images (only if copying)
            if self.copy_images:
                self._organize_images(split_images, split_name)
            else:
                # Create image path list file for Ultralytics
                self._create_image_list(split_images, split_name)
        
        # Create Ultralytics data YAML
        data_yaml = self._create_data_yaml(dataset.classes, run_id)
        yaml_path = self.output_dir / 'coco.yaml'
        
        from ..utils import save_yaml
        save_yaml(data_yaml, yaml_path)
        
        logger.info(f"Created Ultralytics data YAML: {yaml_path}")
        
        # Create summary file
        self._create_summary(dataset, splits)
        
        return coco_files
    
    def _create_coco_json(self, images: List[ImageAnnotation],
                         classes: List[str], split_name: str) -> Dict[str, Any]:
        """
        Create COCO format JSON for a set of images.
        
        Args:
            images: List of image annotations
            classes: List of class names
            split_name: Split name (train/val/test)
            
        Returns:
            COCO format dictionary
        """
        coco = {
            'info': {
                'description': f'Dice Detection Dataset - {split_name}',
                'version': '1.0',
                'year': datetime.now().year,
                'contributor': 'DINOv3 Experiment',
                'date_created': datetime.now().isoformat(),
            },
            'licenses': [],
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        # Add categories
        for idx, class_name in enumerate(classes):
            coco['categories'].append({
                'id': idx,
                'name': class_name,
                'supercategory': 'object'
            })
        
        # Add images and annotations
        annotation_id = 0
        
        for img_idx, img_ann in enumerate(images):
            # Add image info
            # Use absolute path if not copying images, otherwise relative path
            if self.copy_images:
                file_path = f"{split_name}/{img_ann.image_path.name}"
            else:
                # Use absolute path to original image
                file_path = str(img_ann.image_path.resolve())
            
            # Get actual image dimensions if not available
            width = img_ann.width
            height = img_ann.height
            
            if width == 0 or height == 0:
                try:
                    from PIL import Image
                    with Image.open(img_ann.image_path) as img:
                        width, height = img.size
                except Exception as e:
                    logger.warning(f"Could not get image dimensions for {img_ann.image_path}: {e}")
                    width, height = 640, 640  # Default fallback
            
            image_info = {
                'id': img_idx,
                'file_name': file_path,
                'width': width,
                'height': height,
            }
            coco['images'].append(image_info)
            
            # Add annotations
            for ann in img_ann.annotations:
                bbox = ann.bbox.to_xywh()
                
                coco_ann = {
                    'id': annotation_id,
                    'image_id': img_idx,
                    'category_id': ann.class_id,
                    'bbox': list(bbox),
                    'area': ann.area or ann.bbox.area(),
                    'iscrowd': 1 if ann.is_crowd else 0,
                    'segmentation': []  # No segmentation for detection
                }
                
                coco['annotations'].append(coco_ann)
                annotation_id += 1
        
        logger.info(f"Created COCO JSON with {len(coco['images'])} images "
                   f"and {len(coco['annotations'])} annotations")
        
        return coco
    
    def _organize_images(self, images: List[ImageAnnotation], 
                        split_name: str) -> None:
        """
        Organize images into split-specific directories.
        
        Args:
            images: List of images
            split_name: Split name
        """
        split_dir = self.images_dir / split_name
        ensure_dir(split_dir)
        
        logger.info(f"Organizing {len(images)} images for {split_name} split")
        
        for img_ann in images:
            src_path = img_ann.image_path
            dst_path = split_dir / src_path.name
            
            # Handle name collisions
            counter = 1
            while dst_path.exists() and not self._are_same_file(src_path, dst_path):
                stem = src_path.stem
                suffix = src_path.suffix
                dst_path = split_dir / f"{stem}_{counter}{suffix}"
                counter += 1
            
            # Copy or link image
            if not dst_path.exists():
                try:
                    if self.copy_images:
                        shutil.copy2(src_path, dst_path)
                    else:
                        # Try to create symlink, fallback to copy
                        try:
                            dst_path.symlink_to(src_path.resolve())
                        except (OSError, NotImplementedError):
                            shutil.copy2(src_path, dst_path)
                except Exception as e:
                    logger.error(f"Failed to copy/link {src_path} to {dst_path}: {e}")
    
    def _are_same_file(self, path1: Path, path2: Path) -> bool:
        """Check if two paths point to the same file."""
        try:
            return path1.resolve() == path2.resolve()
        except Exception:
            return False
    
    def _create_image_list(self, images: List[ImageAnnotation], 
                          split_name: str) -> None:
        """
        Create text file with image paths and YOLO format labels for Ultralytics.
        
        When not copying images, this creates:
        - {split}.txt with image paths
        - Label files in parallel to images (images/ -> labels/)
        
        Args:
            images: List of images
            split_name: Split name (train/val/test)
        """
        # Create image list file
        list_path = self.output_dir / f'{split_name}.txt'
        
        logger.info(f"Creating image list and labels for {split_name}")
        
        label_dirs_created = set()
        
        with open(list_path, 'w', encoding='utf-8') as f:
            for img_ann in images:
                # Write absolute path to image
                img_path = img_ann.image_path.resolve()
                f.write(f"{str(img_path)}\n")
                
                # Create label path by replacing 'images' with 'labels'
                # and changing extension to .txt
                label_path = self._get_label_path(img_path)
                
                # Ensure label directory exists
                label_dir = label_path.parent
                if label_dir not in label_dirs_created:
                    ensure_dir(label_dir)
                    label_dirs_created.add(label_dir)
                
                self._write_yolo_label(img_ann, label_path)
        
        logger.info(f"Wrote {len(images)} image paths to {list_path}")
        logger.info(f"Created labels in {len(label_dirs_created)} directories")
    
    def _get_label_path(self, image_path: Path) -> Path:
        """
        Get label path for an image by replacing 'images' with 'labels'.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Path to label file
        """
        # Convert path to string for manipulation
        path_str = str(image_path)
        
        # Replace 'images' with 'labels' in the path
        if '\\images\\' in path_str:
            label_str = path_str.replace('\\images\\', '\\labels\\')
        elif '/images/' in path_str:
            label_str = path_str.replace('/images/', '/labels/')
        else:
            # Fallback: put labels next to images
            label_str = str(image_path.parent / 'labels' / image_path.name)
        
        # Change extension to .txt
        label_path = Path(label_str).with_suffix('.txt')
        
        return label_path
    
    def _write_yolo_label(self, img_ann: ImageAnnotation, label_path: Path) -> None:
        """
        Write YOLO format label file.
        
        Format: class_id center_x center_y width height (normalized to [0, 1])
        
        Args:
            img_ann: Image annotation
            label_path: Path to save label file
        """
        with open(label_path, 'w') as f:
            for ann in img_ann.annotations:
                # Get bbox in center format
                cx, cy, w, h = ann.bbox.to_cxcywh()
                
                # Normalize to [0, 1]
                if img_ann.width > 0 and img_ann.height > 0:
                    cx_norm = cx / img_ann.width
                    cy_norm = cy / img_ann.height
                    w_norm = w / img_ann.width
                    h_norm = h / img_ann.height
                else:
                    # Fallback if dimensions unknown
                    logger.warning(f"Unknown dimensions for {img_ann.image_path}, skipping label")
                    continue
                
                # Clip to [0, 1] range (handle any edge cases)
                cx_norm = max(0.0, min(1.0, cx_norm))
                cy_norm = max(0.0, min(1.0, cy_norm))
                w_norm = max(0.0, min(1.0, w_norm))
                h_norm = max(0.0, min(1.0, h_norm))
                
                # Write line: class_id cx cy w h
                f.write(f"{ann.class_id} {cx_norm:.6f} {cy_norm:.6f} "
                       f"{w_norm:.6f} {h_norm:.6f}\n")
    
    def _create_data_yaml(self, classes: List[str], run_id: str) -> Dict[str, Any]:
        """
        Create Ultralytics data YAML configuration.
        
        Args:
            classes: List of class names
            run_id: Dataset identifier
            
        Returns:
            Data YAML dictionary
        """
        if self.copy_images:
            # Use relative paths when images are copied
            data_yaml = {
                'path': str(self.output_dir.resolve()),
                'train': 'images/train',
                'val': 'images/val',
                'test': 'images/test',
                'names': {idx: name for idx, name in enumerate(classes)},
                'nc': len(classes),
                'run_id': run_id,
            }
        else:
            # When not copying images, use image list with labels in output dir
            # Create parallel label directory structure
            data_yaml = {
                'path': str(self.output_dir.resolve()),
                'train': 'train.txt',  # File with image paths
                'val': 'val.txt',
                'test': 'test.txt',
                'names': {idx: name for idx, name in enumerate(classes)},
                'nc': len(classes),
                'run_id': run_id,
            }
        
        return data_yaml
    
    def _create_summary(self, dataset: Dataset, splits: DatasetSplits) -> None:
        """
        Create a summary file with dataset statistics.
        
        Args:
            dataset: Complete dataset
            splits: Split assignments
        """
        stats = dataset.get_statistics()
        split_stats = splits.get_statistics()
        
        summary = {
            'dataset': stats,
            'splits': split_stats,
            'export_info': {
                'output_dir': str(self.output_dir),
                'export_time': datetime.now().isoformat(),
                'copy_images': self.copy_images,
            }
        }
        
        summary_path = self.output_dir / 'summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Created dataset summary: {summary_path}")


class UltralyticsExporter:
    """
    Export to Ultralytics native format (labels in txt files).
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize exporter.
        
        Args:
            output_dir: Output directory
        """
        self.output_dir = normalize_path(output_dir)
        self.images_dir = self.output_dir / 'images'
        self.labels_dir = self.output_dir / 'labels'
    
    def export(self, dataset: Dataset, splits: DatasetSplits) -> Path:
        """
        Export dataset in Ultralytics format.
        
        Creates txt label files with format:
        class_id center_x center_y width height (normalized to [0, 1])
        
        Args:
            dataset: Complete dataset
            splits: Split assignments
            
        Returns:
            Path to data YAML file
        """
        logger.info(f"Exporting dataset to Ultralytics format at {self.output_dir}")
        
        ensure_dir(self.output_dir)
        
        # Create ID to image mapping
        id_to_img = {img.image_id: img for img in dataset.images}
        
        # Export each split
        for split_name in ['train', 'val', 'test']:
            split = splits.get_split(split_name)
            
            # Create directories
            img_dir = self.images_dir / split_name
            lbl_dir = self.labels_dir / split_name
            ensure_dir(img_dir)
            ensure_dir(lbl_dir)
            
            # Process images
            for img_id in split.image_ids:
                if img_id not in id_to_img:
                    continue
                
                img_ann = id_to_img[img_id]
                
                # Copy image
                dst_img = img_dir / img_ann.image_path.name
                if not dst_img.exists():
                    shutil.copy2(img_ann.image_path, dst_img)
                
                # Create label file
                lbl_path = lbl_dir / f"{img_ann.image_path.stem}.txt"
                self._write_label_file(lbl_path, img_ann)
        
        # Create data YAML
        data_yaml = {
            'path': str(self.output_dir.resolve()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': {idx: name for idx, name in enumerate(dataset.classes)},
            'nc': len(dataset.classes),
        }
        
        yaml_path = self.output_dir / 'data.yaml'
        from ..utils import save_yaml
        save_yaml(data_yaml, yaml_path)
        
        logger.info(f"Created Ultralytics data YAML: {yaml_path}")
        
        return yaml_path
    
    def _write_label_file(self, label_path: Path, img_ann: ImageAnnotation) -> None:
        """
        Write label file in YOLO format.
        
        Format: class_id center_x center_y width height (normalized)
        """
        with open(label_path, 'w') as f:
            for ann in img_ann.annotations:
                # Get bbox in center format
                cx, cy, w, h = ann.bbox.to_cxcywh()
                
                # Normalize to [0, 1]
                if img_ann.width > 0 and img_ann.height > 0:
                    cx_norm = cx / img_ann.width
                    cy_norm = cy / img_ann.height
                    w_norm = w / img_ann.width
                    h_norm = h / img_ann.height
                else:
                    # Fallback if dimensions unknown
                    cx_norm, cy_norm, w_norm, h_norm = 0.5, 0.5, 0.1, 0.1
                
                # Write line: class_id cx cy w h
                f.write(f"{ann.class_id} {cx_norm:.6f} {cy_norm:.6f} "
                       f"{w_norm:.6f} {h_norm:.6f}\n")

