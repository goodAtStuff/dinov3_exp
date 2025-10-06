"""
DataMuro format adapter.

Loads annotations from DataMuro format into internal schema.
DataMuro is assumed to store annotations in a structured format with bounding boxes.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from .schemas import (
    BoundingBox, Annotation, ImageAnnotation, Dataset
)
from .dice_classes import get_class_id, get_class_name, CLASS_NAMES
from ..utils import normalize_path, get_image_files

logger = logging.getLogger(__name__)


class DataMuroAdapter:
    """
    Adapter to load DataMuro annotations into internal schema.
    
    DataMuro format assumptions:
    - Annotations stored in JSON files (one per image or single manifest)
    - Bounding boxes in various formats (will normalize to xywh)
    - Class labels as strings or IDs
    """
    
    def __init__(self, class_names: List[str]):
        """
        Initialize adapter.
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names
        self.class_to_id = {name: idx for idx, name in enumerate(class_names)}
        
    def load_dataset(self, root_path: Path, 
                    annotation_file: Optional[Path] = None) -> Dataset:
        """
        Load a dataset from DataMuro format.
        
        Args:
            root_path: Root directory containing images and annotations
            annotation_file: Optional path to central annotation file
                           If None, will look for per-image annotation files
                           
        Returns:
            Dataset object
        """
        root_path = normalize_path(root_path)
        
        if not root_path.exists():
            raise ValueError(f"Dataset root does not exist: {root_path}")
        
        logger.info(f"Loading DataMuro dataset from {root_path}")
        
        dataset = Dataset(
            name=root_path.name,
            classes=self.class_names,
            metadata={'source': str(root_path)}
        )
        
        if annotation_file and annotation_file.exists():
            # Load from central annotation file
            self._load_from_central_file(root_path, annotation_file, dataset)
        else:
            # Look for per-image annotations or common annotation files
            self._load_from_directory(root_path, dataset)
        
        logger.info(f"Loaded {dataset.num_images()} images with "
                   f"{dataset.num_annotations()} annotations")
        
        return dataset
    
    def _load_from_central_file(self, root_path: Path, 
                               annotation_file: Path,
                               dataset: Dataset) -> None:
        """Load annotations from a central JSON file."""
        logger.info(f"Loading annotations from {annotation_file}")
        
        try:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load annotation file: {e}")
            raise
        
        # Try different DataMuro formats
        if 'items' in data:
            # Items-based format (your format!)
            self._parse_items_format(root_path, data, dataset)
        elif 'images' in data and 'annotations' in data:
            # COCO-like format
            self._parse_coco_format(root_path, data, dataset)
        elif 'frames' in data:
            # Video annotation format
            self._parse_frames_format(root_path, data, dataset)
        elif isinstance(data, list):
            # List of image annotations
            self._parse_list_format(root_path, data, dataset)
        else:
            logger.warning(f"Unknown annotation format in {annotation_file}")
    
    def _load_from_directory(self, root_path: Path, dataset: Dataset) -> None:
        """Load annotations from per-image files or discover structure."""
        # Look for annotations subdirectory first (common structure)
        annotations_dir = root_path / 'annotations'
        if annotations_dir.exists() and annotations_dir.is_dir():
            logger.info(f"Found annotations directory: {annotations_dir}")
            # Look for JSON files in annotations directory
            for ann_file in annotations_dir.glob('*.json'):
                logger.info(f"Loading annotations from: {ann_file}")
                self._load_from_central_file(root_path, ann_file, dataset)
            if dataset.num_images() > 0:
                return
        
        # Look for common annotation file names in root
        common_names = [
            'annotations.json',
            'labels.json',
            'datamuro.json',
            'manifest.json',
        ]
        
        for name in common_names:
            ann_file = root_path / name
            if ann_file.exists():
                logger.info(f"Found annotation file: {ann_file}")
                self._load_from_central_file(root_path, ann_file, dataset)
                return
        
        # Look for per-image annotation files
        image_files = get_image_files(root_path)
        logger.info(f"Looking for per-image annotations for {len(image_files)} images")
        
        for img_path in image_files:
            # Try common annotation file patterns
            ann_path = img_path.with_suffix('.json')
            if ann_path.exists():
                self._load_image_annotation(img_path, ann_path, dataset)
            else:
                # Add image without annotations (for unlabeled data)
                self._add_image_without_annotations(img_path, dataset)
        
        if dataset.num_images() == 0:
            logger.warning(f"No annotations found in {root_path}")
    
    def _parse_coco_format(self, root_path: Path, data: Dict, 
                          dataset: Dataset) -> None:
        """Parse COCO-like annotation format."""
        images_info = {img['id']: img for img in data.get('images', [])}
        
        # Group annotations by image
        image_annotations = {}
        for ann in data.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            image_annotations[img_id].append(ann)
        
        # Process each image
        for img_id, img_info in images_info.items():
            img_path = root_path / img_info['file_name']
            
            if not img_path.exists():
                logger.warning(f"Image file not found: {img_path}")
                continue
            
            image_ann = ImageAnnotation(
                image_path=img_path,
                image_id=str(img_id),
                width=img_info.get('width', 0),
                height=img_info.get('height', 0),
                metadata=img_info
            )
            
            # Add annotations
            for ann in image_annotations.get(img_id, []):
                try:
                    annotation = self._parse_annotation(ann)
                    image_ann.add_annotation(annotation)
                except Exception as e:
                    logger.warning(f"Failed to parse annotation: {e}")
            
            dataset.add_image(image_ann)
    
    def _parse_frames_format(self, root_path: Path, data: Dict,
                            dataset: Dataset) -> None:
        """Parse frame-based annotation format."""
        for frame in data.get('frames', []):
            img_name = frame.get('name') or frame.get('file_name')
            if not img_name:
                continue
            
            img_path = root_path / img_name
            if not img_path.exists():
                logger.warning(f"Image file not found: {img_path}")
                continue
            
            image_ann = ImageAnnotation(
                image_path=img_path,
                image_id=frame.get('id', img_name),
                width=frame.get('width', 0),
                height=frame.get('height', 0)
            )
            
            # Add objects
            for obj in frame.get('objects', []):
                try:
                    annotation = self._parse_annotation(obj)
                    image_ann.add_annotation(annotation)
                except Exception as e:
                    logger.warning(f"Failed to parse object: {e}")
            
            dataset.add_image(image_ann)
    
    def _parse_list_format(self, root_path: Path, data: List,
                          dataset: Dataset) -> None:
        """Parse list-based annotation format."""
        for item in data:
            img_name = item.get('image') or item.get('file_name')
            if not img_name:
                continue
            
            img_path = root_path / img_name
            if not img_path.exists():
                logger.warning(f"Image file not found: {img_path}")
                continue
            
            image_ann = ImageAnnotation(
                image_path=img_path,
                image_id=item.get('id', img_name),
                width=item.get('width', 0),
                height=item.get('height', 0)
            )
            
            # Add annotations
            for ann in item.get('annotations', []):
                try:
                    annotation = self._parse_annotation(ann)
                    image_ann.add_annotation(annotation)
                except Exception as e:
                    logger.warning(f"Failed to parse annotation: {e}")
            
            dataset.add_image(image_ann)
    
    def _parse_items_format(self, root_path: Path, data: Dict,
                           dataset: Dataset) -> None:
        """Parse items-based annotation format (common in DataMuro exports)."""
        items = data.get('items', [])
        
        # Look for images in common subdirectories
        images_dir = root_path / 'images'
        if images_dir.exists():
            # Try common subdirectory names
            for subdir_name in ['default', 'images', '']:
                search_dir = images_dir / subdir_name if subdir_name else images_dir
                if search_dir.exists():
                    break
        else:
            search_dir = root_path
        
        logger.info(f"Processing {len(items)} items from annotations")
        
        for item in items:
            # Get image info
            img_info = item.get('image', {})
            img_filename = img_info.get('path') or item.get('id')
            
            if not img_filename:
                continue
            
            # Find the actual image file
            img_path = self._find_image_file(root_path, img_filename, search_dir)
            if not img_path:
                logger.warning(f"Image file not found: {img_filename}")
                continue
            
            # Get image dimensions
            img_size = img_info.get('size', [0, 0])
            width = img_size[0] if len(img_size) > 0 else 0
            height = img_size[1] if len(img_size) > 1 else 0
            
            image_ann = ImageAnnotation(
                image_path=img_path,
                image_id=item.get('id', img_filename),
                width=width,
                height=height
            )
            
            # Add annotations
            annotations = item.get('annotations', [])
            for ann in annotations:
                try:
                    # Only process bbox type annotations
                    if ann.get('type') == 'bbox':
                        annotation = self._parse_annotation(ann)
                        image_ann.add_annotation(annotation)
                except Exception as e:
                    logger.warning(f"Failed to parse annotation for {img_filename}: {e}")
            
            dataset.add_image(image_ann)
    
    def _find_image_file(self, root_path: Path, filename: str, search_dir: Path) -> Optional[Path]:
        """Find image file in various possible locations."""
        # Try direct path
        img_path = search_dir / filename
        if img_path.exists():
            return img_path
        
        # Try in images/default
        img_path = root_path / 'images' / 'default' / filename
        if img_path.exists():
            return img_path
        
        # Try in images
        img_path = root_path / 'images' / filename
        if img_path.exists():
            return img_path
        
        # Try in root
        img_path = root_path / filename
        if img_path.exists():
            return img_path
        
        # Search recursively as last resort
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            stem = Path(filename).stem
            matches = list(root_path.rglob(f"{stem}{ext}"))
            if matches:
                return matches[0]
        
        return None
    
    def _parse_annotation(self, ann_data: Dict) -> Annotation:
        """Parse a single annotation from various formats."""
        # Extract die type and value from attributes (if present)
        attributes = ann_data.get('attributes', {})
        die_type = attributes.get('die type') or attributes.get('die_type')
        die_value = attributes.get('value')
        
        # Try to get class from die type + value
        if die_type and die_value is not None:
            class_id = get_class_id(die_type, die_value)
            class_name = get_class_name(die_type, die_value)
            
            if class_id is None or class_name is None:
                logger.warning(f"Invalid die type/value: {die_type}={die_value}")
                # Fallback to background class
                class_id = 0
                class_name = 'background'
        else:
            # Fallback: try to extract from other fields
            class_name = ann_data.get('class') or ann_data.get('category') or ann_data.get('label')
            class_id = ann_data.get('category_id')
            
            if class_id is None and class_name:
                class_id = self.class_to_id.get(class_name, 0)
            elif class_id is not None and not class_name:
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else 'unknown'
            
            if class_id is None:
                class_id = 0
                class_name = 'background'
        
        # Extract bounding box
        bbox = self._parse_bbox(ann_data)
        
        # Create annotation
        return Annotation(
            bbox=bbox,
            class_id=class_id,
            class_name=class_name,
            confidence=ann_data.get('score', 1.0),
            attributes=attributes,
            is_crowd=ann_data.get('iscrowd', False),
            area=ann_data.get('area')
        )
    
    def _parse_bbox(self, ann_data: Dict) -> BoundingBox:
        """Parse bounding box from various formats."""
        # Try direct bbox field
        if 'bbox' in ann_data:
            bbox_data = ann_data['bbox']
            if len(bbox_data) == 4:
                # Assume [x, y, width, height]
                return BoundingBox(x=bbox_data[0], y=bbox_data[1],
                                 width=bbox_data[2], height=bbox_data[3])
        
        # Try xyxy format
        if all(k in ann_data for k in ['x1', 'y1', 'x2', 'y2']):
            return BoundingBox.from_xyxy(
                ann_data['x1'], ann_data['y1'],
                ann_data['x2'], ann_data['y2']
            )
        
        # Try xywh format
        if all(k in ann_data for k in ['x', 'y', 'width', 'height']):
            return BoundingBox(
                x=ann_data['x'], y=ann_data['y'],
                width=ann_data['width'], height=ann_data['height']
            )
        
        # Try center format
        if all(k in ann_data for k in ['cx', 'cy', 'width', 'height']):
            return BoundingBox.from_cxcywh(
                ann_data['cx'], ann_data['cy'],
                ann_data['width'], ann_data['height']
            )
        
        raise ValueError(f"Could not parse bounding box from: {ann_data}")
    
    def _load_image_annotation(self, img_path: Path, ann_path: Path,
                               dataset: Dataset) -> None:
        """Load annotation for a single image."""
        try:
            with open(ann_path, 'r', encoding='utf-8') as f:
                ann_data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load annotation file {ann_path}: {e}")
            return
        
        # Get image dimensions (would need to actually load image)
        # For now, use dummy values or extract from annotation
        width = ann_data.get('image_width', 0)
        height = ann_data.get('image_height', 0)
        
        image_ann = ImageAnnotation(
            image_path=img_path,
            image_id=img_path.stem,
            width=width,
            height=height
        )
        
        # Parse annotations
        annotations = ann_data.get('annotations') or ann_data.get('objects') or []
        if isinstance(annotations, dict):
            annotations = [annotations]
        
        for ann in annotations:
            try:
                annotation = self._parse_annotation(ann)
                image_ann.add_annotation(annotation)
            except Exception as e:
                logger.warning(f"Failed to parse annotation: {e}")
        
        dataset.add_image(image_ann)
    
    def _add_image_without_annotations(self, img_path: Path,
                                      dataset: Dataset) -> None:
        """Add an image without annotations (for unlabeled data)."""
        image_ann = ImageAnnotation(
            image_path=img_path,
            image_id=img_path.stem,
            width=0,  # Will be filled later if needed
            height=0
        )
        dataset.add_image(image_ann)

