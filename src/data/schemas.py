"""Data schemas for annotations and datasets."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class BoundingBox:
    """Bounding box in [x_min, y_min, width, height] format (pixels)."""
    x: float
    y: float
    width: float
    height: float
    
    def to_xyxy(self) -> tuple:
        """Convert to [x_min, y_min, x_max, y_max] format."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    def to_xywh(self) -> tuple:
        """Convert to [x, y, width, height] format."""
        return (self.x, self.y, self.width, self.height)
    
    def to_cxcywh(self) -> tuple:
        """Convert to [center_x, center_y, width, height] format."""
        cx = self.x + self.width / 2
        cy = self.y + self.height / 2
        return (cx, cy, self.width, self.height)
    
    def area(self) -> float:
        """Calculate bounding box area."""
        return self.width * self.height
    
    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> 'BoundingBox':
        """Create from [x_min, y_min, x_max, y_max] format."""
        return cls(x=x1, y=y1, width=x2-x1, height=y2-y1)
    
    @classmethod
    def from_cxcywh(cls, cx: float, cy: float, w: float, h: float) -> 'BoundingBox':
        """Create from [center_x, center_y, width, height] format."""
        return cls(x=cx - w/2, y=cy - h/2, width=w, height=h)


@dataclass
class Annotation:
    """Single object annotation."""
    bbox: BoundingBox
    class_id: int
    class_name: str
    confidence: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    is_crowd: bool = False
    area: Optional[float] = None
    
    def __post_init__(self):
        """Calculate area if not provided."""
        if self.area is None:
            self.area = self.bbox.area()


@dataclass
class ImageAnnotation:
    """Annotations for a single image."""
    image_path: Path
    image_id: str
    width: int
    height: int
    annotations: List[Annotation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_annotation(self, annotation: Annotation) -> None:
        """Add an annotation to this image."""
        self.annotations.append(annotation)
    
    def num_objects(self) -> int:
        """Get number of annotated objects."""
        return len(self.annotations)
    
    def get_class_counts(self) -> Dict[str, int]:
        """Get count of objects per class."""
        counts = {}
        for ann in self.annotations:
            counts[ann.class_name] = counts.get(ann.class_name, 0) + 1
        return counts


@dataclass
class Dataset:
    """Complete dataset with images and annotations."""
    name: str
    images: List[ImageAnnotation] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_image(self, image_annotation: ImageAnnotation) -> None:
        """Add an image annotation to the dataset."""
        self.images.append(image_annotation)
    
    def num_images(self) -> int:
        """Get number of images."""
        return len(self.images)
    
    def num_annotations(self) -> int:
        """Get total number of annotations."""
        return sum(img.num_objects() for img in self.images)
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of classes across all images."""
        distribution = {cls: 0 for cls in self.classes}
        for img in self.images:
            for ann in img.annotations:
                distribution[ann.class_name] = distribution.get(ann.class_name, 0) + 1
        return distribution
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            'num_images': self.num_images(),
            'num_annotations': self.num_annotations(),
            'classes': self.classes,
            'class_distribution': self.get_class_distribution(),
            'avg_objects_per_image': self.num_annotations() / max(self.num_images(), 1),
        }


@dataclass
class Split:
    """Dataset split (train/val/test)."""
    name: str
    image_ids: List[str] = field(default_factory=list)
    
    def add_image(self, image_id: str) -> None:
        """Add an image to this split."""
        self.image_ids.append(image_id)
    
    def num_images(self) -> int:
        """Get number of images in split."""
        return len(self.image_ids)


@dataclass
class DatasetSplits:
    """Train/val/test splits for a dataset."""
    train: Split = field(default_factory=lambda: Split(name='train'))
    val: Split = field(default_factory=lambda: Split(name='val'))
    test: Split = field(default_factory=lambda: Split(name='test'))
    
    def get_split(self, name: str) -> Split:
        """Get split by name."""
        if name == 'train':
            return self.train
        elif name in ['val', 'validation']:
            return self.val
        elif name == 'test':
            return self.test
        else:
            raise ValueError(f"Unknown split name: {name}")
    
    def get_statistics(self) -> Dict[str, int]:
        """Get split statistics."""
        return {
            'train': self.train.num_images(),
            'val': self.val.num_images(),
            'test': self.test.num_images(),
            'total': self.train.num_images() + self.val.num_images() + self.test.num_images(),
        }

