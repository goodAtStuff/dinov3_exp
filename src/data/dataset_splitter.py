"""
Dataset splitter with deterministic split logic.
"""

import random
from typing import List, Dict, Tuple
import logging
from collections import defaultdict

from .schemas import Dataset, DatasetSplits, ImageAnnotation
from ..utils import hash_string

logger = logging.getLogger(__name__)


class DatasetSplitter:
    """
    Split dataset into train/val/test with deterministic hashing.
    
    Features:
    - Deterministic splitting based on seed
    - Prevents data leakage for duplicate images
    - Configurable split ratios
    - Validates minimum samples per split
    """
    
    def __init__(self, 
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1,
                 test_ratio: float = 0.1,
                 seed: int = 42,
                 min_train: int = 10,
                 min_val: int = 3,
                 min_test: int = 3):
        """
        Initialize splitter.
        
        Args:
            train_ratio: Fraction of data for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            seed: Random seed for reproducibility
            min_train: Minimum training samples
            min_val: Minimum validation samples
            min_test: Minimum test samples
        """
        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.min_train = min_train
        self.min_val = min_val
        self.min_test = min_test
        
        # Set random seed
        random.seed(seed)
    
    def split_dataset(self, dataset: Dataset) -> DatasetSplits:
        """
        Split a dataset into train/val/test.
        
        Args:
            dataset: Dataset to split
            
        Returns:
            DatasetSplits with train/val/test assignments
        """
        if dataset.num_images() == 0:
            raise ValueError("Cannot split empty dataset")
        
        logger.info(f"Splitting {dataset.num_images()} images into train/val/test")
        
        # Create splits
        splits = DatasetSplits()
        
        # Get all images
        images = dataset.images
        
        # Group by content hash to prevent leakage
        hash_groups = self._group_by_content(images)
        
        logger.info(f"Grouped into {len(hash_groups)} unique image groups")
        
        # Shuffle groups deterministically
        group_items = list(hash_groups.items())
        random.shuffle(group_items)
        
        # Calculate split points
        n_groups = len(group_items)
        n_train = int(n_groups * self.train_ratio)
        n_val = int(n_groups * self.val_ratio)
        
        # Ensure minimum counts
        if n_train < self.min_train:
            n_train = min(self.min_train, n_groups - self.min_val - self.min_test)
        if n_val < self.min_val:
            n_val = min(self.min_val, n_groups - n_train - self.min_test)
        
        # Assign groups to splits
        train_groups = group_items[:n_train]
        val_groups = group_items[n_train:n_train + n_val]
        test_groups = group_items[n_train + n_val:]
        
        # Collect image IDs
        for _, img_list in train_groups:
            for img in img_list:
                splits.train.add_image(img.image_id)
        
        for _, img_list in val_groups:
            for img in img_list:
                splits.val.add_image(img.image_id)
        
        for _, img_list in test_groups:
            for img in img_list:
                splits.test.add_image(img.image_id)
        
        # Log statistics
        stats = splits.get_statistics()
        logger.info(f"Split statistics: {stats}")
        
        # Validate splits
        self._validate_splits(splits, dataset)
        
        return splits
    
    def _group_by_content(self, 
                         images: List[ImageAnnotation]) -> Dict[str, List[ImageAnnotation]]:
        """
        Group images by content to prevent leakage.
        
        Uses image path hash as a simple content identifier.
        In a production system, would use actual content hashes.
        """
        groups = defaultdict(list)
        
        for img in images:
            # Create a stable hash from the image path
            # This ensures duplicates (same path) stay together
            img_hash = hash_string(str(img.image_path))
            groups[img_hash].append(img)
        
        return groups
    
    def _validate_splits(self, splits: DatasetSplits, dataset: Dataset) -> None:
        """
        Validate that splits are reasonable.
        
        Checks:
        - Minimum sample requirements
        - No overlap between splits
        - All images assigned
        """
        stats = splits.get_statistics()
        
        # Check minimum counts
        if stats['train'] < self.min_train:
            logger.warning(f"Train split has only {stats['train']} samples "
                         f"(minimum: {self.min_train})")
        
        if stats['val'] < self.min_val:
            logger.warning(f"Val split has only {stats['val']} samples "
                         f"(minimum: {self.min_val})")
        
        if stats['test'] < self.min_test:
            logger.warning(f"Test split has only {stats['test']} samples "
                         f"(minimum: {self.min_test})")
        
        # Check for overlap
        train_set = set(splits.train.image_ids)
        val_set = set(splits.val.image_ids)
        test_set = set(splits.test.image_ids)
        
        overlap_train_val = train_set & val_set
        overlap_train_test = train_set & test_set
        overlap_val_test = val_set & test_set
        
        if overlap_train_val or overlap_train_test or overlap_val_test:
            raise ValueError("Data leakage detected: splits have overlapping images")
        
        # Check all images assigned
        total_assigned = len(train_set) + len(val_set) + len(test_set)
        if total_assigned != dataset.num_images():
            logger.warning(f"Not all images assigned: {total_assigned}/{dataset.num_images()}")
        
        # Check class distribution
        self._check_class_distribution(splits, dataset)
        
        logger.info("Split validation passed")
    
    def _check_class_distribution(self, splits: DatasetSplits, 
                                  dataset: Dataset) -> None:
        """
        Check and log class distribution across splits.
        """
        # Create image ID to annotation mapping
        id_to_img = {img.image_id: img for img in dataset.images}
        
        # Count classes per split
        for split_name in ['train', 'val', 'test']:
            split = splits.get_split(split_name)
            class_counts = defaultdict(int)
            
            for img_id in split.image_ids:
                img = id_to_img.get(img_id)
                if img:
                    for ann in img.annotations:
                        class_counts[ann.class_name] += 1
            
            logger.info(f"{split_name.capitalize()} class distribution: {dict(class_counts)}")
            
            # Check for missing classes
            for class_name in dataset.classes:
                if class_counts[class_name] == 0:
                    logger.warning(f"Class '{class_name}' has no samples in {split_name} split")
    
    def get_split_datasets(self, dataset: Dataset, 
                          splits: DatasetSplits) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Create separate Dataset objects for each split.
        
        Args:
            dataset: Original dataset
            splits: Split assignments
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Create ID to image mapping
        id_to_img = {img.image_id: img for img in dataset.images}
        
        # Create split datasets
        train_dataset = Dataset(
            name=f"{dataset.name}_train",
            classes=dataset.classes,
            metadata=dataset.metadata.copy()
        )
        
        val_dataset = Dataset(
            name=f"{dataset.name}_val",
            classes=dataset.classes,
            metadata=dataset.metadata.copy()
        )
        
        test_dataset = Dataset(
            name=f"{dataset.name}_test",
            classes=dataset.classes,
            metadata=dataset.metadata.copy()
        )
        
        # Assign images to splits
        for img_id in splits.train.image_ids:
            if img_id in id_to_img:
                train_dataset.add_image(id_to_img[img_id])
        
        for img_id in splits.val.image_ids:
            if img_id in id_to_img:
                val_dataset.add_image(id_to_img[img_id])
        
        for img_id in splits.test.image_ids:
            if img_id in id_to_img:
                test_dataset.add_image(id_to_img[img_id])
        
        return train_dataset, val_dataset, test_dataset

