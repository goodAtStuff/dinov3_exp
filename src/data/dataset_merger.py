"""
Dataset merger for combining multiple dataset roots.
"""

from pathlib import Path
from typing import List, Dict, Tuple
import logging
from collections import defaultdict

from .schemas import Dataset, ImageAnnotation
from .datamuro_adapter import DataMuroAdapter
from ..utils import (
    normalize_path,
    compute_file_hash,
    deduplicate_by_hash,
    get_image_files
)

logger = logging.getLogger(__name__)


class DatasetMerger:
    """
    Merge multiple dataset roots into a unified dataset.
    
    Features:
    - Deduplication by image hash
    - Path normalization across different roots
    - Consistent image ID assignment
    """
    
    def __init__(self, class_names: List[str], 
                 deduplicate: bool = True,
                 hash_algorithm: str = 'md5'):
        """
        Initialize merger.
        
        Args:
            class_names: List of class names
            deduplicate: Whether to deduplicate images by hash
            hash_algorithm: Hash algorithm for deduplication
        """
        self.class_names = class_names
        self.deduplicate = deduplicate
        self.hash_algorithm = hash_algorithm
        self.adapter = DataMuroAdapter(class_names)
    
    def merge_labeled_roots(self, 
                           root_paths: List[Dict[str, str]]) -> Dataset:
        """
        Merge multiple labeled dataset roots.
        
        Args:
            root_paths: List of dicts with 'path' keys pointing to dataset roots
            
        Returns:
            Merged dataset
        """
        if not root_paths:
            logger.warning("No labeled roots provided")
            return Dataset(
                name='empty',
                classes=self.class_names
            )
        
        logger.info(f"Merging {len(root_paths)} labeled dataset roots")
        
        # Load all datasets
        datasets = []
        for root_config in root_paths:
            root_path = normalize_path(root_config['path'])
            
            if not root_path.exists():
                logger.warning(f"Root path does not exist: {root_path}")
                continue
            
            try:
                dataset = self.adapter.load_dataset(root_path)
                datasets.append(dataset)
                logger.info(f"Loaded {dataset.num_images()} images from {root_path}")
            except Exception as e:
                logger.error(f"Failed to load dataset from {root_path}: {e}")
        
        if not datasets:
            raise ValueError("No valid datasets loaded")
        
        # Merge datasets
        merged = self._merge_datasets(datasets)
        
        return merged
    
    def merge_unlabeled_roots(self, 
                             root_paths: List[Dict[str, str]]) -> List[Path]:
        """
        Collect images from multiple unlabeled roots.
        
        Args:
            root_paths: List of dicts with 'path' keys pointing to image directories
            
        Returns:
            List of unique image paths
        """
        if not root_paths:
            logger.warning("No unlabeled roots provided")
            return []
        
        logger.info(f"Collecting images from {len(root_paths)} unlabeled roots")
        
        all_images = []
        
        for root_config in root_paths:
            root_path = normalize_path(root_config['path'])
            
            if not root_path.exists():
                logger.warning(f"Root path does not exist: {root_path}")
                continue
            
            try:
                images = get_image_files(root_path, recursive=True)
                all_images.extend(images)
                logger.info(f"Found {len(images)} images in {root_path}")
            except Exception as e:
                logger.error(f"Failed to scan directory {root_path}: {e}")
        
        if self.deduplicate and all_images:
            logger.info(f"Deduplicating {len(all_images)} unlabeled images...")
            all_images, duplicates = deduplicate_by_hash(
                all_images, 
                algorithm=self.hash_algorithm
            )
            logger.info(f"After deduplication: {len(all_images)} unique images")
        
        return all_images
    
    def _merge_datasets(self, datasets: List[Dataset]) -> Dataset:
        """
        Merge multiple datasets into one.
        
        Handles:
        - Deduplication by image hash
        - Consistent image ID assignment
        - Preserving annotations
        """
        merged = Dataset(
            name='merged',
            classes=self.class_names,
            metadata={'num_sources': len(datasets)}
        )
        
        # Collect all images
        all_images = []
        for dataset in datasets:
            all_images.extend(dataset.images)
        
        logger.info(f"Total images before deduplication: {len(all_images)}")
        
        if self.deduplicate:
            # Deduplicate by image content hash
            unique_images = self._deduplicate_images(all_images)
            logger.info(f"After deduplication: {len(unique_images)} unique images")
        else:
            unique_images = all_images
        
        # Add to merged dataset with new IDs
        for idx, img_ann in enumerate(unique_images):
            # Assign new consistent ID
            img_ann.image_id = f"img_{idx:06d}"
            merged.add_image(img_ann)
        
        stats = merged.get_statistics()
        logger.info(f"Merged dataset statistics: {stats}")
        
        return merged
    
    def _deduplicate_images(self, 
                           images: List[ImageAnnotation]) -> List[ImageAnnotation]:
        """
        Deduplicate images by content hash.
        
        When duplicates are found, prefers the image with more annotations.
        """
        hash_to_images: Dict[str, List[ImageAnnotation]] = defaultdict(list)
        
        # Group by hash
        for img_ann in images:
            try:
                img_hash = compute_file_hash(img_ann.image_path, self.hash_algorithm)
                hash_to_images[img_hash].append(img_ann)
            except Exception as e:
                logger.warning(f"Could not hash image {img_ann.image_path}: {e}")
                # Include unhashable images
                hash_to_images[str(img_ann.image_path)].append(img_ann)
        
        # Select best representative for each hash
        unique_images = []
        num_duplicates = 0
        
        for img_hash, img_list in hash_to_images.items():
            if len(img_list) > 1:
                num_duplicates += len(img_list) - 1
                # Prefer image with more annotations
                best = max(img_list, key=lambda img: img.num_objects())
                logger.debug(f"Found {len(img_list)} duplicates with hash {img_hash[:8]}... "
                           f"Selected {best.image_path} with {best.num_objects()} annotations")
                unique_images.append(best)
            else:
                unique_images.append(img_list[0])
        
        logger.info(f"Removed {num_duplicates} duplicate images")
        
        return unique_images
    
    def create_image_id_mapping(self, dataset: Dataset) -> Dict[str, Path]:
        """
        Create a mapping from image IDs to file paths.
        
        Args:
            dataset: Dataset
            
        Returns:
            Dictionary mapping image_id to image_path
        """
        return {img.image_id: img.image_path for img in dataset.images}

