"""
Dataset builder script.

Merges multiple dataset roots, parses DataMuro annotations,
and exports unified dataset in requested format (COCO/Ultralytics).
"""

import argparse
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DatasetMerger, DatasetSplitter, COCOExporter, UltralyticsExporter
from src.data.dice_classes import CLASS_NAMES
from src.utils import (
    setup_logger,
    load_manifest,
    save_yaml,
    normalize_path,
    ensure_dir
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Build unified dataset from multiple roots',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--manifest',
        type=str,
        required=True,
        help='Path to manifest YAML file'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: data/processed/<run_id>)'
    )
    
    parser.add_argument(
        '--export-format',
        type=str,
        choices=['coco', 'ultralytics', 'both'],
        default=None,
        help='Export format (default: from manifest)'
    )
    
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Maximum images to process (for debugging)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (default: from manifest)'
    )
    
    parser.add_argument(
        '--no-deduplicate',
        action='store_true',
        help='Disable image deduplication'
    )
    
    parser.add_argument(
        '--no-copy-images',
        action='store_true',
        help='Use symlinks instead of copying images'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run: show what would be done without executing'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger('build_dataset', level=log_level, console=True)
    
    logger.info("=" * 80)
    logger.info("Dataset Builder")
    logger.info("=" * 80)
    
    try:
        # Load manifest
        logger.info(f"Loading manifest: {args.manifest}")
        manifest = load_manifest(args.manifest)
        
        run_id = manifest['run_id']
        classes = manifest['classes']
        
        # Handle auto-generated class list
        if classes == 'auto' or (isinstance(classes, list) and len(classes) == 1 and classes[0] == 'auto'):
            logger.info("Using auto-generated class list (71 classes)")
            classes = CLASS_NAMES
            manifest['classes'] = classes  # Update manifest
        
        label_format = manifest.get('label_format', 'datamuro')
        
        # Override settings from command line
        if args.seed is not None:
            manifest['seed'] = args.seed
        
        if args.export_format is not None:
            manifest['export']['format'] = args.export_format
        
        seed = manifest.get('seed', 42)
        export_format = manifest['export'].get('format', 'coco')
        
        # Determine output directory
        if args.output_dir:
            output_dir = normalize_path(args.output_dir)
        else:
            output_dir = Path('data') / 'processed' / run_id
        
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Classes: {classes}")
        logger.info(f"Label format: {label_format}")
        logger.info(f"Export format: {export_format}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Seed: {seed}")
        
        if args.dry_run:
            logger.info("DRY RUN: No files will be created")
            return
        
        # Create output directory
        ensure_dir(output_dir)
        
        # Setup log file
        log_file = output_dir / 'build_dataset.log'
        logger = setup_logger('build_dataset', level=log_level, 
                            log_file=log_file, console=True)
        
        # Step 1: Merge labeled datasets
        logger.info("\n" + "=" * 80)
        logger.info("Step 1: Merging labeled datasets")
        logger.info("=" * 80)
        
        merger = DatasetMerger(
            class_names=classes,
            deduplicate=not args.no_deduplicate,
            hash_algorithm='md5'
        )
        
        labeled_roots = manifest['roots'].get('labeled', [])
        
        if labeled_roots:
            dataset = merger.merge_labeled_roots(labeled_roots)
            logger.info(f"Merged dataset: {dataset.num_images()} images, "
                       f"{dataset.num_annotations()} annotations")
            
            # Apply max images limit
            if args.max_images and dataset.num_images() > args.max_images:
                logger.info(f"Limiting to {args.max_images} images")
                dataset.images = dataset.images[:args.max_images]
            
            # Log statistics
            stats = dataset.get_statistics()
            logger.info(f"Dataset statistics: {stats}")
        else:
            logger.warning("No labeled roots specified")
            dataset = None
        
        # Step 2: Split dataset
        if dataset:
            logger.info("\n" + "=" * 80)
            logger.info("Step 2: Splitting dataset")
            logger.info("=" * 80)
            
            splits_config = manifest.get('splits', {})
            splitter = DatasetSplitter(
                train_ratio=splits_config.get('train_ratio', 0.8),
                val_ratio=splits_config.get('val_ratio', 0.1),
                test_ratio=splits_config.get('test_ratio', 0.1),
                seed=seed
            )
            
            splits = splitter.split_dataset(dataset)
            
            # Save split manifests
            split_manifest = {
                'train': splits.train.image_ids,
                'val': splits.val.image_ids,
                'test': splits.test.image_ids,
            }
            save_yaml(split_manifest, output_dir / 'splits.yaml')
            logger.info(f"Saved split manifest: {output_dir / 'splits.yaml'}")
        else:
            splits = None
        
        # Step 3: Export dataset
        if dataset and splits:
            logger.info("\n" + "=" * 80)
            logger.info("Step 3: Exporting dataset")
            logger.info("=" * 80)
            
            if export_format in ['coco', 'both']:
                logger.info("Exporting to COCO format...")
                coco_exporter = COCOExporter(
                    output_dir=output_dir,
                    copy_images=not args.no_copy_images
                )
                coco_files = coco_exporter.export(dataset, splits, run_id)
                
                for split_name, json_path in coco_files.items():
                    logger.info(f"  {split_name}: {json_path}")
            
            if export_format in ['ultralytics', 'both']:
                logger.info("Exporting to Ultralytics format...")
                ultralytics_dir = output_dir / 'ultralytics'
                ultralytics_exporter = UltralyticsExporter(ultralytics_dir)
                yaml_path = ultralytics_exporter.export(dataset, splits)
                logger.info(f"  Data YAML: {yaml_path}")
        
        # Step 4: Collect unlabeled images
        logger.info("\n" + "=" * 80)
        logger.info("Step 4: Collecting unlabeled images")
        logger.info("=" * 80)
        
        unlabeled_roots = manifest['roots'].get('unlabeled', [])
        
        if unlabeled_roots:
            unlabeled_images = merger.merge_unlabeled_roots(unlabeled_roots)
            logger.info(f"Collected {len(unlabeled_images)} unlabeled images")
            
            # Save unlabeled image list
            unlabeled_list = [str(img) for img in unlabeled_images]
            save_yaml({'images': unlabeled_list}, 
                     output_dir / 'unlabeled_images.yaml')
            logger.info(f"Saved unlabeled image list: "
                       f"{output_dir / 'unlabeled_images.yaml'}")
        else:
            logger.info("No unlabeled roots specified")
        
        # Save complete manifest
        manifest['output_dir'] = str(output_dir)
        save_yaml(manifest, output_dir / 'manifest.yaml')
        
        # Success
        logger.info("\n" + "=" * 80)
        logger.info("Dataset build complete!")
        logger.info("=" * 80)
        logger.info(f"Output directory: {output_dir}")
        
        if dataset:
            logger.info(f"Total images: {dataset.num_images()}")
            logger.info(f"Total annotations: {dataset.num_annotations()}")
        
        if splits:
            split_stats = splits.get_statistics()
            logger.info(f"Splits: {split_stats}")
        
        logger.info(f"\nNext steps:")
        logger.info(f"  1. Train baseline: python scripts/train.py --data {output_dir / 'coco.yaml'}")
        logger.info(f"  2. Or distill first: python scripts/distill.py --manifest {args.manifest}")
        
    except Exception as e:
        logger.error(f"Error building dataset: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

