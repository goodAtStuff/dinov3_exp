"""File hashing and deduplication utilities."""

import hashlib
from pathlib import Path
from typing import Dict, List, Union, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_file_hash(file_path: Union[str, Path], 
                     algorithm: str = 'md5',
                     buffer_size: int = 65536) -> str:
    """
    Compute hash of a file.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5', 'sha256', etc.)
        buffer_size: Read buffer size in bytes
        
    Returns:
        Hex digest of file hash
        
    Raises:
        ValueError: If algorithm is not supported
        IOError: If file cannot be read
    """
    if algorithm not in hashlib.algorithms_available:
        raise ValueError(f"Hash algorithm '{algorithm}' not available. "
                       f"Available: {hashlib.algorithms_available}")
    
    hasher = hashlib.new(algorithm)
    
    try:
        with open(file_path, 'rb') as f:
            while True:
                data = f.read(buffer_size)
                if not data:
                    break
                hasher.update(data)
    except IOError as e:
        logger.error(f"Error reading file {file_path}: {e}")
        raise
    
    return hasher.hexdigest()


def compute_content_hash(content: bytes, algorithm: str = 'md5') -> str:
    """
    Compute hash of binary content.
    
    Args:
        content: Binary content
        algorithm: Hash algorithm
        
    Returns:
        Hex digest of content hash
    """
    if algorithm not in hashlib.algorithms_available:
        raise ValueError(f"Hash algorithm '{algorithm}' not available")
    
    hasher = hashlib.new(algorithm)
    hasher.update(content)
    return hasher.hexdigest()


def deduplicate_by_hash(file_paths: List[Path], 
                       algorithm: str = 'md5',
                       keep: str = 'first') -> Tuple[List[Path], Dict[str, List[Path]]]:
    """
    Deduplicate a list of files by their content hash.
    
    Args:
        file_paths: List of file paths
        algorithm: Hash algorithm to use
        keep: Which duplicate to keep ('first', 'last', 'shortest_path')
        
    Returns:
        Tuple of (unique_files, duplicates_dict)
        - unique_files: List of unique file paths
        - duplicates_dict: Dict mapping hash to list of duplicate paths
    """
    hash_to_paths: Dict[str, List[Path]] = {}
    
    logger.info(f"Computing hashes for {len(file_paths)} files...")
    
    for file_path in file_paths:
        try:
            file_hash = compute_file_hash(file_path, algorithm)
            if file_hash not in hash_to_paths:
                hash_to_paths[file_hash] = []
            hash_to_paths[file_hash].append(file_path)
        except Exception as e:
            logger.warning(f"Could not hash file {file_path}: {e}")
            continue
    
    unique_files = []
    duplicates = {}
    
    for file_hash, paths in hash_to_paths.items():
        if len(paths) > 1:
            # Store duplicates
            duplicates[file_hash] = paths
            
            # Select which one to keep
            if keep == 'first':
                selected = paths[0]
            elif keep == 'last':
                selected = paths[-1]
            elif keep == 'shortest_path':
                selected = min(paths, key=lambda p: len(str(p)))
            else:
                raise ValueError(f"Invalid 'keep' option: {keep}")
            
            unique_files.append(selected)
            logger.debug(f"Found {len(paths)} duplicates with hash {file_hash[:8]}...")
        else:
            unique_files.append(paths[0])
    
    num_duplicates = sum(len(paths) - 1 for paths in duplicates.values())
    logger.info(f"Deduplication: {len(file_paths)} files → {len(unique_files)} unique "
               f"({num_duplicates} duplicates removed)")
    
    return unique_files, duplicates


def get_file_identifier(file_path: Path, base_path: Path = None) -> str:
    """
    Create a stable identifier for a file based on its path and content.
    
    This is useful for consistent splitting across runs.
    
    Args:
        file_path: Path to file
        base_path: Optional base path to compute relative path from
        
    Returns:
        String identifier combining relative path and content hash
    """
    if base_path:
        try:
            rel_path = file_path.relative_to(base_path)
        except ValueError:
            rel_path = file_path
    else:
        rel_path = file_path
    
    # Use path as primary identifier
    path_str = str(rel_path).replace('\\', '/')
    
    # Add content hash for uniqueness
    try:
        content_hash = compute_file_hash(file_path, 'md5')
        identifier = f"{path_str}::{content_hash}"
    except Exception:
        # Fallback to just path if hashing fails
        identifier = path_str
    
    return identifier


def hash_string(s: str, algorithm: str = 'md5') -> str:
    """
    Compute hash of a string.
    
    Args:
        s: Input string
        algorithm: Hash algorithm
        
    Returns:
        Hex digest of string hash
    """
    return compute_content_hash(s.encode('utf-8'), algorithm)

