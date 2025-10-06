"""Path normalization and file discovery utilities."""

import os
from pathlib import Path
from typing import List, Union, Optional
import logging

logger = logging.getLogger(__name__)

# Supported image formats
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def normalize_path(path: Union[str, Path]) -> Path:
    """
    Normalize a path for cross-platform compatibility.
    
    Handles:
    - Windows paths (C:\\, E:\\)
    - UNC paths (\\\\server\\share)
    - Forward/backward slashes
    - Relative paths
    
    Args:
        path: Input path as string or Path
        
    Returns:
        Normalized Path object
    """
    if isinstance(path, str):
        # Handle UNC paths on Windows
        if path.startswith('\\\\') or path.startswith('//'):
            # Keep UNC prefix intact
            path = Path(path)
        else:
            # Convert to Path and resolve
            path = Path(path)
    
    # Expand user home directory if needed
    path = path.expanduser()
    
    # Resolve to absolute path if it exists, otherwise just normalize
    try:
        if path.exists():
            path = path.resolve()
        else:
            # For non-existent paths, just normalize without resolving
            path = Path(os.path.normpath(str(path)))
    except (OSError, RuntimeError) as e:
        # Handle cases where resolve() fails (e.g., network issues)
        logger.warning(f"Could not resolve path {path}: {e}")
        path = Path(os.path.normpath(str(path)))
    
    return path


def validate_path(path: Union[str, Path], must_exist: bool = True, 
                 is_dir: bool = False, is_file: bool = False) -> Path:
    """
    Validate and normalize a path.
    
    Args:
        path: Path to validate
        must_exist: If True, raise error if path doesn't exist
        is_dir: If True, check that path is a directory
        is_file: If True, check that path is a file
        
    Returns:
        Normalized Path object
        
    Raises:
        ValueError: If validation fails
    """
    path = normalize_path(path)
    
    if must_exist and not path.exists():
        raise ValueError(f"Path does not exist: {path}")
    
    if is_dir and path.exists() and not path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")
    
    if is_file and path.exists() and not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    
    return path


def get_image_files(directory: Union[str, Path], 
                   recursive: bool = True,
                   extensions: Optional[set] = None) -> List[Path]:
    """
    Get all image files in a directory.
    
    Args:
        directory: Directory to search
        recursive: If True, search subdirectories
        extensions: Set of file extensions to include (default: IMAGE_EXTENSIONS)
        
    Returns:
        List of image file paths
    """
    directory = normalize_path(directory)
    
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []
    
    if not directory.is_dir():
        logger.warning(f"Path is not a directory: {directory}")
        return []
    
    if extensions is None:
        extensions = IMAGE_EXTENSIONS
    
    # Normalize extensions to lowercase with dots
    extensions = {ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                  for ext in extensions}
    
    image_files = []
    
    try:
        if recursive:
            # Recursively search all subdirectories
            for ext in extensions:
                image_files.extend(directory.rglob(f'*{ext}'))
                # Also search uppercase variants
                image_files.extend(directory.rglob(f'*{ext.upper()}'))
        else:
            # Search only immediate directory
            for ext in extensions:
                image_files.extend(directory.glob(f'*{ext}'))
                image_files.extend(directory.glob(f'*{ext.upper()}'))
    except PermissionError as e:
        logger.error(f"Permission denied accessing directory {directory}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error scanning directory {directory}: {e}")
        return []
    
    # Remove duplicates and sort
    image_files = sorted(set(image_files))
    
    logger.info(f"Found {len(image_files)} image files in {directory}")
    
    return image_files


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Normalized Path object
    """
    path = normalize_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_relative_path(path: Union[str, Path], base: Union[str, Path]) -> Path:
    """
    Get the relative path from base to path.
    
    Args:
        path: Target path
        base: Base directory
        
    Returns:
        Relative path
    """
    path = normalize_path(path)
    base = normalize_path(base)
    
    try:
        return path.relative_to(base)
    except ValueError:
        # Paths are not relative, return the full path
        return path

