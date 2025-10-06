"""
Dice class mapping for multi-class detection.

Maps die type + face value to unique class IDs.
Total: 71 classes (including background class 0)
"""

from typing import Dict, List, Tuple, Optional


# Die types and their possible values
DIE_TYPES = {
    'd4': [1, 2, 3, 4],
    'd6': [1, 2, 3, 4, 5, 6],
    'd8': [1, 2, 3, 4, 5, 6, 7, 8],
    'd10': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 0 maps to 10
    'd12': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'd20': list(range(1, 21)),  # 1-20
    'd100': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],  # 0 maps to 100
}


def normalize_die_value(die_type: str, value: float) -> int:
    """
    Normalize die value, handling special cases for d10 and d100.
    
    Special cases:
    - d10: value 0 → 10
    - d100: value 0 → 100
    
    Args:
        die_type: Die type (d4, d6, d8, d10, d12, d20, d100)
        value: Face value shown on die
        
    Returns:
        Normalized integer value
    """
    value = int(value)
    
    # Special case: d10 shows 0 instead of 10
    if die_type == 'd10' and value == 0:
        return 10
    
    # Special case: d100 shows 0 instead of 100
    if die_type == 'd100' and value == 0:
        return 100
    
    return value


def create_class_mapping() -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """
    Create class name list and bidirectional mapping.
    
    Returns:
        Tuple of:
        - class_names: List of class names (index = class_id)
        - name_to_id: Dict mapping class name to class ID
        - id_to_name: Dict mapping class ID to class name
    """
    class_names = ['background']  # Class 0 is background
    name_to_id = {'background': 0}
    id_to_name = {0: 'background'}
    
    class_id = 1
    
    # Build classes in order: d4, d6, d8, d10, d12, d20, d100
    for die_type in ['d4', 'd6', 'd8', 'd10', 'd12', 'd20', 'd100']:
        for value in DIE_TYPES[die_type]:
            class_name = f"{die_type}_{value}"
            class_names.append(class_name)
            name_to_id[class_name] = class_id
            id_to_name[class_id] = class_name
            class_id += 1
    
    return class_names, name_to_id, id_to_name


def get_class_id(die_type: str, value: float) -> Optional[int]:
    """
    Get class ID for a die type and value combination.
    
    Args:
        die_type: Die type (d4, d6, d8, d10, d12, d20, d100)
        value: Face value shown on die
        
    Returns:
        Class ID (1-70), or None if invalid
    """
    # Normalize die type (handle variations)
    die_type = die_type.lower().strip()
    if not die_type.startswith('d'):
        die_type = f'd{die_type}'
    
    # Check if valid die type
    if die_type not in DIE_TYPES:
        return None
    
    # Normalize value (handle special cases)
    normalized_value = normalize_die_value(die_type, value)
    
    # Check if valid value for this die type
    if normalized_value not in DIE_TYPES[die_type]:
        return None
    
    # Get class name and look up ID
    class_name = f"{die_type}_{normalized_value}"
    _, name_to_id, _ = create_class_mapping()
    
    return name_to_id.get(class_name)


def get_class_name(die_type: str, value: float) -> Optional[str]:
    """
    Get class name for a die type and value combination.
    
    Args:
        die_type: Die type (d4, d6, d8, d10, d12, d20, d100)
        value: Face value shown on die
        
    Returns:
        Class name (e.g., 'd6_4'), or None if invalid
    """
    # Normalize die type
    die_type = die_type.lower().strip()
    if not die_type.startswith('d'):
        die_type = f'd{die_type}'
    
    if die_type not in DIE_TYPES:
        return None
    
    # Normalize value
    normalized_value = normalize_die_value(die_type, value)
    
    if normalized_value not in DIE_TYPES[die_type]:
        return None
    
    return f"{die_type}_{normalized_value}"


def get_die_info_from_class_id(class_id: int) -> Optional[Tuple[str, int]]:
    """
    Get die type and value from class ID.
    
    Args:
        class_id: Class ID (0-70)
        
    Returns:
        Tuple of (die_type, value), or None if class_id is 0 (background)
    """
    if class_id == 0:
        return None  # Background
    
    _, _, id_to_name = create_class_mapping()
    class_name = id_to_name.get(class_id)
    
    if not class_name:
        return None
    
    # Parse class name: "d6_4" -> ("d6", 4)
    parts = class_name.split('_')
    if len(parts) != 2:
        return None
    
    die_type = parts[0]
    value = int(parts[1])
    
    return (die_type, value)


def get_class_statistics() -> Dict[str, int]:
    """
    Get statistics about class distribution.
    
    Returns:
        Dictionary with statistics
    """
    class_names, _, _ = create_class_mapping()
    
    stats = {
        'total_classes': len(class_names),
        'background_class': 1,
        'd4_classes': len(DIE_TYPES['d4']),
        'd6_classes': len(DIE_TYPES['d6']),
        'd8_classes': len(DIE_TYPES['d8']),
        'd10_classes': len(DIE_TYPES['d10']),
        'd12_classes': len(DIE_TYPES['d12']),
        'd20_classes': len(DIE_TYPES['d20']),
        'd100_classes': len(DIE_TYPES['d100']),
    }
    
    return stats


# Pre-compute the class mapping for efficiency
CLASS_NAMES, NAME_TO_ID, ID_TO_NAME = create_class_mapping()


def validate_class_mapping():
    """Validate that class mapping is correct."""
    assert len(CLASS_NAMES) == 71, f"Expected 71 classes, got {len(CLASS_NAMES)}"
    assert CLASS_NAMES[0] == 'background'
    
    # Verify no duplicates
    assert len(CLASS_NAMES) == len(set(CLASS_NAMES))
    assert len(NAME_TO_ID) == len(CLASS_NAMES)
    assert len(ID_TO_NAME) == len(CLASS_NAMES)
    
    # Verify special cases
    assert get_class_id('d10', 0) == get_class_id('d10', 10)
    assert get_class_id('d100', 0) == get_class_id('d100', 100)
    
    print("✓ Class mapping validation passed")
    print(f"  Total classes: {len(CLASS_NAMES)}")
    print(f"  Background: {CLASS_NAMES[0]}")
    print(f"  First die class: {CLASS_NAMES[1]}")
    print(f"  Last die class: {CLASS_NAMES[-1]}")


if __name__ == '__main__':
    # Validate and print statistics
    validate_class_mapping()
    
    print("\nClass Statistics:")
    stats = get_class_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nSample Class Names:")
    print(f"  Class 0: {CLASS_NAMES[0]}")
    print(f"  Class 1: {CLASS_NAMES[1]}")
    print(f"  Class 10: {CLASS_NAMES[10]}")
    print(f"  Class 28: {CLASS_NAMES[28]} (d10_10, also covers d10 showing 0)")
    print(f"  Class 70: {CLASS_NAMES[70]} (d100_100, also covers d100 showing 0)")
    
    print("\nSpecial Cases:")
    print(f"  d10 value 0 → class {get_class_id('d10', 0)} ({get_class_name('d10', 0)})")
    print(f"  d10 value 10 → class {get_class_id('d10', 10)} ({get_class_name('d10', 10)})")
    print(f"  d100 value 0 → class {get_class_id('d100', 0)} ({get_class_name('d100', 0)})")
    print(f"  d100 value 100 → class {get_class_id('d100', 100)} ({get_class_name('d100', 100)})")

