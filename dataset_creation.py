import copy
import json

def create_pcb_only_annotations(input_json_path, output_json_path):
    """
    Create a new annotation file with only PCB annotations.
    This is used to train the first stage PCB detector.
    
    Args:
        input_json_path: Path to the original COCO annotation file
        output_json_path: Path to save the PCB-only annotation file
    """
    
    # Load original annotations
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    # Create a copy with only PCB annotations (category_id 37, which is index 36)
    pcb_data = copy.deepcopy(data)
    
    # Filter annotations to only keep PCBs
    pcb_annotations = []
    for anno in data['annotations']:
        if anno['category_id'] == 37:  # PCBA class (original category ID)
            # Change category_id to 1 (since we'll only have one class)
            anno_copy = copy.deepcopy(anno)
            anno_copy['category_id'] = 1
            pcb_annotations.append(anno_copy)
    
    # Update the categories list to only have PCB
    pcb_data['categories'] = [{'id': 1, 'name': 'PCBA', 'supercategory': 'PCBA'}]
    
    # Update the annotations
    pcb_data['annotations'] = pcb_annotations
    
    # Save the new annotation file
    with open(output_json_path, 'w') as f:
        json.dump(pcb_data, f)
    
    print(f"Created PCB-only annotation file with {len(pcb_annotations)} annotations")


def create_component_only_annotations(input_json_path, output_json_path):
    """
    Create a new annotation file with only component annotations (no PCB).
    
    Args:
        input_json_path: Path to the original COCO annotation file
        output_json_path: Path to save the component-only annotation file
    """
    
    # Load original annotations
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    # Create a copy with only component annotations (exclude category_id 37)
    component_data = copy.deepcopy(data)
    
    # Filter annotations to exclude PCBs
    component_annotations = []
    for anno in data['annotations']:
        if anno['category_id'] != 37:  # Not PCBA class
            component_annotations.append(copy.deepcopy(anno))
    
    # Update the categories list to exclude PCB
    component_categories = []
    for category in data['categories']:
        if category['id'] != 37:  # Not PCBA class
            component_categories.append(copy.deepcopy(category))
    
    # Update the annotations and categories
    component_data['annotations'] = component_annotations
    component_data['categories'] = component_categories
    
    # Save the new annotation file
    with open(output_json_path, 'w') as f:
        json.dump(component_data, f)
    
    print(f"Created component-only annotation file with {len(component_annotations)} annotations")


def create_rotation_dataset(input_json_path, output_json_path):
    """
    Create a new annotation file with only capacitor electrolytic and connector annotations.

    Args:
        input_json_path: Path to the original COCO annotation file
        output_json_path: Path to save the filtered annotation file
    """

    # Load original annotations
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    # Create a copy for the filtered data
    filtered_data = copy.deepcopy(data)

    # Filter annotations to only keep capacitor electrolytic (ID 4) and connector (ID 18)
    filtered_annotations = []
    for anno in data['annotations']:
        if anno['category_id'] in [4, 18]:
            # Keep the original category IDs
            filtered_annotations.append(copy.deepcopy(anno))

    # Update the categories list to only have the two categories
    filtered_categories = []
    for category in data['categories']:
        if category['id'] in [4, 18]:
            filtered_categories.append(copy.deepcopy(category))

    # Update the filtered data
    filtered_data['annotations'] = filtered_annotations
    filtered_data['categories'] = filtered_categories

    # Save the new annotation file
    with open(output_json_path, 'w') as f:
        json.dump(filtered_data, f)

    # Count annotations per category
    cap_count = sum(1 for anno in filtered_annotations if anno['category_id'] == 4)
    conn_count = sum(1 for anno in filtered_annotations if anno['category_id'] == 18)

    print(f"Created annotation file with {len(filtered_annotations)} annotations:")
    print(f"  - Capacitor electrolytic: {cap_count} annotations")
    print(f"  - Connector: {conn_count} annotations")


