import torch
import numpy as np
import cv2
import copy
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings
from collections import Counter
import torchvision.ops

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def tile_image_inference(image, overlap=0.2, visualize=False):
    """
    Tile an image and its annotations into smaller patches based on image size.
    Ensures that sqrt(height*width) < 1200 for each tile, and the longest edge is at most 1333px.
    Returns a list of dicts, each representing a tile and its annotations.
    
    Args:
        image: Input image
        annotations: List of annotation dictionaries
        overlap: Overlap between adjacent tiles (0.0 to 1.0)
        visualize: Whether to visualize the tiles
        
    Returns:
        tiles: List of dictionaries containing tile images and annotations
    """
    tiles = []
    img_h, img_w = image.shape[:2]
    
    # Calculate the maximum area constraint
    max_area = 1333 * 800
    # Maximum edge length constraint
    max_edge = 1333
    
    # Calculate optimal tile dimensions based on constraints
    if img_w >= img_h:  # Wider image
        # First, cap the width at max_edge
        tile_w = min(img_w, max_edge)
        # Calculate height based on max area
        tile_h = min(int(max_area / tile_w), img_h)
    else:  # Taller image
        # First, cap the height at max_edge
        tile_h = min(img_h, max_edge)
        # Calculate width based on max area
        tile_w = min(int(max_area / tile_h), img_w)
    
    # Verify our constraint is met for sqrt(h*w) < sqrt(max_area)
    while math.sqrt(tile_h * tile_w) > math.sqrt(max_area):
        # Reduce dimensions proportionally
        scale_factor = math.sqrt(max_area) / math.sqrt(tile_h * tile_w)
        tile_h = int(tile_h * scale_factor)
        tile_w = int(tile_w * scale_factor)
    
    # Ensure shortest edge is at least 640px
    min_edge = min(tile_h, tile_w)
    if min_edge < 640:
        scale_factor = 640 / min_edge
        # Scale up, but maintain constraints
        new_h = int(tile_h * scale_factor)
        new_w = int(tile_w * scale_factor)
        
        # Make sure we don't exceed max_edge
        if new_h > max_edge:
            new_h = max_edge
            new_w = min(int(max_area / new_h), max_edge)
        elif new_w > max_edge:
            new_w = max_edge
            new_h = min(int(max_area / new_w), max_edge)
        
        # Check if sqrt(h*w) constraint is still met
        if math.sqrt(new_h * new_w) <=  math.sqrt(max_area):
            tile_h, tile_w = new_h, new_w
        else:
            # If scaling up would violate our constraint, adjust differently
            if tile_h < tile_w:
                tile_h = 640
                tile_w = min(int(max_area / tile_h), max_edge)
            else:
                tile_w = 640
                tile_h = min(int(max_area / tile_w), max_edge)
    
    # If the image already fits our constraints, use it as one tile
    if img_w <= tile_w and img_h <= tile_h:
        tiles.append({
            "image": image,
            "coordinates": (0, 0, img_w, img_h)
        })
        
        if visualize:
            print(f"Image fits in one tile: {img_w}x{img_h}")
            print(f"Sqrt(h*w) = {math.sqrt(img_h * img_w):.2f}")
            
        return tiles
    
    # Calculate number of tiles needed to cover the image with minimum overlap
    # First determine how many tiles we need in each dimension
    num_tiles_x = max(1, math.ceil(img_w / (tile_w * (1 - overlap))))
    num_tiles_y = max(1, math.ceil(img_h / (tile_h * (1 - overlap))))
    
    # Then calculate the actual step size based on image size and number of tiles
    # This ensures even spacing and consistent overlap
    if num_tiles_x > 1:
        step_x = (img_w - tile_w) / (num_tiles_x - 1)
    else:
        step_x = 0
        
    if num_tiles_y > 1:
        step_y = (img_h - tile_h) / (num_tiles_y - 1)
    else:
        step_y = 0
    
    if visualize:
        # Create a copy of the image for visualization
        vis_image = image.copy()
        
        # Print tiling information
        print(f"Original image size: {img_w}x{img_h}")
        print(f"Tile size: {tile_w}x{tile_h}")
        print(f"Sqrt(h*w) = {math.sqrt(tile_h * tile_w):.2f}")
        print(f"Number of tiles: {num_tiles_x}x{num_tiles_y} = {num_tiles_x * num_tiles_y}")
        print(f"Step size: {step_x}x{step_y}")
        print(f"Overlap: {overlap * 100:.1f}%")
        
        # Create a random color for each tile
        colors = np.random.randint(0, 255, size=(num_tiles_x * num_tiles_y, 3), dtype=np.uint8)
        
    # For each position, calculate the start point
    tile_idx = 0
    for y_idx in range(num_tiles_y):
        for x_idx in range(num_tiles_x):
            # Calculate start positions
            if num_tiles_x == 1:
                # Center the single tile in this dimension
                x = max(0, (img_w - tile_w) // 2)
            else:
                # Distribute tiles evenly with calculated step
                x = min(round(x_idx * step_x), max(0, img_w - tile_w))
            
            if num_tiles_y == 1:
                # Center the single tile in this dimension
                y = max(0, (img_h - tile_h) // 2)
            else:
                # Distribute tiles evenly with calculated step
                y = min(round(y_idx * step_y), max(0, img_h - tile_h))
            
            # Calculate end positions, ensuring full tiles
            x_end = min(x + tile_w, img_w)
            y_end = min(y + tile_h, img_h)
            tile = image[y:y_end, x:x_end]
            tiles.append({
                "image": tile,
                "coordinates": (x, y, x_end, y_end)
            })
            
            # Since we're using evenly distributed tiles, we don't need to skip edge tiles
            # They will all be full size unless the image itself is smaller than our tile size
            if visualize:
                # Draw tile rectangle with a random color
                color = colors[tile_idx].tolist()
                cv2.rectangle(vis_image, (x, y), (x_end, y_end), color, 2)
                
                # Add tile number
                cv2.putText(vis_image, f"{tile_idx+1}", (x + 10, y + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # Print tile information
                print(f"Tile {tile_idx+1}: ({x}, {y}) to ({x_end}, {y_end}), size: {x_end-x}x{y_end-y}")
                print(f"  Sqrt(h*w) = {math.sqrt((y_end-y) * (x_end-x)):.2f}")
                
                tile_idx += 1
    
    if visualize:
        # Display the image with tile visualization
        plt.figure(figsize=(12, 10))
        plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Tiling Visualization: {num_tiles_x}x{num_tiles_y} tiles")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Optionally, display some sample tiles
        num_samples = min(4, len(tiles))
        if num_samples > 0:
            fig, axes = plt.subplots(1, num_samples, figsize=(15, 4))
            if num_samples == 1:
                axes = [axes]  # Make it iterable for the loop
                
            for i in range(num_samples):
                sample_tile = tiles[i]["image"]
                axes[i].imshow(cv2.cvtColor(sample_tile, cv2.COLOR_BGR2RGB))
                axes[i].set_title(f"Tile {i+1}: {sample_tile.shape[1]}x{sample_tile.shape[0]}")
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.show()

    return tiles

def apply_nms(boxes, scores, classes, iou_threshold=0.7, containment_threshold=0.5, visualize_before=False, image=None):
    """
    Apply NMS across all detections, with additional filtering for containment.
    Keeps the largest box when containment is detected. For standard NMS, keeps
    the highest confidence box.
    
    Args:
        boxes: Array of bounding boxes in format [x1, y1, x2, y2]
        scores: Array of confidence scores
        classes: Array of class IDs
        iou_threshold: IoU threshold for traditional NMS
        containment_threshold: Threshold for containment filtering. If box A contains
                               more than this fraction of box B, they're considered overlapping.
        visualize_before: Whether to visualize all detections before NMS
        image: Original image for visualization (required if visualize_before=True)
        
    Returns:
        Filtered boxes, scores, and classes
    """
    if len(boxes) == 0:
        return np.zeros((0, 4)), np.array([]), np.array([])
    
    # Visualize all detections before NMS (if requested)
    if visualize_before and image is not None:
        component_names = register_component_dataset()
        visualize_all_detections(image, boxes, classes, scores, component_names, "All Detections Before NMS")
    
    # Convert to format expected by NMS functions and ensure same dtype
    boxes_for_nms = torch.as_tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.as_tensor(scores, dtype=torch.float32)
    classes_tensor = torch.as_tensor(classes)
    
    # Function to calculate containment ratio
    def calculate_containment(box1, box2):
        """
        Calculate what fraction of box2 is contained within box1.
        
        Args:
            box1, box2: Boxes in format [x1, y1, x2, y2]
        
        Returns:
            Containment ratio (area of intersection / area of box2)
        """
        # Calculate intersection coordinates
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        # Calculate intersection area
        if x1_inter < x2_inter and y1_inter < y2_inter:
            inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        else:
            return 0.0
        
        # Calculate area of box2
        box2_area = calculate_area(box2)
        
        # Return containment ratio
        if box2_area > 0:
            return inter_area / box2_area
        else:
            return 0.0
    
    # Function to calculate box area
    def calculate_area(box):
        """Calculate area of a box in format [x1, y1, x2, y2]"""
        return (box[2] - box[0]) * (box[3] - box[1])
    
    # Process each class separately
    keep_indices = []
    for class_id in torch.unique(classes_tensor):
        class_mask = (classes_tensor == class_id)
        class_boxes = boxes_for_nms[class_mask]
        class_scores = scores_tensor[class_mask]
        
        # Get original indices for this class
        original_class_indices = torch.where(class_mask)[0].tolist()
        
        # Skip empty classes
        if len(class_boxes) == 0:
            continue
            
        # First, separate step: Check and prioritize larger boxes for containment
        # Create a list of box indices, areas, and a flag indicating if they should be kept
        box_info = []
        for i in range(len(class_boxes)):
            box = class_boxes[i].tolist()
            area = calculate_area(box)
            box_info.append({
                'index': i,
                'box': box,
                'area': area,
                'keep': True  # Initially mark all boxes to keep
            })
        
        # Sort boxes by area (largest first)
        box_info.sort(key=lambda x: x['area'], reverse=True)
        
        # Check each box against all others for containment
        for i in range(len(box_info)):
            # Skip if this box has already been marked for removal
            if not box_info[i]['keep']:
                continue
                
            box1 = box_info[i]['box']
            
            for j in range(len(box_info)):
                # Skip if comparing to itself or if box j already marked for removal
                if i == j or not box_info[j]['keep']:
                    continue
                    
                box2 = box_info[j]['box']
                
                # Check containment in both directions
                containment1 = calculate_containment(box1, box2)
                containment2 = calculate_containment(box2, box1)
                
                # If containment is detected:
                # - Keep the larger box (already sorted by area)
                # - Mark the smaller box for removal
                if containment1 > containment_threshold or containment2 > containment_threshold:
                    box_info[j]['keep'] = False  # Remove the smaller box (j)
        
        # After containment filtering, get indices of boxes to keep
        containment_filtered_indices = [info['index'] for info in box_info if info['keep']]
        
        # Now apply standard NMS on the remaining boxes
        if containment_filtered_indices:
            # Get boxes and scores that survived containment filtering
            filtered_boxes = class_boxes[containment_filtered_indices]
            filtered_scores = class_scores[containment_filtered_indices]
            
            # Apply traditional NMS
            nms_indices = torchvision.ops.nms(
                filtered_boxes, 
                filtered_scores,
                iou_threshold
            ).tolist()
            
            # Map indices back to original class indices
            for idx in nms_indices:
                original_idx = original_class_indices[containment_filtered_indices[idx]]
                keep_indices.append(original_idx)
    
    # Sort by original score for final output
    if keep_indices:
        keep_indices = sorted(keep_indices, key=lambda i: scores[i], reverse=True)
        
        # Visualize after NMS if requested
        if visualize_before and image is not None:
            filtered_boxes = boxes[keep_indices]
            filtered_scores = scores[keep_indices]
            filtered_classes = classes[keep_indices]
            visualize_all_detections(image, filtered_boxes, filtered_classes, filtered_scores, 
                                    component_names, "Detections After NMS")
            
        return boxes[keep_indices], scores[keep_indices], classes[keep_indices]
    else:
        return np.zeros((0, 4)), np.array([]), np.array([])

def visualize_all_detections(image, boxes, classes, scores, component_names, title="Component Detections"):
    """
    Visualize all component detections in an image.
    
    Args:
        image: The image to visualize on
        boxes: Bounding boxes in format [x1, y1, x2, y2]
        classes: Class IDs
        scores: Confidence scores
        component_names: List of component class names
        title: Title for the visualization
    """
    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        if image.dtype == np.uint8:  # Check if image is in 0-255 range
            img_rgb = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image.copy()  # Already in RGB format
    else:
        img_rgb = image.copy()
    
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(15, 15))
    ax.imshow(img_rgb)
    
    # Count components by class
    class_counter = Counter(classes)
    
    # Draw component detections
    for box, cls, score in zip(boxes, classes, scores):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Generate a random color for this class (but consistent for same class)
        # Use a hash of the class ID to generate a color
        color_seed = int(cls) % 10  # Limit to 10 colors for readability
        color_options = ['r', 'g', 'b', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'pink']
        color = color_options[color_seed]
        
        # Draw rectangle
        rect = patches.Rectangle((x1, y1), width, height, 
                                linewidth=1.5, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # Add label
        class_name = component_names[cls] if cls < len(component_names) else f"Class {cls}"
        label = f"{class_name}: {score:.2f}"
        ax.text(x1, y1-5, label, color='white', backgroundcolor=color,
                fontsize=8, weight='bold')
    
    # Add a legend with component counts
    legend_text = []
    for cls, count in class_counter.items():
        class_name = component_names[cls] if cls < len(component_names) else f"Class {cls}"
        legend_text.append(f"{class_name}: {count}")
    
    # Add count information to the title
    title = f"{title} - Total: {len(boxes)} boxes, {len(class_counter)} classes"
    
    # Add legend at the bottom
    plt.figtext(0.5, 0.01, "\n".join(legend_text), ha="center", fontsize=10,
               bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    # Show image
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_tile_predictions(tile_image, detections, component_names, title="Tile Predictions"):
    """
    Visualize component detections on a single tile.
    
    Args:
        tile_image: The tile image (numpy array)
        detections: Component detection results
        component_names: List of component class names
        title: Title for the visualization
    """
    # Convert BGR to RGB if needed
    if len(tile_image.shape) == 3 and tile_image.shape[2] == 3:
        img_rgb = cv2.cvtColor(tile_image, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = tile_image
    
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img_rgb)
    
    # Draw component detections
    boxes = detections["boxes"]
    classes = detections["classes"]
    scores = detections["scores"]
    
    for box, cls, score in zip(boxes, classes, scores):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Draw rectangle
        rect = patches.Rectangle((x1, y1), width, height, 
                                linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        # Add label
        class_name = component_names[cls] if cls < len(component_names) else f"Class {cls}"
        label = f"{class_name}: {score:.2f}"
        ax.text(x1, y1-5, label, color='white', backgroundcolor='red',
                fontsize=8, weight='bold')
    
    # Show image
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_two_stage_detections(image_path, detection_results):
    """Visualize detection results from the two-stage pipeline"""
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(image)
    
    # Draw PCB regions
    pcb_boxes = detection_results["pcb_boxes"]
    pcb_scores = detection_results["pcb_scores"]
    
    for i, (box, score) in enumerate(zip(pcb_boxes, pcb_scores)):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Draw PCB rectangle with different color
        rect = patches.Rectangle((x1, y1), width, height, 
                                linewidth=3, edgecolor='blue', facecolor='none', 
                                linestyle='--')
        ax.add_patch(rect)
        
        # Add PCB label
        ax.text(x1, y1-10, f"PCB #{i+1}: {score:.2f}", color='white', backgroundcolor='blue',
                fontsize=12, weight='bold')
    
    # Draw component detections
    component_boxes = detection_results["component_boxes"]
    component_classes = detection_results["component_classes"]
    component_scores = detection_results["component_scores"]
    component_names = detection_results["component_names"]
    
    component_counter = Counter(component_classes)

    for box, cls, score in zip(component_boxes, component_classes, component_scores):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Draw rectangle
        rect = patches.Rectangle((x1, y1), width, height, 
                                linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        # Add label
        class_name = component_names[cls] if cls < len(component_names) else f"Class {cls}"
        label = f"{class_name}: {score:.2f}"
        ax.text(x1, y1-5, label, color='white', backgroundcolor='red',
                fontsize=10, weight='bold')

    # Show image
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("inferences/two_stage_detection_results.png")
    plt.show()

    # Print the count of detected components
    print("\n Component Count Summary:")
    for cls, count in component_counter.items():
        class_name = component_names[cls] if cls < len(component_names) else f"Class {cls}"
        print(f"{class_name}: {count}")


def tile_image(image, annotations, overlap=0.2, visualize=False):
    """
    Tile an image and its annotations into smaller patches based on image size.
    Ensures that sqrt(height*width) < 1200 for each tile, and the longest edge is at most 1333px.
    Returns a list of dicts, each representing a tile and its annotations.
    
    Args:
        image: Input image
        annotations: List of annotation dictionaries
        overlap: Overlap between adjacent tiles (0.0 to 1.0)
        visualize: Whether to visualize the tiles
        
    Returns:
        tiles: List of dictionaries containing tile images and annotations
    """
    tiles = []
    img_h, img_w = image.shape[:2]
    image_area = img_h * img_w
    # Calculate the maximum area constraint
    max_area = 1333 * 800
    # Maximum edge length constraint
    max_edge = 1333
    if image_area < max_area and max(img_h,img_w)<max_edge: 
        tiles.append({
            "image": image,
            "annotations": annotations,
            "coordinates": (0, 0, img_w, img_h)
        })
        return tiles
    
    # Calculate optimal tile dimensions based on constraints
    if img_w >= img_h:  # Wider image
        # First, cap the width at max_edge
        tile_w = min(img_w, max_edge)
        # Calculate height based on max area
        tile_h = min(int(max_area / tile_w), img_h)
    else:  # Taller image
        # First, cap the height at max_edge
        tile_h = min(img_h, max_edge)
        # Calculate width based on max area
        tile_w = min(int(max_area / tile_h), img_w)
    
    # Verify our constraint is met for sqrt(h*w) < sqrt(max_area)
    while math.sqrt(tile_h * tile_w) > math.sqrt(max_area):
        # Reduce dimensions proportionally
        scale_factor = math.sqrt(max_area) / math.sqrt(tile_h * tile_w)
        tile_h = int(tile_h * scale_factor)
        tile_w = int(tile_w * scale_factor)
    
    # Ensure shortest edge is at least 640px
    min_edge = min(tile_h, tile_w)
    if min_edge < 640:
        scale_factor = 640 / min_edge
        # Scale up, but maintain constraints
        new_h = int(tile_h * scale_factor)
        new_w = int(tile_w * scale_factor)
        
        # Make sure we don't exceed max_edge
        if new_h > max_edge:
            new_h = max_edge
            new_w = min(int(max_area / new_h), max_edge)
        elif new_w > max_edge:
            new_w = max_edge
            new_h = min(int(max_area / new_w), max_edge)
        
        # Check if sqrt(h*w) constraint is still met
        if math.sqrt(new_h * new_w) <=  math.sqrt(max_area):
            tile_h, tile_w = new_h, new_w
        else:
            # If scaling up would violate our constraint, adjust differently
            if tile_h < tile_w:
                tile_h = 640
                tile_w = min(int(max_area / tile_h), max_edge)
            else:
                tile_w = 640
                tile_h = min(int(max_area / tile_w), max_edge)
    
    # If the image already fits our constraints, use it as one tile
    if img_w <= tile_w and img_h <= tile_h:
        tiles.append({
            "image": image,
            "annotations": annotations,
            "coordinates": (0, 0, img_w, img_h)
        })
        
        if visualize:
            print(f"Image fits in one tile: {img_w}x{img_h}")
            print(f"Sqrt(h*w) = {math.sqrt(img_h * img_w):.2f}")
            
        return tiles
    
    # Calculate number of tiles needed to cover the image with minimum overlap
    # First determine how many tiles we need in each dimension
    num_tiles_x = max(1, math.ceil(img_w / (tile_w * (1 - overlap))))
    num_tiles_y = max(1, math.ceil(img_h / (tile_h * (1 - overlap))))
    
    # Then calculate the actual step size based on image size and number of tiles
    # This ensures even spacing and consistent overlap
    if num_tiles_x > 1:
        step_x = (img_w - tile_w) / (num_tiles_x - 1)
    else:
        step_x = 0
        
    if num_tiles_y > 1:
        step_y = (img_h - tile_h) / (num_tiles_y - 1)
    else:
        step_y = 0
    
    if visualize:

        
        # Create a copy of the image for visualization
        vis_image = image.copy()
        
        # Print tiling information
        print(f"Original image size: {img_w}x{img_h}")
        print(f"Tile size: {tile_w}x{tile_h}")
        print(f"Sqrt(h*w) = {math.sqrt(tile_h * tile_w):.2f}")
        print(f"Number of tiles: {num_tiles_x}x{num_tiles_y} = {num_tiles_x * num_tiles_y}")
        print(f"Step size: {step_x}x{step_y}")
        print(f"Overlap: {overlap * 100:.1f}%")
        
        # Create a random color for each tile
        colors = np.random.randint(0, 255, size=(num_tiles_x * num_tiles_y, 3), dtype=np.uint8)
        
    # For each position, calculate the start point
    tile_idx = 0
    for y_idx in range(num_tiles_y):
        for x_idx in range(num_tiles_x):
            # Calculate start positions
            if num_tiles_x == 1:
                # Center the single tile in this dimension
                x = max(0, (img_w - tile_w) // 2)
            else:
                # Distribute tiles evenly with calculated step
                x = min(round(x_idx * step_x), max(0, img_w - tile_w))
            
            if num_tiles_y == 1:
                # Center the single tile in this dimension
                y = max(0, (img_h - tile_h) // 2)
            else:
                # Distribute tiles evenly with calculated step
                y = min(round(y_idx * step_y), max(0, img_h - tile_h))
            
            # Calculate end positions, ensuring full tiles
            x_end = min(x + tile_w, img_w)
            y_end = min(y + tile_h, img_h)
            
            # Since we're using evenly distributed tiles, we don't need to skip edge tiles
    # They will all be full size unless the image itself is smaller than our tile size
                
            # Extract annotations for this tile
            tile_annos = []
            for anno in annotations:
                bbox = anno["bbox"]
                bbox_x, bbox_y, bbox_w, bbox_h = bbox
                bbox_x2, bbox_y2 = bbox_x + bbox_w, bbox_y + bbox_h

                # Check bbox intersection with tile
                if not (bbox_x2 < x or bbox_x > x_end or bbox_y2 < y or bbox_y > y_end):
                    # Calculate intersection bbox (tile-local coordinates)
                    new_x1 = max(bbox_x, x) - x
                    new_y1 = max(bbox_y, y) - y
                    new_x2 = min(bbox_x2, x_end) - x
                    new_y2 = min(bbox_y2, y_end) - y

                    new_bbox = [new_x1, new_y1, new_x2 - new_x1, new_y2 - new_y1]

                    new_anno = copy.deepcopy(anno)
                    new_anno["bbox"] = new_bbox
                    tile_annos.append(new_anno)

            # Always include the tile even if it has no annotations
            tile_image_crop = image[y:y_end, x:x_end]
            tiles.append({
                "image": tile_image_crop,
                "annotations": tile_annos,
                "coordinates": (x, y, x_end, y_end)  # Store coordinates for easier mapping
            })
            
            if visualize:
                # Draw tile rectangle with a random color
                color = colors[tile_idx].tolist()
                cv2.rectangle(vis_image, (x, y), (x_end, y_end), color, 2)
                
                # Add tile number
                cv2.putText(vis_image, f"{tile_idx+1}", (x + 10, y + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # Print tile information
                print(f"Tile {tile_idx+1}: ({x}, {y}) to ({x_end}, {y_end}), size: {x_end-x}x{y_end-y}")
                print(f"  Sqrt(h*w) = {math.sqrt((y_end-y) * (x_end-x)):.2f}")
                print(f"  Number of annotations: {len(tile_annos)}")
                
                tile_idx += 1
    
    if visualize:
        # Display the image with tile visualization
        plt.figure(figsize=(12, 10))
        plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Tiling Visualization: {num_tiles_x}x{num_tiles_y} tiles")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Optionally, display some sample tiles
        num_samples = min(4, len(tiles))
        if num_samples > 0:
            fig, axes = plt.subplots(1, num_samples, figsize=(15, 4))
            if num_samples == 1:
                axes = [axes]  # Make it iterable for the loop
                
            for i in range(num_samples):
                sample_tile = tiles[i]["image"]
                axes[i].imshow(cv2.cvtColor(sample_tile, cv2.COLOR_BGR2RGB))
                axes[i].set_title(f"Tile {i+1}: {sample_tile.shape[1]}x{sample_tile.shape[0]}")
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.show()

    return tiles