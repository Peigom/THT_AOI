import torch.multiprocessing as mp
import numpy as np
import os
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
from tiling import tile_image_inference, apply_nms
from dataset_creation import create_pcb_only_annotations, create_component_only_annotations
from Pretty_json import pretty_json
from PCBA_detector_trainer import train_pcb_detector, get_pcb_detector
from component_detector_trainer import train_component_detector, register_component_dataset, get_component_detector
from EfficientNet_Direction_Classfier import train_efficient_direction_classifier, get_direction_classifier


def visualize_with_directions(image_path, detection_results):
    """Visualize detection results with direction predictions"""
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
        
        # Draw PCB rectangle
        rect = patches.Rectangle((x1, y1), width, height, 
                                linewidth=3, edgecolor='blue', facecolor='none', 
                                linestyle='--')
        ax.add_patch(rect)
        
        # Add PCB label
        ax.text(x1, y1-10, f"PCB #{i+1}: {score:.2f}", color='white', backgroundcolor='blue',
                fontsize=12, weight='bold')
    
    # Draw component detections with directions
    component_boxes = detection_results["component_boxes"]
    component_classes = detection_results["component_classes"]
    component_scores = detection_results["component_scores"]
    component_names = detection_results["component_names"]
    component_directions = detection_results.get("component_directions", [None] * len(component_classes))
    direction_confidences = detection_results.get("direction_confidences", [0.0] * len(component_classes))

    component_counter = Counter(component_classes)
    
    # Direction-specific colors
    direction_colors = {
        "N": "green",
        "NE": "yellowgreen",
        "E": "purple",
        "SE": "orange",
        "S": "red",
        "SW": "pink",
        "W": "blue",
        "NW": "deepskyblue"
    }
    
    for i, (box, cls, score) in enumerate(zip(component_boxes, component_classes, component_scores)):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        class_name = component_names[cls] if cls < len(component_names) else f"Class {cls}"
        direction = component_directions[i]
        
        # Use different color for components with direction
        if direction:
            edge_color = direction_colors.get(direction, 'r')
            rect = patches.Rectangle((x1, y1), width, height, 
                                    linewidth=2, edgecolor=edge_color, facecolor='none')
            ax.add_patch(rect)
            
            # Draw an arrow to indicate direction
            arrow_length = min(width, height) * 0.6
            center_x = x1 + width/2
            center_y = y1 + height/2
            
            # Define arrow directions
            arrow_directions = {
                "N": (0, -1),
                "NE": (-0.7071, -0.7071),
                "E": (1, 0),
                "SE": (0.7071, 0.7071),
                "S": (0, 1),
                "SW": (-0.7071, 0.7071),
                "W": (-1, 0),
                "NW": (-0.7071, -0.7071)
            }
            
            # Get direction vector
            dx, dy = arrow_directions.get(direction, (0, 0))
            ax.arrow(center_x, center_y, dx * arrow_length, dy * arrow_length, 
                   head_width=arrow_length/3, head_length=arrow_length/2, 
                   fc=edge_color, ec=edge_color, linewidth=2)
            
            # Add label with direction
            direction_conf = direction_confidences[i]
            label = f"{class_name}: {score:.2f} | {direction} ({direction_conf:.2f})"
            ax.text(x1, y1-5, label, color='white', backgroundcolor=edge_color,
                    fontsize=10, weight='bold')
        else:
            # Regular component without direction
            rect = patches.Rectangle((x1, y1), width, height, 
                                    linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            
            # Add regular labels
            label = f"{class_name}: {score:.2f}"
            ax.text(x1, y1-5, label, color='white', backgroundcolor='red',
                    fontsize=10, weight='bold')
    
    # Show image
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("inferences/detection_with_directions.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Print the count of detected components
    print("\n Component Count Summary:")
    for cls, count in component_counter.items():
        class_name = component_names[cls] if cls < len(component_names) else f"Class {cls}"
        print(f"{class_name}: {count}")


def crop_pcb_regions(image, pcb_boxes, padding=-20):
    """
    Crop PCB regions from the image with padding
    
    Args:
        image_path: Path to the input image
        pcb_detections: PCB detection results from detect_pcb_regions
        padding: Padding around PCB regions in pixels
        
    Returns:
        list: Cropped PCB images and their coordinates in the original image
    """
    cropped_pcbs = []
    
    for i, box in enumerate(pcb_boxes):
        # Get coordinates
        x1, y1, x2, y2 = map(int, box)
        
        # Apply padding but ensure we stay within image bounds
        height, width = image.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)
        
        # Crop PCB region
        pcb_image = image[y1:y2, x1:x2]
        
        # Store cropped image and its coordinates
        cropped_pcbs.append({
            "image": pcb_image,
            "coordinates": (x1, y1, x2, y2)
        })
    
    return cropped_pcbs

def three_stage_detection_pipeline(image_path, pcb_detector=None, component_detector=None, 
                                 direction_classifier=None, overlap=0.2, use_tiling=True):
    """Complete three-stage detection pipeline
    
    Args:
        image_path: Path to the input image
        pcb_detector: Loaded PCB detector model (optional)
        component_detector: Loaded component detector model (optional)
        direction_classifier: Loaded direction classifier model (optional)
        overlap: Overlap factor for tiling (0.0 to 1.0)
        use_tiling: Whether to use tiling for component detection
    
    Returns:
        dict: Detection results
    """
    
    # Load PCB detector if not provided
    if pcb_detector is None:
        pcb_detector = get_pcb_detector("output/pcb_detector/model_final.pth")  # Use default path
    
    # Read image for PCB detection
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return None
    
    # Run PCB detection directly
    outputs = pcb_detector(image)
    
    # Extract PCB regions
    instances = outputs["instances"].to("cpu")
    pcb_boxes = instances.pred_boxes.tensor.numpy()
    pcb_scores = instances.scores.numpy()
    
    # If no PCB detected, use whole image
    if len(pcb_boxes) == 0:
        pcb_boxes = np.array([[0, 0, image.shape[1], image.shape[0]]])
        pcb_scores = np.array([1.0])
    
    # Create list to store cropped PCBs
    cropped_pcbs = crop_pcb_regions(image, pcb_boxes)
    
    # Load component detector if not provided
    if component_detector is None:
        component_detector = get_component_detector("output/component_detector/model_final.pth")  # Use default path
    
    # Get component names
    component_names = register_component_dataset()
    
    # Get component detections across all PCBs
    all_component_boxes = []
    all_component_classes = []
    all_component_scores = []
    
    for pcb_region in cropped_pcbs:
        pcb_image = pcb_region["image"]
        pcb_coords = pcb_region["coordinates"]
        
        if use_tiling:
            # Tile the PCB image for component detection
            tiles = tile_image_inference(pcb_image, overlap=overlap, visualize=False)
            
            # Detect components in each tile
            for tile in tiles:
                tile_image = tile["image"]
                tile_coords_in_pcb = tile["coordinates"]
                
                # Run component detection directly
                tile_outputs = component_detector(tile_image)
                tile_instances = tile_outputs["instances"].to("cpu")
                tile_boxes = tile_instances.pred_boxes.tensor.numpy()
                tile_classes = tile_instances.pred_classes.numpy()
                tile_scores = tile_instances.scores.numpy()
                
                # Map coordinates back to original image
                for box, cls, score in zip(tile_boxes, tile_classes, tile_scores):
                    x1, y1, x2, y2 = box
                    # Map from tile to PCB to original image coordinates
                    x1_orig = pcb_coords[0] + tile_coords_in_pcb[0] + x1
                    y1_orig = pcb_coords[1] + tile_coords_in_pcb[1] + y1
                    x2_orig = pcb_coords[0] + tile_coords_in_pcb[0] + x2
                    y2_orig = pcb_coords[1] + tile_coords_in_pcb[1] + y2
                    
                    all_component_boxes.append([x1_orig, y1_orig, x2_orig, y2_orig])
                    all_component_classes.append(cls)
                    all_component_scores.append(score)
        else:
            # Process the entire PCB without tiling
            outputs = component_detector(pcb_image)
            instances = outputs["instances"].to("cpu")
            component_boxes = instances.pred_boxes.tensor.numpy()
            component_classes = instances.pred_classes.numpy()
            component_scores = instances.scores.numpy()
            
            # Map coordinates back to original image
            for box, cls, score in zip(component_boxes, component_classes, component_scores):
                x1, y1, x2, y2 = box
                x1_orig = pcb_coords[0] + x1
                y1_orig = pcb_coords[1] + y1
                x2_orig = pcb_coords[0] + x2
                y2_orig = pcb_coords[1] + y2
                
                all_component_boxes.append([x1_orig, y1_orig, x2_orig, y2_orig])
                all_component_classes.append(cls)
                all_component_scores.append(score)
    
    # Convert to numpy arrays for NMS
    all_component_boxes = np.array(all_component_boxes) if all_component_boxes else np.zeros((0, 4))
    all_component_classes = np.array(all_component_classes)
    all_component_scores = np.array(all_component_scores)
    
    # Apply NMS to remove duplicates
    filtered_boxes, filtered_scores, filtered_classes = apply_nms(
        all_component_boxes,
        all_component_scores,
        all_component_classes,
        iou_threshold=0.7,
        visualize_before=False,
        image=image
    )
    
    # Load direction classifier if not provided
    if direction_classifier is None:
        direction_classifier = get_direction_classifier("output/direction/efficient_direction_classifier.pth")  # Use default path
    
    # Initialize arrays for direction predictions
    direction_predictions = [None] * len(filtered_classes)
    direction_confidences = [0.0] * len(filtered_classes)
    
    # Target classes for direction classification (capacitor electrolytic = 3, connector = 17)
    target_categories = [3, 17]
    
    # Classify directions for target components
    for i, (box, class_id, score) in enumerate(zip(filtered_boxes, filtered_classes, filtered_scores)):
        if class_id in target_categories:
            # Extract component image
            x1, y1, x2, y2 = map(int, box)
            
            # Ensure bbox is within image bounds
            height, width = image.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            
            # Crop component
            component_image = image[y1:y2, x1:x2]
            
            # Handle empty crops (should be rare, but just in case)
            if component_image.size == 0:
                component_image = np.zeros((64, 64, 3), dtype=np.uint8)
            
            # Predict direction
            direction, confidence = direction_classifier.predict_direction(component_image)
            
            # Store prediction
            direction_predictions[i] = direction
            direction_confidences[i] = confidence

    # Return complete detection results
    return {
        "pcb_boxes": pcb_boxes,
        "pcb_scores": pcb_scores,
        "component_boxes": filtered_boxes,
        "component_classes": filtered_classes,
        "component_scores": filtered_scores,
        "component_names": component_names,
        "component_directions": direction_predictions,
        "direction_confidences": direction_confidences
    }

if __name__ == "__main__":
    # Fix for Windows
    if os.name == 'nt':
        mp.set_start_method('spawn', force=True)
    
    # Choose operation mode
    train_pcb_model = False
    train_component_model = False
    train_direction_model = True 
    Resume = False
    inference = True
    
    # Paths
    original_annotations = "datasets/pcba_and_component_adjusted.json"
    pcb_only_annotations = "datasets/PCBA_data.json"
    component_only_annotations = "datasets/pcba_crops/pcba_components.json"
    direction_data_path = "datasets/crop_labels_balanced.json"  # Your existing JSON file

    # Create annotation files for separate training if they don't exist
    if not os.path.exists(pcb_only_annotations):
        create_pcb_only_annotations(original_annotations, pcb_only_annotations)
        pretty_json(pcb_only_annotations)
    if not os.path.exists(component_only_annotations):
        create_component_only_annotations(original_annotations, component_only_annotations)
        pretty_json(component_only_annotations)
   
    # Train PCB detector (stage 1)
    if train_pcb_model:
        pcb_checkpoint = "output/pcb_detector/model_0004999.pth"
        if os.path.exists(pcb_checkpoint) and Resume:
            train_pcb_detector(resume_from=pcb_checkpoint)
        else:
            train_pcb_detector()
    
    # Train component detector (stage 2)*
    if train_component_model:
        component_checkpoint = "output/component_detector/model_0009999.pth"
        if os.path.exists(component_checkpoint) and Resume:
            train_component_detector(resume_from=component_checkpoint)
        else:
            train_component_detector()
    
    # Train direction classifier (stage 3)
    if train_direction_model:
        direction_model_path = "output/direction/direction_classifier.pth"
        image_dir = "datasets/crops_augmented"  # Directory with your component crops
        
        if os.path.exists(direction_model_path):
            print(f"Direction classifier model already exists at {direction_model_path}. Skipping training.")
        else:
            print("Training direction classifier...")
            # Train the model
            model, report = train_efficient_direction_classifier(
                json_file=direction_data_path,
                image_dir=image_dir,
                model_name="efficientnet_b0", # You can try b1, b2, etc. for better performance
                batch_size=32,
                num_epochs=20,
                learning_rate=0.001,
                use_validation=False
            )
    
    # Run inference
    if inference:
        image_path = "datasets/label_images/1733735229.jpg"#C:\Users\Tuoma\Desktop\RPN\PracticalThesis\datasets\pcba_crops\images\1733735229_pcba_1.jpg
        pcb_model_path = "output/pcb_detector/model_0009999.pth"
        component_model_path = "output/component_detector/model_0014999.pth"
        direction_model_path = "output/direction/efficient_direction_classifier.pth"
        results = three_stage_detection_pipeline(
                        image_path, 
                        pcb_model_path, 
                        component_model_path, 
                        direction_model_path,
                        use_tiling=True,
                        overlap=0.2)
        visualize_with_directions(image_path, results)
