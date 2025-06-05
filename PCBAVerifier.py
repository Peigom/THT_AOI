import time
total_start_time = time.time()
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter, defaultdict
import json
import pandas as pd
from main import three_stage_detection_pipeline, visualize_with_directions
from scipy.optimize import linear_sum_assignment
from PCBA_detector_trainer import get_pcb_detector
from component_detector_trainer import get_component_detector
from EfficientNet_Direction_Classfier import get_direction_classifier



class PCBAVerifier:
    """A class to verify PCBA component detection results against ground truth annotations."""
    
    def __init__(self, annotation_file_path):
        """Initialize the verifier with annotation file path."""
        self.annotation_file_path = annotation_file_path
        with open(annotation_file_path, 'r') as f:
            self.annotations = json.load(f)
        
        # Extract category mapping
        self.categories = {cat['id']: cat['name'] for cat in self.annotations['categories']}
    
    def list_available_images(self):
        """List all available images in the annotation file with their IDs."""
        return [{
            'id': img['id'],
            'file_name': img['file_name'],
            'width': img['width'],
            'height': img['height']
        } for img in self.annotations['images']]
    
    def get_annotations_by_image_id(self, image_id):
        """Get annotations for a specific image by its ID."""
        return [anno for anno in self.annotations['annotations'] if anno['image_id'] == image_id]
    
    def get_image_info_by_id(self, image_id):
        """Get image information by ID."""
        for img in self.annotations['images']:
            if img['id'] == image_id:
                return img
        return None
    
    def _get_image_annotations(self, image_filename):
        """Get annotations for a specific image."""
        image_id = None
        for image in self.annotations['images']:
            if image['file_name'] == image_filename:
                image_id = image['id']
                break
                
        if image_id is None:
            print(f"Warning: No annotations found for image {image_filename}")
            return []
            
        return self.get_annotations_by_image_id(image_id)
    
    def _rotation_to_direction(self, rotation):
        """Convert rotation angle to cardinal direction."""
        # Normalize rotation to 0-360
        rotation = rotation % 360
        
        # Map rotation to cardinal directions
        if 22.5 <= rotation < 67.5: return "NE"
        elif 67.5 <= rotation < 112.5: return "E"
        elif 112.5 <= rotation < 157.5: return "SE"
        elif 157.5 <= rotation < 202.5: return "S"
        elif 202.5 <= rotation < 247.5: return "SW"
        elif 247.5 <= rotation < 292.5: return "W"
        elif 292.5 <= rotation < 337.5: return "NW"
        else: return "N"  # 337.5-360 or 0-22.5
    
    def verify_detection(self, image_path, detection_results, image_id=None):
        """Verify detection results against ground truth annotations."""
        # Get ground truth annotations
        if image_id is not None:
            gt_annotations = self.get_annotations_by_image_id(image_id)
            if not gt_annotations:
                print(f"Warning: No annotations found for image ID {image_id}")
        else:
            # Get annotations by filename
            image_filename = os.path.basename(image_path)
            gt_annotations = self._get_image_annotations(image_filename)
        
        # Extract detection results
        pcb_boxes = detection_results.get("pcb_boxes", [])
        pcb_scores = detection_results.get("pcb_scores", [1.0] * len(pcb_boxes))
        component_boxes = detection_results.get("component_boxes", [])
        component_classes = detection_results.get("component_classes", [])
        component_scores = detection_results.get("component_scores", [])
        component_directions = detection_results.get("component_directions", [None] * len(component_boxes))
        component_names = detection_results.get("component_names", [])
        
        # Extract ground truth PCB boxes
        gt_pcbs = []
        for anno in gt_annotations:
            if anno['category_id'] == 37:  # 37 is PCBA in the dataset
                x, y, w, h = anno['bbox']
                gt_pcbs.append({
                    'id': anno['id'],
                    'bbox': [x, y, x + w, y + h],
                    'annotation': anno
                })
        
        # Step 1: Categorize which components belong to which PCB
        pcba_components = [[] for _ in range(len(pcb_boxes))]
        unassigned_components = []
        
        for i, comp_box in enumerate(component_boxes):
            x1c, y1c, x2c, y2c = comp_box
            component_info = {
                'index': i,
                'box': comp_box,
                'class': component_classes[i] if i < len(component_classes) else 0,
                'score': component_scores[i] if i < len(component_scores) else 0.0,
                'direction': component_directions[i] if i < len(component_directions) else None,
                'name': component_names[component_classes[i]] if component_classes[i] < len(component_names) else f"Class {component_classes[i]}"
            }
            
            # Find which PCB this component belongs to
            assigned = False
            for pcb_idx, pcb_box in enumerate(pcb_boxes):
                x1p, y1p, x2p, y2p = pcb_box
                center_x, center_y = (x1c + x2c) / 2, (y1c + y2c) / 2
                
                # Component is inside PCB if its center is inside
                if (x1p <= center_x <= x2p and y1p <= center_y <= y2p):
                    pcba_components[pcb_idx].append(component_info)
                    assigned = True
                    break
            
            if not assigned:
                unassigned_components.append(component_info)
        
        # Step 2: Map detected PCBs to ground truth PCBs
        pcb_matches = []
        used_gt_pcbs = set()
        
        for det_idx, pcb_box in enumerate(pcb_boxes):
            x1d, y1d, x2d, y2d = pcb_box
            det_center_x, det_center_y = (x1d + x2d) / 2, (y1d + y2d) / 2
            det_center_abs = (det_center_x, det_center_y)
            det_width, det_height = x2d - x1d, y2d - y1d
            det_area = det_width * det_height
            det_aspect = det_width / det_height if det_height > 0 else 0
            
            best_match_idx = -1
            best_match_score = float('-inf')
            
            for gt_idx, gt_pcb in enumerate(gt_pcbs):
                if gt_idx in used_gt_pcbs:
                    continue
                    
                x1g, y1g, x2g, y2g = gt_pcb['bbox']
                gt_center_x, gt_center_y = (x1g + x2g) / 2, (y1g + y2g) / 2
                gt_center_abs = (gt_center_x, gt_center_y)
                gt_width, gt_height = x2g - x1g, y2g - y1g
                gt_area = gt_width * gt_height
                gt_aspect = gt_width / gt_height if gt_height > 0 else 0
                
                # Calculate normalized distance
                distance = np.sqrt((det_center_abs[0] - gt_center_abs[0])**2 + (det_center_abs[1] - gt_center_abs[1])**2)
                avg_pcb_size = (np.sqrt(det_area) + np.sqrt(gt_area)) / 2
                normalized_distance = distance / avg_pcb_size if avg_pcb_size > 0 else float('inf')
                
                # Calculate similarity metrics
                area_ratio = min(det_area, gt_area) / max(det_area, gt_area)
                aspect_diff = abs(det_aspect - gt_aspect)
                aspect_similarity = 1 / (1 + aspect_diff)
                
                # Combined score with weights
                position_score = 1 - min(normalized_distance, 1.0)
                match_score = 0.7 * position_score + 0.01 * area_ratio + 0.3 * aspect_similarity
                
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_match_idx = gt_idx
            
            # If we found a good match
            if best_match_score > 0.5 and best_match_idx >= 0:
                pcb_matches.append((det_idx, best_match_idx))
                used_gt_pcbs.add(best_match_idx)
        
        # Step 3: Perform component verification for each PCB
        all_pcb_results = []
        
        # For each PCB pair, perform verification
        for det_idx, gt_idx in pcb_matches:
            gt_pcb_id = gt_pcbs[gt_idx]['id']
            
            # Filter ground truth components for this PCB
            gt_by_category = defaultdict(list)
            for anno in gt_annotations:
                cat_id = anno['category_id']
                
                # Skip PCBs for components verification
                if cat_id == 37:  # PCBA category
                    continue
                    
                # Check if component belongs to current PCB
                pcb_anno = gt_pcbs[gt_idx]['annotation']
                pcb_x, pcb_y, pcb_w, pcb_h = pcb_anno['bbox']
                comp_x, comp_y, comp_w, comp_h = anno['bbox']
                
                # Component center
                comp_center_x = comp_x + comp_w/2
                comp_center_y = comp_y + comp_h/2
                
                # Check if center is inside PCB
                if not (pcb_x <= comp_center_x <= pcb_x + pcb_w and 
                        pcb_y <= comp_center_y <= pcb_y + pcb_h):
                    continue  # Skip, not on our PCB
            
                # Convert to [x1, y1, x2, y2] format
                x, y, w, h = anno['bbox']
                bbox = [x, y, x + w, y + h]
                
                # Calculate features
                gt_center_x, gt_center_y = x + w/2, y + h/2
                gt_area = w * h
                gt_aspect = w / h if h > 0 else 0
                
                # Normalize position by PCB size
                norm_x = (gt_center_x - pcb_x) / pcb_w if pcb_w > 0 else 0
                norm_y = (gt_center_y - pcb_y) / pcb_h if pcb_h > 0 else 0
                
                gt_by_category[cat_id].append({
                    'bbox': bbox,
                    'matched': False,
                    'annotation': anno,
                    'features': {
                        'center': (gt_center_x, gt_center_y),
                        'center_norm': (norm_x, norm_y),
                        'area': gt_area,
                        'aspect': gt_aspect
                    }
                })
            
            # Get PCB box for component feature calculation
            pcb_box = pcb_boxes[det_idx]
            x1p, y1p, x2p, y2p = pcb_box
            pcb_width, pcb_height = x2p - x1p, y2p - y1p
            
            # Extract components for this PCB
            pcb_components = pcba_components[det_idx]
            
            # Calculate features for detected components
            for comp in pcb_components:
                box = comp['box']
                x1c, y1c, x2c, y2c = box
                center_x, center_y = (x1c + x2c) / 2, (y1c + y2c) / 2
                width, height = x2c - x1c, y2c - y1c
                area = width * height
                aspect = width / height if height > 0 else 0
                
                # Normalize position by PCB size
                norm_x = (center_x - x1p) / pcb_width if pcb_width > 0 else 0
                norm_y = (center_y - y1p) / pcb_height if pcb_height > 0 else 0
                
                comp['features'] = {
                    'center': (center_x, center_y),
                    'center_norm': (norm_x, norm_y),
                    'area': area,
                    'aspect': aspect
                }
            
            # Prepare for matching
            matches, missed_detections, false_positives = [], [], []
            
            # Match components by class
            for cls in set([comp['class'] for comp in pcb_components]):
                # Extract components of this class
                cls_components = [comp for comp in pcb_components if comp['class'] == cls]
                
                # Match ground truth components of the same class
                gt_same_cls = gt_by_category.get(cls + 1, [])  # +1 because COCO categories start at 1
                
                # If no ground truth for this class, all detections are false positives
                if not gt_same_cls:
                    for comp in cls_components:
                        false_positives.append({
                            'component': comp['name'],
                            'detection_box': comp['box'],
                            'score': comp['score'],
                            'detected_direction': comp['direction'],
                            'pcb_id': gt_pcb_id
                        })
                    continue
                
                # Create feature vectors for matching
                gt_features = [
                    {
                        'idx': i,
                        'features': [
                            gt['features']['center_norm'][0],
                            gt['features']['center_norm'][1],
                            np.log(gt['features']['area'] + 1),
                            gt['features']['aspect']
                        ]
                    }
                    for i, gt in enumerate(gt_same_cls) if not gt['matched']
                ]
                
                comp_features = [
                    {
                        'idx': i,
                        'features': [
                            comp['features']['center_norm'][0],
                            comp['features']['center_norm'][1],
                            np.log(comp['features']['area'] + 1),
                            comp['features']['aspect']
                        ]
                    }
                    for i, comp in enumerate(cls_components)
                ]
                
                # Skip if either list is empty
                if not gt_features or not comp_features:
                    continue
                    
                # Weights for different features
                weights = [7, 7, 0.2, 0.1]  # x, y, area, aspect
                
                # Create cost matrix
                cost_matrix = np.zeros((len(comp_features), len(gt_features)))
                for i, comp in enumerate(comp_features):
                    for j, gt in enumerate(gt_features):
                        weighted_dist = sum(weights[k] * abs(comp['features'][k] - gt['features'][k]) for k in range(len(comp['features'])))
                        cost_matrix[i, j] = weighted_dist
                
                # Use Hungarian algorithm for optimal assignment
                comp_indices, gt_indices = linear_sum_assignment(cost_matrix)
                
                # Threshold for acceptable matches
                threshold = 0.7  # Lower is stricter
                
                for comp_idx, gt_idx in zip(comp_indices, gt_indices):
                    cost = cost_matrix[comp_idx, gt_idx]
                    comp = cls_components[comp_features[comp_idx]['idx']]
                    gt = gt_same_cls[gt_features[gt_idx]['idx']]
                    
                    # Accept match if feature distance is low enough 
                    if cost < threshold: 
                        gt['matched'] = True
                        
                        # Get rotation from ground truth
                        gt_rotation = gt['annotation'].get('attributes', {}).get('rotation', 0.0)
                        
                        # Check orientation
                        orientation_match = "N/A"
                        if comp['direction']:
                            gt_direction = self._rotation_to_direction(gt_rotation)
                            orientation_match = "Match" if gt_direction == comp['direction'] else "Mismatch"
                        
                        matches.append({
                            'component': comp['name'],
                            'detection_box': comp['box'],
                            'gt_box': gt['bbox'],
                            'score': comp['score'],
                            'detected_direction': comp['direction'],
                            'gt_rotation': gt_rotation,
                            'orientation_match': orientation_match,
                            'pcb_id': gt_pcb_id,
                            'feature_cost': cost
                        })
                    else:
                        # Reject the match
                        false_positives.append({
                            'component': comp['name'],
                            'detection_box': comp['box'],
                            'score': comp['score'],
                            'detected_direction': comp['direction'],
                            'pcb_id': gt_pcb_id
                        })
                
                # Find unmatched detections
                matched_comp_indices = set(comp_features[idx]['idx'] for idx in comp_indices)
                for i, comp in enumerate(cls_components):
                    if i not in matched_comp_indices:
                        false_positives.append({
                            'component': comp['name'],
                            'detection_box': comp['box'],
                            'score': comp['score'],
                            'detected_direction': comp['direction'],
                            'pcb_id': gt_pcb_id
                        })
            
            # Find missed detections
            for cat_id, gt_boxes in gt_by_category.items():
                for gt in gt_boxes:
                    if not gt['matched']:
                        missed_detections.append({
                            'component': self.categories.get(cat_id, f"Category {cat_id}"),
                            'gt_box': gt['bbox'],
                            'gt_rotation': gt['annotation'].get('attributes', {}).get('rotation', 0.0),
                            'pcb_id': gt_pcb_id
                        })
            
            # Calculate metrics
            pcb_gt_count = sum(len(boxes) for boxes in gt_by_category.values())
            pcb_detection_count = len(pcb_components)
            pcb_matches_count = len(matches)
            
            pcb_precision = pcb_matches_count / pcb_detection_count if pcb_detection_count > 0 else 0
            pcb_recall = pcb_matches_count / pcb_gt_count if pcb_gt_count > 0 else 0
            pcb_f1_score = 2 * pcb_precision * pcb_recall / (pcb_precision + pcb_recall) if (pcb_precision + pcb_recall) > 0 else 0
            
            # Create filtered results for this PCB
            pcb_results = {
                "component_boxes": [comp['box'] for comp in pcb_components],
                "component_classes": [comp['class'] for comp in pcb_components],
                "component_scores": [comp['score'] for comp in pcb_components],
                "component_directions": [comp['direction'] for comp in pcb_components],
                "component_names": component_names,
                "pcb_boxes": [pcb_boxes[det_idx]],
                "pcb_scores": [pcb_scores[det_idx]]
            }
            
            all_pcb_results.append({
                'pcb_id': gt_pcb_id,
                'matches': matches,
                'missed_detections': missed_detections,
                'false_positives': false_positives,
                'metrics': {
                    'total_gt': pcb_gt_count,
                    'total_detected': pcb_detection_count,
                    'correct_detections': pcb_matches_count,
                    'precision': pcb_precision,
                    'recall': pcb_recall,
                    'f1_score': pcb_f1_score
                },
                'pcb_results': pcb_results
            })
        
        # Handle unassigned components
        if unassigned_components:
            # Create unassigned result
            unassigned_result = {
                "component_boxes": [comp['box'] for comp in unassigned_components],
                "component_classes": [comp['class'] for comp in unassigned_components],
                "component_scores": [comp['score'] for comp in unassigned_components],
                "component_directions": [comp['direction'] for comp in unassigned_components],
                "component_names": component_names,
                "pcb_boxes": [],
                "pcb_scores": []
            }
            
            # Find GT components not assigned to any PCB
            gt_used_by_pcbs = set()
            for det_idx, gt_idx in pcb_matches:
                gt_pcb = gt_pcbs[gt_idx]
                pcb_anno = gt_pcb['annotation']
                pcb_x, pcb_y, pcb_w, pcb_h = pcb_anno['bbox']
                
                for anno in gt_annotations:
                    if anno['category_id'] == 37:  # Skip PCBs
                        continue
                    
                    comp_x, comp_y, comp_w, comp_h = anno['bbox']
                    comp_center_x, comp_center_y = comp_x + comp_w/2, comp_y + comp_h/2
                    
                    if (pcb_x <= comp_center_x <= pcb_x + pcb_w and pcb_y <= comp_center_y <= pcb_y + pcb_h):
                        gt_used_by_pcbs.add(anno['id'])
            
            # Unmatched ground truth not belonging to any PCB
            unassigned_gt = []
            for anno in gt_annotations:
                if anno['category_id'] == 37 or anno['id'] in gt_used_by_pcbs:
                    continue
                
                x, y, w, h = anno['bbox']
                unassigned_gt.append({
                    'bbox': [x, y, x+w, y+h],
                    'matched': False,
                    'annotation': anno,
                    'category_id': anno['category_id']
                })
            
            # Treat all unassigned components as false positives
            unassigned_false_pos = [
                {
                    'component': comp['name'],
                    'detection_box': comp['box'],
                    'score': comp['score'],
                    'detected_direction': comp['direction'],
                    'pcb_id': None  # No PCB assignment
                }
                for comp in unassigned_components
            ]
            
            # Find missed detections among unassigned ground truth
            unassigned_missed = [
                {
                    'component': self.categories.get(gt['category_id'], f"Category {gt['category_id']}"),
                    'gt_box': gt['bbox'],
                    'gt_rotation': gt['annotation'].get('attributes', {}).get('rotation', 0.0),
                    'pcb_id': None  # No PCB assignment
                }
                for gt in unassigned_gt if not gt['matched']
            ]
            
            # Add unassigned component result if needed
            if unassigned_false_pos or unassigned_missed:
                unassigned_gt_count = len(unassigned_gt)
                unassigned_det_count = len(unassigned_components)
                unassigned_matches_count = 0  # All are false positives
                
                all_pcb_results.append({
                    'pcb_id': None,  # No PCB assignment
                    'matches': [],
                    'missed_detections': unassigned_missed,
                    'false_positives': unassigned_false_pos,
                    'metrics': {
                        'total_gt': unassigned_gt_count,
                        'total_detected': unassigned_det_count,
                        'correct_detections': unassigned_matches_count,
                        'precision': 0,
                        'recall': 0,
                        'f1_score': 0
                    },
                    'pcb_results': unassigned_result
                })
        
        # Aggregate results
        all_matches = []
        all_missed = []
        all_false_pos = []
        
        for result in all_pcb_results:
            all_matches.extend(result['matches'])
            all_missed.extend(result['missed_detections'])
            all_false_pos.extend(result['false_positives'])
        
        # Calculate overall metrics
        total_gt = len(gt_annotations) - sum(1 for anno in gt_annotations if anno['category_id'] == 37)  # Exclude PCBs
        total_detected = len(detection_results.get("component_boxes", []))
        correct_detections = len(all_matches)
        
        precision = correct_detections / total_detected if total_detected > 0 else 0
        recall = correct_detections / total_gt if total_gt > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'matches': all_matches,
            'missed_detections': all_missed,
            'false_positives': all_false_pos,
            'metrics': {
                'total_gt': total_gt,
                'total_detected': total_detected,
                'correct_detections': correct_detections,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            },
            'per_pcb_results': all_pcb_results
        }

    def component_summary(self, verification_results):
        """Generate a summary of component verification results."""
        # Count components by type
        gt_counts = Counter()
        detected_counts = Counter()
        matched_counts = Counter()
        orientation_matches = Counter()
        orientation_mismatches = Counter()
        
        # Process missed detections and matches
        for miss in verification_results['missed_detections']:
            gt_counts[miss['component']] += 1
        
        for match in verification_results['matches']:
            gt_counts[match['component']] += 1
            matched_counts[match['component']] += 1
            detected_counts[match['component']] += 1
            
            if match['orientation_match'] == "Match":
                orientation_matches[match['component']] += 1
            elif match['orientation_match'] == "Mismatch":
                orientation_mismatches[match['component']] += 1
        
        # Process false positives
        for fp in verification_results['false_positives']:
            detected_counts[fp['component']] += 1
        
        # Compile results
        all_components = set(list(gt_counts.keys()) + list(detected_counts.keys()))
        
        return pd.DataFrame([
            {
                'Component': component,
                'Ground Truth': gt_counts.get(component, 0),
                'Detected': detected_counts.get(component, 0),
                'Matched': matched_counts.get(component, 0),
                'Missed': gt_counts.get(component, 0) - matched_counts.get(component, 0),
                'False Positives': detected_counts.get(component, 0) - matched_counts.get(component, 0),
                'Orientation Match': orientation_matches.get(component, 0),
                'Orientation Mismatch': orientation_mismatches.get(component, 0),
                'Recall': matched_counts.get(component, 0) / gt_counts.get(component, 1),
                'Precision': matched_counts.get(component, 0) / detected_counts.get(component, 1) if detected_counts.get(component, 0) > 0 else 0
            }
            for component in sorted(all_components)
        ])
        
    def generate_reference_image(self, image_path, annotations):
        """Generate a reference image from ground truth annotations."""
        # Load original image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            return np.zeros((800, 800, 3), dtype=np.uint8)
        
        reference_image = image.copy()
        
        # Draw components
        for anno in annotations:
            if anno['category_id'] == 37:  # Skip PCBs
                continue
                
            x, y, w, h = anno['bbox']
            class_name = self.categories.get(anno['category_id'], f"Category {anno['category_id']}")
            
            # Draw bounding box and label
            cv2.rectangle(reference_image, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
            cv2.putText(reference_image, class_name, (int(x), int(y-5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Add direction arrow if rotation available
            if 'attributes' in anno and 'rotation' in anno['attributes']:
                rotation = anno['attributes']['rotation']
                direction = self._rotation_to_direction(rotation)
                
                # Define arrow directions
                arrow_dirs = {
                    "N": (0, -1), "NE": (0.7071, -0.7071), "E": (1, 0), "SE": (0.7071, 0.7071),
                    "S": (0, 1), "SW": (-0.7071, 0.7071), "W": (-1, 0), "NW": (-0.7071, -0.7071)
                }
                
                # Draw arrow
                arrow_length = min(w, h) * 0.6
                center_x, center_y = int(x + w/2), int(y + h/2)
                dx, dy = arrow_dirs.get(direction, (0, 0))
                end_x, end_y = int(center_x + dx * arrow_length), int(center_y + dy * arrow_length)
                cv2.arrowedLine(reference_image, (center_x, center_y), (end_x, end_y), 
                            (0, 255, 0), 2, tipLength=0.3)
        
        # Draw PCB regions
        for anno in annotations:
            if anno['category_id'] == 37:  # PCB regions
                x, y, w, h = anno['bbox']
                
                # Draw dashed PCB rectangle
                dash_length, gap_length = 10, 5
                
                # Draw dashed rectangle (all 4 sides)
                for i in range(0, w, dash_length + gap_length):
                    start_x = int(x + i)
                    end_x = int(min(start_x + dash_length, x + w))
                    cv2.line(reference_image, (start_x, int(y)), (end_x, int(y)), (255, 0, 0), 2)
                
                for i in range(0, h, dash_length + gap_length):
                    start_y = int(y + i)
                    end_y = int(min(start_y + dash_length, y + h))
                    cv2.line(reference_image, (int(x + w), start_y), (int(x + w), end_y), (255, 0, 0), 2)
                
                for i in range(0, w, dash_length + gap_length):
                    start_x = int(x + w - i)
                    end_x = int(max(start_x - dash_length, x))
                    cv2.line(reference_image, (start_x, int(y + h)), (end_x, int(y + h)), (255, 0, 0), 2)
                
                for i in range(0, h, dash_length + gap_length):
                    start_y = int(y + h - i)
                    end_y = int(max(start_y - dash_length, y))
                    cv2.line(reference_image, (int(x), start_y), (int(x), end_y), (255, 0, 0), 2)
                
                # Add PCB label
                cv2.putText(reference_image, f"PCB ID: {anno['id']}", (int(x), int(y-10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Convert BGR to RGB for matplotlib
        return cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)


def visualize_verification(verifier, image_path, reference_image_path, 
                         detection_results, verification_results, title=None):
    """Visualize verification results with side-by-side comparison."""
    # Direction arrow mappings
    arrow_dirs = {
        "N": (0, -1), "NE": (0.7071, -0.7071), "E": (1, 0), "SE": (0.7071, 0.7071),
        "S": (0, 1), "SW": (-0.7071, 0.7071), "W": (-1, 0), "NW": (-0.7071, -0.7071)
    }
    
    # Load images
    inspection_image = cv2.imread(image_path)
    if inspection_image is None:
        print(f"Warning: Could not read image {image_path}")
        return
    inspection_image = cv2.cvtColor(inspection_image, cv2.COLOR_BGR2RGB)
    
    # Get reference image
    if not os.path.exists(reference_image_path):
        print(f"Warning: Reference image not found at {reference_image_path}")
        # Fallback to generating a reference image
        image_filename = os.path.basename(image_path)
        annotations = verifier._get_image_annotations(image_filename)
        reference_image = verifier.generate_reference_image(image_path, annotations)
    else:
        reference_image = cv2.imread(reference_image_path)
        if reference_image is None:
            image_filename = os.path.basename(image_path)
            annotations = verifier._get_image_annotations(image_filename)
            reference_image = verifier.generate_reference_image(image_path, annotations)
        else:
            reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
    
    # Create figure for side-by-side comparison
    fig, (ax_ref, ax_insp) = plt.subplots(1, 2, figsize=(20, 10))
    ax_ref.set_title("Reference Image", fontsize=16)
    ax_insp.set_title("Inspection Image", fontsize=16)
    
    # Display images
    ax_ref.imshow(reference_image)
    ax_insp.imshow(inspection_image)
    
    if title:
        fig.suptitle(title, fontsize=18)
    
    # Draw PCB regions
    pcb_color = 'purple'
    if "pcb_boxes" in detection_results and "pcb_scores" in detection_results:
        for i, (box, score) in enumerate(zip(detection_results["pcb_boxes"], detection_results["pcb_scores"])):
            x1, y1, x2, y2 = box
            width, height = x2 - x1, y2 - y1
            
            # Draw PCB rectangle on inspection image
            rect_insp = patches.Rectangle((x1, y1), width, height, 
                                linewidth=3, edgecolor=pcb_color, facecolor='none', 
                                linestyle='--')
            ax_insp.add_patch(rect_insp)
            ax_insp.text(x1, y1-10, f"PCB #{i+1}: {score:.2f}", color='white', 
                    backgroundcolor=pcb_color, fontsize=12, weight='bold')
    
    # Draw missed detections (red) on reference image
    for miss in verification_results['missed_detections']:
        x1, y1, x2, y2 = miss['gt_box']
        width, height = x2 - x1, y2 - y1
        
        rect = patches.Rectangle((x1, y1), width, height, 
                            linewidth=2, edgecolor='red', facecolor='none',
                            linestyle='--')
        ax_ref.add_patch(rect)
        ax_ref.text(x1, y1-5, f"Missed: {miss['component']}", color='white', 
                backgroundcolor='red', fontsize=5, weight='bold')
    
    # Draw matches (green) on inspection image
    for match in verification_results['matches']:
        x1, y1, x2, y2 = match['detection_box']
        width, height = x2 - x1, y2 - y1
        
        # Set color based on orientation match
        edge_color = {
            "Match": 'green',
            "Mismatch": 'orange',
            "N/A": 'green'
        }.get(match['orientation_match'], 'green')
            
        # Get PCB indicator
        pcb_indicator = f" (Conf {match['score']:.2f})" if 'pcb_id' in match and match['pcb_id'] is not None else ""
            
        rect = patches.Rectangle((x1, y1), width, height, 
                            linewidth=2, edgecolor=edge_color, facecolor='none')
        ax_insp.add_patch(rect)
        
        # Add label with direction info
        label = f"{pcb_indicator}"
        if match['detected_direction']:
            label += f", Dir={match['detected_direction']}"
            
            # Draw direction arrow
            arrow_length = min(width, height) * 0.6
            center_x, center_y = x1 + width/2, y1 + height/2
            dx, dy = arrow_dirs.get(match['detected_direction'], (0, 0))
            ax_insp.arrow(center_x, center_y, dx * arrow_length, dy * arrow_length, 
                head_width=arrow_length/3, head_length=arrow_length/2, 
                fc=edge_color, ec=edge_color, linewidth=2)
                
        if edge_color == 'orange':
            ax_insp.text(x1, y1-5, label, color='white', backgroundcolor=edge_color,
                fontsize=5, weight='bold')
                    
    # Draw false positives (red with dotted line) on inspection image
    for fp in verification_results['false_positives']:
        x1, y1, x2, y2 = fp['detection_box']
        width, height = x2 - x1, y2 - y1
        
        pcb_indicator = f" (Conf {fp['score']:.2f})" if 'pcb_id' in fp and fp['pcb_id'] is not None else ""
        
        rect = patches.Rectangle((x1, y1), width, height, 
                            linewidth=2, edgecolor='red', facecolor='none',
                            linestyle='-.')
        ax_insp.add_patch(rect)
        ax_insp.text(x1, y1-5, f"False: {fp['component']}{pcb_indicator}", 
                color='white', backgroundcolor='red', fontsize=5, weight='bold')
        
        # Draw direction arrow if available
        if fp['detected_direction']:
            arrow_length = min(width, height) * 0.6
            center_x, center_y = x1 + width/2, y1 + height/2
            dx, dy = arrow_dirs.get(fp['detected_direction'], (0, 0))
            ax_insp.arrow(center_x, center_y, dx * arrow_length, dy * arrow_length, 
                head_width=arrow_length/3, head_length=arrow_length/2, 
                fc='blue', ec='blue', linewidth=2)
                
    # Add metrics text
    metrics = verification_results['metrics']
    metrics_text = (
        f"Ground Truth: {metrics['total_gt']}\n"
        f"Detections: {metrics['total_detected']}\n"
        f"Matches: {metrics['correct_detections']}\n"
        f"Precision: {metrics['precision']:.2f}\n"
        f"Recall: {metrics['recall']:.2f}\n"
        f"F1 Score: {metrics['f1_score']:.2f}"
    )
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
    ax_ref.text(0.02, 0.98, metrics_text, transform=ax_ref.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    ax_insp.text(0.02, 0.98, metrics_text, transform=ax_insp.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Add component count summary
    if "component_classes" in detection_results and "component_names" in detection_results:
        component_counter = Counter(detection_results["component_classes"])
        component_names = detection_results["component_names"]
        
        components_text = "Component Count:\n" + "\n".join(
            f"{component_names[cls] if cls < len(component_names) else f'Class {cls}'}: {count}" 
            for cls, count in component_counter.items()
        )
        
        props = dict(boxstyle='round', facecolor='lightgray', alpha=0.7)
        ax_insp.text(0.02, 0.80, components_text, transform=ax_insp.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
    
    # Add legends
    legend_elements_ref = [patches.Patch(facecolor='none', edgecolor='red', linestyle='--', label='Missed Detection')]
    legend_elements_insp = [
        patches.Patch(facecolor='none', edgecolor='purple', linestyle='--', label='PCB Region'),
        patches.Patch(facecolor='none', edgecolor='green', label='Match'),
        patches.Patch(facecolor='none', edgecolor='orange', label='Position Match, Orientation Mismatch'),
        patches.Patch(facecolor='none', edgecolor='red', linestyle='-.', label='False Positive')
    ]
    
    ax_ref.legend(handles=legend_elements_ref, loc='lower right')
    ax_insp.legend(handles=legend_elements_insp, loc='lower right')
    
    # Turn off axes
    ax_ref.axis('off')
    ax_insp.axis('off')
    
    plt.tight_layout()
    plt.show()


def display_available_images(annotation_file):
    """Display all available images in the annotation file."""
    verifier = PCBAVerifier(annotation_file)
    images = verifier.list_available_images()
    
    print("\nAvailable Images in Annotation File:")
    print("ID\tFilename\t\tDimensions")
    print("-" * 60)
    for img in images:
        print(f"{img['id']}\t{img['file_name']}\t{img['width']}x{img['height']}")


def verify_pcba(image_path,pcb_detector, component_detector,
                direction_classifier, annotation_file, use_tiling=True, 
                image_id=None, reference_image_dir="datasets/label_images"):
    """End-to-end PCBA verification with side-by-side visualization."""
    # Create verifier
    verifier = PCBAVerifier(annotation_file)
    
    # Find reference image if image_id is provided
    reference_image_path = None
    if image_id is not None:
        image_info = verifier.get_image_info_by_id(image_id)
        if image_info:
            reference_filename = image_info['file_name']
            reference_image_path = os.path.join(reference_image_dir, reference_filename)
            print(f"Using reference image: {reference_image_path}")
    

    # Run detection pipeline
    detection_results = three_stage_detection_pipeline(
    image_path, 
    pcb_detector=pcb_detector,
    component_detector=component_detector,
    direction_classifier=direction_classifier,
    use_tiling=use_tiling,
    overlap=0.2
)
    
    # Verify detection
    verification_results = verifier.verify_detection(image_path, detection_results, image_id=image_id)
    
    # Generate component summary
    summary_df = verifier.component_summary(verification_results)
    
    # Set title for visualization
    title = None
    if image_id:
        image_info = verifier.get_image_info_by_id(image_id)
        if image_info:
            title = f"ID_{image_id}_{image_info['file_name']}"
    else:
        title = f"Verification of {os.path.basename(image_path)}"
    total_detection_time = time.time() - total_detection_start_time
    print(f"Total pipeline time: {total_detection_time:.3f} seconds")
    # Use visualization
    if reference_image_path:
        visualize_verification(
            verifier, image_path, reference_image_path,
            detection_results, verification_results, title=title
        )
    else:
        # Fallback to original method if no reference image path was found
        visualize_with_directions(image_path, detection_results)
    
    return verification_results, summary_df


if __name__ == "__main__":
    # Paths
    default_image_path = "datasets/unlabel_images/WIN_20250325_10_32_03_Pro.jpg"
    pcb_model_path = "output/pcb_detector/model_0009999.pth"
    component_model_path = "output/component_detector/model_0014999.pth"
    direction_model_path = "output/direction/efficient_direction_classifier.pth"
    annotation_file = "datasets/pcba_and_component_adjusted.json"
    reference_image_dir = "datasets/label_images"
    
    # Get cached models (loaded only once)
    pcb_detector = get_pcb_detector(pcb_model_path)
    component_detector = get_component_detector(component_model_path)
    direction_classifier = get_direction_classifier(direction_model_path)
    
    print("PCBA Verification Tool")
    print("=====================")
    print("Enter 'q' at any prompt to exit")
    
    running = True
    total_time = time.time() - total_start_time
    print(f"Total load time: {total_time:.3f} seconds") 
    while running:
        # Ask user for image path or ID
        image_input = input(f"\nEnter image path or press Enter to use default path ({default_image_path}) or press q to quit: ")
        if image_input.lower() == 'q':
            running = False
            print("Exiting program.")
            break

        # Set image path (default or user-provided)
        image_path = image_input if image_input else default_image_path
        
        # Check if image exists
        while not os.path.exists(image_path):
            print(f"Error: Image path '{image_path}' does not exist.")
            image_input = input("Please enter a valid image path or press Enter to use default path or 'q' to quit: ")
            
            if image_input.lower() == 'q':
                running = False
                break
            
            # Only use default path if user presses Enter without typing anything
            image_path = image_input if image_input else default_image_path
        
        # If user chose to quit during image path validation, break the main loop
        if not running:
            print("Exiting program.")
            break
            
        print(f"Using image: {image_path}")
        
        # Display available images 
        display_available_images(annotation_file)
        
        # Ask user for image ID
        id_input = input("\nEnter the image ID to use for verification (or press Enter to skip): ")
        if id_input.lower() == 'q':
            running = False
            print("Exiting program.")
            break
            
        # Process image ID
        try:
            image_id = int(id_input) if id_input else None
        except ValueError:
            print("Invalid input. Using filename matching instead.")
            image_id = None
        
        # Verify PCBA - we've already validated the image path exists
        total_detection_start_time = time.time()
        print("Image exists!")
        print("\nProcessing image...\n")
        
        # Process the image with pre-loaded models
        try:
            results, summary = verify_pcba(
                image_path, 
                pcb_detector=pcb_detector,
                component_detector=component_detector,
                direction_classifier=direction_classifier,
                annotation_file=annotation_file, 
                image_id=image_id, 
                reference_image_dir=reference_image_dir
            )
            
            # Print summary table
            if image_id is not None:
                print("\nComponent Verification Summary:")
                pd.set_option('display.max_rows', None)
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', 1000)
                print(summary)
        
        except Exception as e:
            print(f"Error processing image: {e}")
            print("Continuing to next image...")
            continue
        
        # Ask if user wants to continue
        continue_input = input("\nProcess another image? (y/n): ")
        if continue_input.lower() != 'y':
            running = False
            print("Exiting program.")
