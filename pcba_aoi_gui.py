import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle, Patch
from matplotlib.patches import Rectangle
import queue
import gc
import tempfile
from collections import Counter, defaultdict
# Import your existing modules
from main import three_stage_detection_pipeline
from PCBAVerifier import PCBAVerifier
from PCBA_detector_trainer import get_pcb_detector
from component_detector_trainer import get_component_detector
from EfficientNet_Direction_Classfier import get_direction_classifier


class PCBAAOIApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("PCBA AOI Inspection System")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        self.root.state('zoomed')
        
        # Model management for memory optimization
        self.models_loaded = False
        self.pcb_detector = None
        self.component_detector = None
        self.direction_classifier = None
        
        # Camera management
        self.camera = None
        self.camera_active = False
        
        # Current image paths for visualization
        self.current_image_path = None
        self.inspection_image_path = None
        self.comparison_image_path = None
        
        # Reference data
        self.annotation_file = "datasets/pcba_and_component_adjusted.json"
        self.reference_images = []
        self.selected_reference_id = None
        
        # Results storage
        self.last_verification_results = None
        self.last_summary_df = None
        self.last_detection_results = None  # Store detection results for export
        
        # Threading
        self.result_queue = queue.Queue()
        
        # Create GUI
        self.create_widgets()
        self.load_reference_data()
        
        # Check for updates from background threads
        self.root.after(100, self.check_queue)
    
    def close_app(self):
        self.root.destroy()

    def export_bounding_boxes_to_json(self):
        """Export bounding boxes from last detection to JSON file"""
        # This method is no longer needed since save_results handles COCO format
        pass

    def _draw_dashed_line(self, image, pt1, pt2, color, thickness, dash_length, gap_length):
        """Draw a dashed line on the image"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Calculate line length and direction
        length = int(np.sqrt((x2-x1)**2 + (y2-y1)**2))
        if length == 0:
            return
        
        # Unit vector
        dx = (x2 - x1) / length
        dy = (y2 - y1) / length
        
        # Draw dashed line
        pos = 0
        while pos < length:
            # Start of dash
            start_x = int(x1 + dx * pos)
            start_y = int(y1 + dy * pos)
            
            # End of dash
            end_pos = min(pos + dash_length, length)
            end_x = int(x1 + dx * end_pos)
            end_y = int(y1 + dy * end_pos)
            
            # Draw dash
            cv2.line(image, (start_x, start_y), (end_x, end_y), color, thickness)
            
            # Move to next dash
            pos += dash_length + gap_length

    def create_inspection_visualization(self, detection_results, verification_results=None):
        """Create visualization of inspection results with detections overlaid"""
        try:
            if not self.current_image_path:
                return None
            
            # Load the original image
            image = cv2.imread(self.current_image_path)
            if image is None:
                return None
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Direction arrow mappings
            arrow_dirs = {
                "N": (0, -1), "NE": (0.7071, -0.7071), "E": (1, 0), "SE": (0.7071, 0.7071),
                "S": (0, 1), "SW": (-0.7071, 0.7071), "W": (-1, 0), "NW": (-0.7071, -0.7071)
            }
            
            # Create figure for inspection visualization
            fig, ax = plt.subplots(1, figsize=(20, 15))
            ax.set_title("Inspection Results", fontsize=20)
            
            # Display image
            ax.imshow(image_rgb)
            
            # Draw PCB regions
            pcb_color = 'purple'
            if "pcb_boxes" in detection_results and "pcb_scores" in detection_results:
                for i, (box, score) in enumerate(zip(detection_results["pcb_boxes"], detection_results["pcb_scores"])):
                    x1, y1, x2, y2 = box
                    width, height = x2 - x1, y2 - y1
                    
                    # Draw PCB rectangle
                    rect = Rectangle((x1, y1), width, height, 
                                        linewidth=3, edgecolor=pcb_color, facecolor='none', 
                                        linestyle='--')
                    ax.add_patch(rect)
                    ax.text(x1, y1-10, f"PCB #{i+1}: {score:.2f}", color='white', 
                            backgroundcolor=pcb_color, fontsize=12, weight='bold')
            
            # Draw component detections based on verification results
            if verification_results:
                # Draw matches (green) 
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
                        
                    rect = Rectangle((x1, y1), width, height, 
                                        linewidth=2, edgecolor=edge_color, facecolor='none')
                    ax.add_patch(rect)
                    
                    # Add label with direction info
                    label = f"{pcb_indicator}"
                    if match['detected_direction']:
                        label += f", Dir={match['detected_direction']}"
                        
                        # Draw direction arrow
                        arrow_length = min(width, height) * 0.6
                        center_x, center_y = x1 + width/2, y1 + height/2
                        dx, dy = arrow_dirs.get(match['detected_direction'], (0, 0))
                        ax.arrow(center_x, center_y, dx * arrow_length, dy * arrow_length, 
                            head_width=arrow_length/3, head_length=arrow_length/2, 
                            fc=edge_color, ec=edge_color, linewidth=2)
                            
                    if edge_color == 'orange':
                        ax.text(x1, y1-5, label, color='white', backgroundcolor=edge_color,
                            fontsize=8, weight='bold')
                                
                # Draw false positives (red with dotted line)
                for fp in verification_results['false_positives']:
                    x1, y1, x2, y2 = fp['detection_box']
                    width, height = x2 - x1, y2 - y1
                    
                    pcb_indicator = f" (Conf {fp['score']:.2f})" if 'pcb_id' in fp and fp['pcb_id'] is not None else ""
                    
                    rect = Rectangle((x1, y1), width, height, 
                                        linewidth=2, edgecolor='red', facecolor='none',
                                        linestyle='-.')
                    ax.add_patch(rect)
                    ax.text(x1, y1-5, f"False: {fp['component']}{pcb_indicator}", 
                            color='white', backgroundcolor='red', fontsize=8, weight='bold')
                    
                    # Draw direction arrow if available
                    if fp['detected_direction']:
                        arrow_length = min(width, height) * 0.6
                        center_x, center_y = x1 + width/2, y1 + height/2
                        dx, dy = arrow_dirs.get(fp['detected_direction'], (0, 0))
                        ax.arrow(center_x, center_y, dx * arrow_length, dy * arrow_length, 
                            head_width=arrow_length/3, head_length=arrow_length/2, 
                            fc='blue', ec='blue', linewidth=2)
            else:
                # No verification results - draw all detections as green
                component_boxes = detection_results.get("component_boxes", [])
                component_classes = detection_results.get("component_classes", [])
                component_scores = detection_results.get("component_scores", [])
                component_names = detection_results.get("component_names", [])
                component_directions = detection_results.get("component_directions", [None] * len(component_boxes))
                
                for i, (box, cls, score) in enumerate(zip(component_boxes, component_classes, component_scores)):
                    x1, y1, x2, y2 = box
                    width, height = x2 - x1, y2 - y1
                    center_x, center_y = x1 + width/2, y1 + height/2
                    
                    class_name = component_names[cls] if cls < len(component_names) else f"Class {cls}"
                    direction = component_directions[i] if i < len(component_directions) else None
                    
                    # Draw rectangle in green
                    rect = Rectangle((x1, y1), width, height, 
                                        linewidth=2, edgecolor='green', facecolor='none')
                    ax.add_patch(rect)
                    
                    # Add label
                    label = f"{class_name}: {score:.2f}"
                    if direction:
                        label += f" ({direction})"
                        
                        # Draw direction arrow
                        arrow_length = min(width, height) * 0.6
                        dx, dy = arrow_dirs.get(direction, (0, 0))
                        ax.arrow(center_x, center_y, dx * arrow_length, dy * arrow_length, 
                            head_width=arrow_length/3, head_length=arrow_length/2, 
                            fc='green', ec='green', linewidth=2)
                    
                    ax.text(x1, y1-5, label, color='white', backgroundcolor='green',
                            fontsize=8, weight='bold')
            
            # Add metrics text if verification results available
            if verification_results:
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
                ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
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
                ax.text(0.02, 0.80, components_text, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=props)
            
            # Add legend
            if verification_results:
                legend_elements = [
                    Patch(facecolor='none', edgecolor='purple', linestyle='--', label='PCB Region'),
                    Patch(facecolor='none', edgecolor='green', label='Match'),
                    Patch(facecolor='none', edgecolor='orange', label='Position Match, Orientation Mismatch'),
                    Patch(facecolor='none', edgecolor='red', linestyle='-.', label='False Positive')
                ]
            else:
                legend_elements = [
                    Patch(facecolor='none', edgecolor='purple', linestyle='--', label='PCB Region'),
                    Patch(facecolor='none', edgecolor='green', label='Detection')
                ]
            
            ax.legend(handles=legend_elements, loc='lower right')
            
            # Turn off axes
            ax.axis('off')
            
            plt.tight_layout()
            
            # Save visualization to temporary file
            temp_dir = tempfile.gettempdir()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.inspection_image_path = os.path.join(temp_dir, f"inspection_viz_{timestamp}.png")
            
            plt.savefig(self.inspection_image_path, dpi=200, bbox_inches='tight')
            plt.close()  # Important: close the figure to free memory
            
            return self.inspection_image_path
            
        except Exception as e:
            print(f"Error creating inspection visualization: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    
    def create_reference_visualization_only(self, detection_results, verification_results):
        """Create visualization showing only the reference image and missed detections."""
        try:
            if not self.current_image_path or self.selected_reference_id is None:
                print(f"Debug: current_image_path={self.current_image_path}, selected_reference_id={self.selected_reference_id}")
                return None
            
            # Create verifier
            verifier = PCBAVerifier(self.annotation_file)
            
            # Get reference image info
            image_info = verifier.get_image_info_by_id(self.selected_reference_id)
            if not image_info:
                print(f"Debug: No image info found for ID {self.selected_reference_id}")
                return None
            
            reference_image_dir = "datasets/label_images"
            reference_filename = image_info['file_name']
            reference_image_path = os.path.join(reference_image_dir, reference_filename)
            
            print(f"Debug: Looking for reference image at: {reference_image_path}")
            
            # Try to load actual reference image
            reference_image = None
            if os.path.exists(reference_image_path):
                reference_image = cv2.imread(reference_image_path)
                if reference_image is not None:
                    reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
                    print("Debug: Loaded actual reference image")
                else:
                    print("Debug: Reference image file exists but could not be loaded")
            
            # Fallback to generating it from annotations
            if reference_image is None:
                print("Debug: Generating reference image from annotations")
                annotations = verifier.get_annotations_by_image_id(self.selected_reference_id)
                reference_image = verifier.generate_reference_image(self.current_image_path, annotations)

            # Create figure for only the reference image
            fig, ax_ref = plt.subplots(figsize=(20, 15))
            ax_ref.set_title("Reference Image (Ground Truth)", fontsize=16)
            ax_ref.imshow(reference_image)
            
            # Draw missed detections
            for miss in verification_results['missed_detections']:
                x1, y1, x2, y2 = miss['gt_box']
                width, height = x2 - x1, y2 - y1
                
                rect = Rectangle((x1, y1), width, height,
                                linewidth=2, edgecolor='red', facecolor='none',
                                linestyle='--')
                ax_ref.add_patch(rect)
                ax_ref.text(x1, y1 - 5, f"Missed: {miss['component']}", color='white',
                            backgroundcolor='red', fontsize=8, weight='bold')
            
            # Metrics
            metrics = verification_results['metrics']
            metrics_text = (
                f"Ground Truth: {metrics['total_gt']}\n"
                f"Detections: {metrics['total_detected']}\n"
                f"Matches: {metrics['correct_detections']}\n"
                f"Precision: {metrics['precision']:.2f}\n"
                f"Recall: {metrics['recall']:.2f}\n"
                f"F1 Score: {metrics['f1_score']:.2f}"
            )
            ax_ref.text(0.02, 0.98, metrics_text, transform=ax_ref.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

            # Legend
            legend_elements = [Patch(facecolor='none', edgecolor='red', linestyle='--', label='Missed Detection')]
            ax_ref.legend(handles=legend_elements, loc='lower right')
            
            ax_ref.axis('off')
            plt.tight_layout()
            
            # Save to temp file
            temp_dir = tempfile.gettempdir()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.comparison_image_path = os.path.join(temp_dir, f"reference_only_{timestamp}.png")
            
            plt.savefig(self.comparison_image_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Debug: Reference-only visualization saved to: {self.comparison_image_path}")
            return self.comparison_image_path

        except Exception as e:
            print(f"Error creating reference-only visualization: {e}")
            import traceback
            traceback.print_exc()
            return None


    def display_visualization_image(self, image_path, canvas, label):
        """Display visualization image in the specified canvas"""
        try:
            if not image_path or not os.path.exists(image_path):
                return
            
            # Load and resize image for display
            image = Image.open(image_path)
            
            # Calculate display size - INCREASED from 1000px to 1600px max width
            width, height = image.size
            max_width = 1600  # Increased from 1000
            if width > max_width:
                scale = max_width / width
                new_width = max_width
                new_height = int(height * scale)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Update label
            label.config(image=photo, text="")
            label.image = photo
            
            # Update canvas scroll region
            canvas.configure(scrollregion=canvas.bbox("all"))
            
        except Exception as e:
            print(f"Error displaying visualization: {e}")
            messagebox.showerror("Error", f"Failed to display visualization: {str(e)}")
        
    
    def create_widgets(self):
        """Create the main GUI layout"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Left panel - Controls
        self.create_control_panel(main_frame)
        
        # Right panel - Image display and results
        self.create_display_panel(main_frame)
        
        # Status bar
        self.create_status_bar(main_frame)
    
    def create_control_panel(self, parent):
        """Create the left control panel"""
        control_frame = ttk.LabelFrame(parent, text="Control Panel", padding="10")
        control_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Model Management Section
        model_frame = ttk.LabelFrame(control_frame, text="Model Management", padding="5")
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.load_models_btn = ttk.Button(model_frame, text="Load AI Models", 
                                         command=self.load_models_threaded)
        self.load_models_btn.pack(fill=tk.X, pady=2)
        
        self.unload_models_btn = ttk.Button(model_frame, text="Unload Models", 
                                           command=self.unload_models, state=tk.DISABLED)
        self.unload_models_btn.pack(fill=tk.X, pady=2)
        
        self.model_status_label = ttk.Label(model_frame, text="Status: Models not loaded", 
                                           foreground="red")
        self.model_status_label.pack(pady=2)

        
        
        # Camera Section
        camera_frame = ttk.LabelFrame(control_frame, text="Camera Controls", padding="5")
        camera_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.start_camera_btn = ttk.Button(camera_frame, text="Start Camera", 
                                          command=self.start_camera)
        self.start_camera_btn.pack(fill=tk.X, pady=2)
        
        self.capture_btn = ttk.Button(camera_frame, text="Capture Image", 
                                     command=self.capture_image, state=tk.DISABLED)
        self.capture_btn.pack(fill=tk.X, pady=2)
        
        self.stop_camera_btn = ttk.Button(camera_frame, text="Stop Camera", 
                                         command=self.stop_camera, state=tk.DISABLED)
        self.stop_camera_btn.pack(fill=tk.X, pady=2)
        
        # Image Loading Section
        image_frame = ttk.LabelFrame(control_frame, text="Image Loading", padding="5")
        image_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.load_image_btn = ttk.Button(image_frame, text="Load Image from File", 
                                        command=self.load_image_file)
        self.load_image_btn.pack(fill=tk.X, pady=2)
        
        # Reference Selection
        ref_frame = ttk.LabelFrame(control_frame, text="Reference Selection", padding="5")
        ref_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(ref_frame, text="Select Reference PCBA:").pack(anchor=tk.W)
        self.reference_var = tk.StringVar()
        self.reference_combo = ttk.Combobox(ref_frame, textvariable=self.reference_var, 
                                           state="readonly", width=30)
        self.reference_combo.pack(fill=tk.X, pady=2)
        self.reference_combo.bind('<<ComboboxSelected>>', self.on_reference_selected)
        
        # Inspection Section
        inspect_frame = ttk.LabelFrame(control_frame, text="Inspection", padding="5")
        inspect_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.inspect_btn = ttk.Button(inspect_frame, text="Start Inspection", 
                                     command=self.start_inspection_threaded, 
                                     state=tk.DISABLED)
        self.inspect_btn.pack(fill=tk.X, pady=2)
        
        # Tiling option
        self.use_tiling_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(inspect_frame, text="Use Tiling for Detection", 
                       variable=self.use_tiling_var).pack(anchor=tk.W, pady=2)
        
        # Results Section
        results_frame = ttk.LabelFrame(control_frame, text="Results", padding="5")
        results_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.view_results_btn = ttk.Button(results_frame, text="View Detailed Results", 
                                          command=self.view_detailed_results, 
                                          state=tk.DISABLED)
        self.view_results_btn.pack(fill=tk.X, pady=2)
        
        self.save_results_btn = ttk.Button(results_frame, text="Save Results", 
                                          command=self.save_results, state=tk.DISABLED)
        self.save_results_btn.pack(fill=tk.X, pady=2)



        close_frame = ttk.LabelFrame(control_frame, text="", padding="5")
        close_frame.pack(fill=tk.X, pady=(0, 10))
        self.close_button = tk.Button(close_frame, text="Close", command=self.close_app)
        self.close_button.pack(pady=20)
        
        # Progress bar
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=(10, 0))
    
    def create_display_panel(self, parent):
        """Create the right display panel"""
        display_frame = ttk.Frame(parent)
        display_frame.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)
        
        # Notebook for different views
        self.notebook = ttk.Notebook(display_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Original image tab
        self.image_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.image_frame, text="Original Image")
        
        # Create scrollable image canvas
        self.create_image_canvas(self.image_frame)
        
        # Inspection visualization tab
        self.inspection_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.inspection_frame, text="Inspection Results")
        
        # Create inspection visualization
        self.create_inspection_canvas(self.inspection_frame)
        
        # Comparison view tab
        self.comparison_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.comparison_frame, text="Reference Image")
        
        # Create comparison view
        self.create_comparison_canvas(self.comparison_frame)
        
        # Results tab
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Results")
        
        # Create results display
        self.create_results_display(self.results_frame)
        
        # Summary tab
        self.summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.summary_frame, text="Summary")
        
        # Create summary display
        self.create_summary_display(self.summary_frame)
    
    def create_image_canvas(self, parent):
        """Create scrollable image canvas"""
        # Frame for canvas and scrollbars
        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas with scrollbars
        self.image_canvas = tk.Canvas(canvas_frame, bg='white')
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.image_canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.image_canvas.xview)
        
        self.image_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Grid layout
        self.image_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))

        # Configure grid weights
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
        # Image display label
        self.image_label = ttk.Label(self.image_canvas, text="No image loaded", 
                                    background='white')
        self.image_canvas.create_window(0, 0, window=self.image_label, anchor=tk.NW)
        
    def create_inspection_canvas(self, parent):
        """Create scrollable canvas for inspection visualization"""
        # Frame for canvas and scrollbars
        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas with scrollbars
        self.inspection_canvas = tk.Canvas(canvas_frame, bg='white')
        v_scrollbar2 = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.inspection_canvas.yview)
        h_scrollbar2 = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.inspection_canvas.xview)
        
        self.inspection_canvas.configure(yscrollcommand=v_scrollbar2.set, xscrollcommand=h_scrollbar2.set)
        
        # Grid layout
        self.inspection_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        v_scrollbar2.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar2.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Configure grid weights
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
        # Inspection image display label
        self.inspection_label = ttk.Label(self.inspection_canvas, text="Run inspection to see results", 
                                         background='white')
        self.inspection_canvas.create_window(0, 0, window=self.inspection_label, anchor=tk.NW)
    
    def create_comparison_canvas(self, parent):
        """Create scrollable canvas for side-by-side comparison"""
        # Frame for canvas and scrollbars
        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas with scrollbars
        self.comparison_canvas = tk.Canvas(canvas_frame, bg='white')
        v_scrollbar3 = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.comparison_canvas.yview)
        h_scrollbar3 = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.comparison_canvas.xview)
        
        self.comparison_canvas.configure(yscrollcommand=v_scrollbar3.set, xscrollcommand=h_scrollbar3.set)
        
        # Grid layout
        self.comparison_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        v_scrollbar3.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar3.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Configure grid weights
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
        # Comparison image display label
        self.comparison_label = ttk.Label(self.comparison_canvas, text="Select reference and run inspection to see comparison", 
                                         background='white')
        self.comparison_canvas.create_window(0, 0, window=self.comparison_label, anchor=tk.NW)
    
    def create_results_display(self, parent):
        """Create results display area"""
        # Text widget with scrollbar
        text_frame = ttk.Frame(parent)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.results_text = tk.Text(text_frame, wrap=tk.WORD, state=tk.DISABLED)
        results_scroll = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scroll.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_summary_display(self, parent):
        """Create summary display area"""
        # Frame for summary table
        summary_frame = ttk.Frame(parent)
        summary_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Treeview for table display
        columns = ('Component', 'Ground Truth', 'Detected', 'Matched', 'Missed', 
                  'False Positives', 'Recall', 'Precision')
        
        self.summary_tree = ttk.Treeview(summary_frame, columns=columns, show='headings', height=15)
        
        # Configure columns
        for col in columns:
            self.summary_tree.heading(col, text=col)
            self.summary_tree.column(col, width=100, anchor=tk.CENTER)
        
        # Scrollbars for treeview
        tree_scroll_v = ttk.Scrollbar(summary_frame, orient=tk.VERTICAL, command=self.summary_tree.yview)
        tree_scroll_h = ttk.Scrollbar(summary_frame, orient=tk.HORIZONTAL, command=self.summary_tree.xview)
        self.summary_tree.configure(yscrollcommand=tree_scroll_v.set, xscrollcommand=tree_scroll_h.set)
        
        # Grid layout
        self.summary_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        tree_scroll_v.grid(row=0, column=1, sticky=(tk.N, tk.S))
        tree_scroll_h.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Configure grid weights
        summary_frame.columnconfigure(0, weight=1)
        summary_frame.rowconfigure(0, weight=1)
    
    def create_status_bar(self, parent):
        """Create status bar"""
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        
        status_bar = ttk.Label(parent, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def load_reference_data(self):
        """Load reference data from annotation file"""
        try:
            if os.path.exists(self.annotation_file):
                verifier = PCBAVerifier(self.annotation_file)
                self.reference_images = verifier.list_available_images()
                
                # Populate combobox
                ref_options = [f"ID {img['id']}: {img['file_name']}" for img in self.reference_images]
                self.reference_combo['values'] = ref_options
                
                self.update_status(f"Loaded {len(self.reference_images)} reference images")
            else:
                self.update_status("Warning: Annotation file not found")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load reference data: {str(e)}")
    
    def load_models_threaded(self):
        """Load AI models in background thread"""
        if self.models_loaded:
            messagebox.showinfo("Info", "Models are already loaded")
            return
        
        self.progress.start()
        self.load_models_btn.config(state=tk.DISABLED)
        self.update_status("Loading AI models...")
        
        def load_models():
            try:
                # Model paths
                pcb_model_path = "output/pcb_detector/model_final.pth"
                component_model_path = "output/component_detector/model_final.pth"
                direction_model_path = "output/direction/efficient_direction_classifier.pth"
                
                # Load models
                self.pcb_detector = get_pcb_detector(pcb_model_path)
                self.component_detector = get_component_detector(component_model_path)
                self.direction_classifier = get_direction_classifier(direction_model_path)
                
                self.result_queue.put(('models_loaded', True))
                
            except Exception as e:
                self.result_queue.put(('models_loaded', False, str(e)))
        
        thread = threading.Thread(target=load_models, daemon=True)
        thread.start()
    
    def unload_models(self):
        """Unload models to free memory"""
        if not self.models_loaded:
            return
        
        self.pcb_detector = None
        self.component_detector = None
        self.direction_classifier = None
        
        # Force garbage collection
        gc.collect()
        
        self.models_loaded = False
        self.model_status_label.config(text="Status: Models not loaded", foreground="red")
        self.load_models_btn.config(state=tk.NORMAL)
        self.unload_models_btn.config(state=tk.DISABLED)
        self.inspect_btn.config(state=tk.DISABLED)
        
        self.update_status("Models unloaded from memory")
    
    def start_camera(self):
        """Start camera preview"""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                messagebox.showerror("Error", "Could not open camera")
                return
            
            self.camera_active = True
            self.start_camera_btn.config(state=tk.DISABLED)
            self.capture_btn.config(state=tk.NORMAL)
            self.stop_camera_btn.config(state=tk.NORMAL)
            
            self.update_camera_preview()
            self.update_status("Camera started")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
    
    def update_camera_preview(self):
        """Update camera preview"""
        if self.camera_active and self.camera is not None:
            ret, frame = self.camera.read()
            if ret:
                # Resize frame for display - INCREASED size
                height, width = frame.shape[:2]
                max_width = 900  # Increased from 600
                if width > max_width:
                    scale = max_width / width
                    new_width = max_width
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Convert to RGB and display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(image)
                
                self.image_label.config(image=photo, text="")
                self.image_label.image = photo
                
                # Update canvas scroll region
                self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all"))
            
            # Schedule next update
            self.root.after(30, self.update_camera_preview)
    
    def capture_image(self):
        """Capture image from camera"""
        if self.camera is not None and self.camera_active:
            ret, frame = self.camera.read()
            if ret:
                # Save captured image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.current_image_path = f"captured_images/capture_{timestamp}.jpg"
                
                # Create directory if it doesn't exist
                os.makedirs("captured_images", exist_ok=True)
                
                cv2.imwrite(self.current_image_path, frame)
                
                # Display captured image
                self.display_image(self.current_image_path)
                
                # Enable inspection if models are loaded
                if self.models_loaded:
                    self.inspect_btn.config(state=tk.NORMAL)
                
                self.update_status(f"Image captured: {self.current_image_path}")
            else:
                messagebox.showerror("Error", "Failed to capture image")
    
    def stop_camera(self):
        """Stop camera preview"""
        self.camera_active = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        
        self.start_camera_btn.config(state=tk.NORMAL)
        self.capture_btn.config(state=tk.DISABLED)
        self.stop_camera_btn.config(state=tk.DISABLED)
        
        self.update_status("Camera stopped")
    
    def load_image_file(self):
        """Load image from file"""
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            
            # Enable inspection if models are loaded
            if self.models_loaded:
                self.inspect_btn.config(state=tk.NORMAL)
            
            self.update_status(f"Image loaded: {os.path.basename(file_path)}")
    
    def display_image(self, image_path):
        """Display image in the canvas"""
        try:
            # Load and resize image for display
            image = Image.open(image_path)
            
            # Calculate display size - INCREASED from 800px to 1200px max width
            width, height = image.size
            max_width = 1200  # Increased from 800
            if width > max_width:
                scale = max_width / width
                new_width = max_width
                new_height = int(height * scale)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Update label
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
            
            # Update canvas scroll region
            self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all"))
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image: {str(e)}")
    
    def on_reference_selected(self, event):
        """Handle reference selection"""
        selection = self.reference_var.get()
        if selection:
            # Extract ID from selection
            self.selected_reference_id = int(selection.split(":")[0].replace("ID ", ""))
            self.update_status(f"Reference selected: {selection}")
    
    def start_inspection_threaded(self):
        """Start inspection in background thread"""
        if not self.models_loaded:
            messagebox.showwarning("Warning", "Please load AI models first")
            return
        
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please load or capture an image first")
            return
        
        self.progress.start()
        self.inspect_btn.config(state=tk.DISABLED)
        self.update_status("Running inspection...")
        
        def run_inspection():
            try:
                start_time = time.time()
                
                # Run detection pipeline
                detection_results = three_stage_detection_pipeline(
                    self.current_image_path,
                    pcb_detector=self.pcb_detector,
                    component_detector=self.component_detector,
                    direction_classifier=self.direction_classifier,
                    use_tiling=self.use_tiling_var.get(),
                    overlap=0.2
                )
                
                # Run verification if reference is selected
                verification_results = None
                summary_df = None
                
                if self.selected_reference_id is not None:
                    verifier = PCBAVerifier(self.annotation_file)
                    verification_results = verifier.verify_detection(
                        self.current_image_path, 
                        detection_results, 
                        image_id=self.selected_reference_id
                    )
                    summary_df = verifier.component_summary(verification_results)
                
                processing_time = time.time() - start_time
                
                self.result_queue.put(('inspection_complete', {
                    'detection_results': detection_results,
                    'verification_results': verification_results,
                    'summary_df': summary_df,
                    'processing_time': processing_time
                }))
                
            except Exception as e:
                self.result_queue.put(('inspection_error', str(e)))
        
        thread = threading.Thread(target=run_inspection, daemon=True)
        thread.start()
    
    def view_detailed_results(self):
        """Open detailed results in new window"""
        if self.last_verification_results is None:
            messagebox.showinfo("Info", "No results available")
            return
        
        # Create new window for detailed results
        results_window = tk.Toplevel(self.root)
        results_window.title("Detailed Inspection Results")
        results_window.geometry("1200x800")
        
        # Create notebook for different views
        notebook = ttk.Notebook(results_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Matches tab
        matches_frame = ttk.Frame(notebook)
        notebook.add(matches_frame, text="Matches")
        self.create_matches_display(matches_frame)
        
        # Missed detections tab
        missed_frame = ttk.Frame(notebook)
        notebook.add(missed_frame, text="Missed Detections")
        self.create_missed_display(missed_frame)
        
        # False positives tab
        false_pos_frame = ttk.Frame(notebook)
        notebook.add(false_pos_frame, text="False Positives")
        self.create_false_pos_display(false_pos_frame)
    
    def create_matches_display(self, parent):
        """Create matches display in detailed results"""
        columns = ('Component', 'Score', 'Direction', 'GT Rotation', 'Orientation Match', 'PCB ID')
        tree = ttk.Treeview(parent, columns=columns, show='headings')
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)
        
        # Add data
        for match in self.last_verification_results['matches']:
            tree.insert('', tk.END, values=(
                match['component'],
                f"{match['score']:.3f}",
                match.get('detected_direction', 'N/A'),
                f"{match.get('gt_rotation', 0):.1f}°",
                match.get('orientation_match', 'N/A'),
                match.get('pcb_id', 'N/A')
            ))
        
        tree.pack(fill=tk.BOTH, expand=True)
    
    def create_missed_display(self, parent):
        """Create missed detections display"""
        columns = ('Component', 'GT Rotation', 'PCB ID')
        tree = ttk.Treeview(parent, columns=columns, show='headings')
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150)
        
        # Add data
        for miss in self.last_verification_results['missed_detections']:
            tree.insert('', tk.END, values=(
                miss['component'],
                f"{miss.get('gt_rotation', 0):.1f}°",
                miss.get('pcb_id', 'N/A')
            ))
        
        tree.pack(fill=tk.BOTH, expand=True)
    
    def create_false_pos_display(self, parent):
        """Create false positives display"""
        columns = ('Component', 'Score', 'Direction', 'PCB ID')
        tree = ttk.Treeview(parent, columns=columns, show='headings')
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150)
        
        # Add data
        for fp in self.last_verification_results['false_positives']:
            tree.insert('', tk.END, values=(
                fp['component'],
                f"{fp['score']:.3f}",
                fp.get('detected_direction', 'N/A'),
                fp.get('pcb_id', 'N/A')
            ))
        
        tree.pack(fill=tk.BOTH, expand=True)
    
    def save_results(self):
        """Save results to file in COCO format"""
        if self.last_detection_results is None:
            messagebox.showwarning("Warning", "No detection results available. Please run inspection first.")
            return
        
        # Ask for save location
        file_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                if file_path.endswith('.csv') and self.last_summary_df is not None:
                    # Save summary as CSV
                    self.last_summary_df.to_csv(file_path, index=False)
                else:
                    # Save detection results in COCO format
                    if self.current_image_path:
                        img = cv2.imread(self.current_image_path)
                        if img is not None:
                            height, width = img.shape[:2]
                            image_filename = os.path.basename(self.current_image_path)
                        else:
                            # Fallback values
                            height, width = 1080, 1920
                            image_filename = "unknown.jpg"
                    else:
                        # Fallback values
                        height, width = 1080, 1920
                        image_filename = "unknown.jpg"
                    
                    # Create COCO-style JSON structure
                    coco_data = {
                        "licenses": [
                            {
                                "name": "",
                                "id": 0,
                                "url": ""
                            }
                        ],
                        "info": {
                            "contributor": "PCBA AOI System",
                            "date_created": datetime.now().isoformat(),
                            "description": "Exported detection results",
                            "url": "",
                            "version": "1.0",
                            "year": str(datetime.now().year)
                        },
                        "categories": [],
                        "images": [
                            {
                                "id": 1,
                                "width": width,
                                "height": height,
                                "file_name": image_filename,
                                "license": 0,
                                "flickr_url": "",
                                "coco_url": "",
                                "date_captured": int(time.time())
                            }
                        ],
                        "annotations": []
                    }
                    
                    # Create categories from component names
                    component_names = self.last_detection_results.get("component_names", [])
                    for idx, name in enumerate(component_names):
                        coco_data["categories"].append({
                            "id": idx + 1,  # COCO categories start from 1
                            "name": name,
                            "supercategory": ""
                        })
                    
                    # Add PCB category if PCBs were detected
                    pcb_boxes = self.last_detection_results.get("pcb_boxes", [])
                    if len(pcb_boxes) > 0:
                        coco_data["categories"].append({
                            "id": len(component_names) + 1,
                            "name": "PCBA",
                            "supercategory": ""
                        })
                    
                    annotation_id = 1
                    
                    # Add PCB annotations
                    pcb_scores = self.last_detection_results.get("pcb_scores", [])
                    for i, (box, score) in enumerate(zip(pcb_boxes, pcb_scores)):
                        x1, y1, x2, y2 = box
                        width_box = x2 - x1
                        height_box = y2 - y1
                        area = width_box * height_box
                        
                        annotation = {
                            "id": annotation_id,
                            "image_id": 1,
                            "category_id": len(component_names) + 1,  # PCBA category
                            "segmentation": [],
                            "area": float(area),
                            "bbox": [round(float(x1), 2), round(float(y1), 2), round(float(width_box), 2), round(float(height_box), 2)],
                            "iscrowd": 0,
                            "attributes": {
                                "occluded": False,
                                 "rotation": 0.0
                            }
                        }
                        coco_data["annotations"].append(annotation)
                        annotation_id += 1
                    
                    # Add component annotations
                    component_boxes = self.last_detection_results.get("component_boxes", [])
                    component_classes = self.last_detection_results.get("component_classes", [])
                    component_scores = self.last_detection_results.get("component_scores", [])
                    component_directions = self.last_detection_results.get("component_directions", [])
                    
                    for i, (box, cls, score) in enumerate(zip(component_boxes, component_classes, component_scores)):
                        x1, y1, x2, y2 = box
                        width_box = x2 - x1
                        height_box = y2 - y1
                        area = width_box * height_box
                        
                        # Get direction information if available
                        direction = component_directions[i] if i < len(component_directions) else None
                        
                        # Convert direction to rotation angle
                        direction_to_rotation = {
                            "N": 0.0,
                            "NE": 45.0,
                            "E": 90.0,
                            "SE": 135.0,
                            "S": 180.0,
                            "SW": 225.0,
                            "W": 270.0,
                            "NW": 315.0
                        }
                        rotation = direction_to_rotation.get(direction, 0.0) if direction else 0.0
                        
                        annotation = {
                            "id": annotation_id,
                            "image_id": 1,
                            "category_id": int(cls) + 1,  # COCO categories start from 1
                            "segmentation": [],
                            "area": float(area),
                            "bbox": [round(float(x1),2), round(float(y1),2), round(float(width_box),2), round(float(height_box),2)],
                            "iscrowd": 0,
                            "attributes": {
                                "occluded": False,
                                "rotation": rotation
                            }
                        }
                        
                        
                        coco_data["annotations"].append(annotation)
                        annotation_id += 1
                    
                    # Save to JSON file
                    with open(file_path, 'w') as f:
                        json.dump(coco_data, f, indent=4)
                
                self.update_status(f"Results saved to {file_path}")
                messagebox.showinfo("Success", f"Results saved successfully to {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {str(e)}")
                print(f"Save error: {e}")
                import traceback
                traceback.print_exc()
    
    def update_status(self, message):
        """Update status bar"""
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def check_queue(self):
        """Check for results from background threads"""
        try:
            while True:
                result = self.result_queue.get_nowait()
                self.handle_result(result)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.check_queue)
    
    def handle_result(self, result):
        """Handle results from background threads"""
        if result[0] == 'models_loaded':
            self.progress.stop()
            self.load_models_btn.config(state=tk.DISABLED)
            
            if result[1]:  # Success
                self.models_loaded = True
                self.model_status_label.config(text="Status: Models loaded", foreground="green")
                self.unload_models_btn.config(state=tk.NORMAL)
                if self.current_image_path:
                    self.inspect_btn.config(state=tk.NORMAL)
                self.update_status("AI models loaded successfully")
            else:  # Error
                error_msg = result[2] if len(result) > 2 else "Unknown error"
                self.load_models_btn.config(state=tk.NORMAL)
                self.update_status("Failed to load models")
                messagebox.showerror("Error", f"Failed to load models: {error_msg}")
        
        elif result[0] == 'inspection_complete':
            self.progress.stop()
            self.inspect_btn.config(state=tk.NORMAL)
            
            results = result[1]
            self.last_verification_results = results['verification_results']
            self.last_summary_df = results['summary_df']
            self.last_detection_results = results['detection_results']  # Store detection results
            
            # Update displays
            self.update_results_display(results)
            self.update_summary_display(results['summary_df'])
            
            # Create and display visualizations
            if results['detection_results']:
                # Create inspection visualization
                inspection_viz_path = self.create_inspection_visualization(
                    results['detection_results'], 
                    results['verification_results']
                )
                if inspection_viz_path:
                    self.display_visualization_image(
                        inspection_viz_path, 
                        self.inspection_canvas, 
                        self.inspection_label
                    )
                
                # Create comparison visualization if reference is selected AND verification was performed
                if results['verification_results'] and self.selected_reference_id is not None:
                    print(f"Debug: Creating comparison visualization for reference ID {self.selected_reference_id}")
                    comparison_viz_path = self.create_reference_visualization_only(
                        results['detection_results'],
                        results['verification_results']
                    )
                    if comparison_viz_path:
                        print("Debug: Displaying comparison visualization")
                        self.display_visualization_image(
                            comparison_viz_path,
                            self.comparison_canvas,
                            self.comparison_label
                        )
                    else:
                        print("Debug: Failed to create comparison visualization")
                        # Update the label to show an error message
                        self.comparison_label.config(
                            text="Failed to create comparison visualization. Check console for errors.",
                            image=""
                        )
                        self.comparison_label.image = None
                else:
                    print(f"Debug: Not creating comparison - verification_results: {results['verification_results'] is not None}, reference_id: {self.selected_reference_id}")
                    # Update the label to indicate no reference was selected
                    if self.selected_reference_id is None:
                        self.comparison_label.config(
                            text="No reference selected. Select a reference PCBA and run inspection again.",
                            image=""
                        )
                    else:
                        self.comparison_label.config(
                            text="Verification not performed. Make sure reference is properly selected.",
                            image=""
                        )
                    self.comparison_label.image = None
            
            # Enable result buttons
            if self.last_verification_results:
                self.view_results_btn.config(state=tk.NORMAL)
                self.save_results_btn.config(state=tk.NORMAL)
            
            # Enable save button if detection results are available (even without verification)
            if self.last_detection_results:
                self.save_results_btn.config(state=tk.NORMAL)
            
            processing_time = results['processing_time']
            self.update_status(f"Inspection completed in {processing_time:.2f} seconds")
            
            # Switch to inspection results tab
            self.notebook.select(self.inspection_frame)
        
        elif result[0] == 'inspection_error':
            self.progress.stop()
            self.inspect_btn.config(state=tk.NORMAL)
            error_msg = result[1]
            self.update_status("Inspection failed")
            messagebox.showerror("Error", f"Inspection failed: {error_msg}")
    
    def update_results_display(self, results):
        """Update the results text display"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        detection_results = results['detection_results']
        verification_results = results['verification_results']
        
        # Display detection summary
        self.results_text.insert(tk.END, "PCBA AOI INSPECTION RESULTS\n")
        self.results_text.insert(tk.END, "=" * 50 + "\n\n")
        
        # PCB Detection Results
        pcb_count = len(detection_results.get('pcb_boxes', []))
        self.results_text.insert(tk.END, f"PCBs Detected: {pcb_count}\n")
        
        # Component Detection Results
        component_count = len(detection_results.get('component_boxes', []))
        self.results_text.insert(tk.END, f"Components Detected: {component_count}\n\n")
        
        # Component breakdown
        if 'component_classes' in detection_results and 'component_names' in detection_results:
            from collections import Counter
            component_classes = detection_results['component_classes']
            component_names = detection_results['component_names']
            component_counter = Counter(component_classes)
            
            self.results_text.insert(tk.END, "Component Breakdown:\n")
            self.results_text.insert(tk.END, "-" * 30 + "\n")
            
            for cls, count in sorted(component_counter.items()):
                name = component_names[cls] if cls < len(component_names) else f"Class {cls}"
                self.results_text.insert(tk.END, f"{name}: {count}\n")
            
            self.results_text.insert(tk.END, "\n")
        
        # Verification Results (if available)
        if verification_results:
            metrics = verification_results['metrics']
            self.results_text.insert(tk.END, "VERIFICATION AGAINST REFERENCE\n")
            self.results_text.insert(tk.END, "=" * 40 + "\n\n")
            
            self.results_text.insert(tk.END, f"Ground Truth Components: {metrics['total_gt']}\n")
            self.results_text.insert(tk.END, f"Detected Components: {metrics['total_detected']}\n")
            self.results_text.insert(tk.END, f"Correct Matches: {metrics['correct_detections']}\n")
            self.results_text.insert(tk.END, f"Missed Detections: {len(verification_results['missed_detections'])}\n")
            self.results_text.insert(tk.END, f"False Positives: {len(verification_results['false_positives'])}\n\n")
            
            self.results_text.insert(tk.END, f"Precision: {metrics['precision']:.3f}\n")
            self.results_text.insert(tk.END, f"Recall: {metrics['recall']:.3f}\n")
            self.results_text.insert(tk.END, f"F1 Score: {metrics['f1_score']:.3f}\n\n")
            
            # Overall assessment
            if metrics['f1_score'] >= 0.9:
                assessment = "EXCELLENT"
                color = "green"
            elif metrics['f1_score'] >= 0.8:
                assessment = "GOOD"
                color = "blue"
            elif metrics['f1_score'] >= 0.7:
                assessment = "ACCEPTABLE"
                color = "orange"
            else:
                assessment = "NEEDS ATTENTION"
                color = "red"
            
            self.results_text.insert(tk.END, f"Overall Assessment: {assessment}\n")
            
            # Orientation accuracy (if available)
            orientation_matches = sum(1 for match in verification_results['matches'] 
                                    if match.get('orientation_match') == 'Match')
            orientation_total = sum(1 for match in verification_results['matches'] 
                                  if match.get('orientation_match') in ['Match', 'Mismatch'])
            
            if orientation_total > 0:
                orientation_accuracy = orientation_matches / orientation_total
                self.results_text.insert(tk.END, f"Orientation Accuracy: {orientation_accuracy:.3f}\n")
        
        self.results_text.config(state=tk.DISABLED)
    
    def update_summary_display(self, summary_df):
        """Update the summary table display"""
        # Clear existing data
        for item in self.summary_tree.get_children():
            self.summary_tree.delete(item)
        
        if summary_df is not None:
            # Add data to treeview
            for _, row in summary_df.iterrows():
                values = (
                    row['Component'],
                    row['Ground Truth'],
                    row['Detected'],
                    row['Matched'],
                    row['Missed'],
                    row['False Positives'],
                    f"{row['Recall']:.3f}",
                    f"{row['Precision']:.3f}"
                )
                
                # Color coding based on performance
                item = self.summary_tree.insert('', tk.END, values=values)
                
                # Highlight rows with issues
                if row['Missed'] > 0 or row['False Positives'] > 0:
                    if row['Recall'] < 0.8 or row['Precision'] < 0.8:
                        self.summary_tree.set(item, 'Component', f"⚠️ {row['Component']}")
    
    def on_closing(self):
        """Handle application closing"""
        # Stop camera if active
        if self.camera_active:
            self.stop_camera()
        
        # Clean up temporary visualization files
        for temp_path in [self.inspection_image_path, self.comparison_image_path]:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
        
        # Unload models to free memory
        if self.models_loaded:
            self.unload_models()
        
        self.root.destroy()


def main():
    """Main application entry point"""
    # Create main window
    root = tk.Tk()
    
    # Create application
    app = PCBAAOIApplication(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")
    
    # Start application
    root.mainloop()


if __name__ == "__main__":
    main()