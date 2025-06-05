import torch
import random
import numpy as np
import os
import copy
import math
import cv2
import matplotlib.pyplot as plt
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.datasets import register_coco_instances
from detectron2.data.build import build_detection_train_loader
from detectron2 import model_zoo

_COMPONENT_DETECTOR = None

_DATASETS_REGISTERED = {
    "components_only": False
}
def get_component_detector(model_path):
    """Get component detector singleton"""
    global _COMPONENT_DETECTOR
    
    if _COMPONENT_DETECTOR is None:
        register_component_dataset()
        # Setup config
        cfg = setup_component_detector_config()
        cfg.MODEL.WEIGHTS = model_path
        
        # Create predictor
        _COMPONENT_DETECTOR = DefaultPredictor(cfg)
        print("Component detector loaded from:", model_path)
    
    return _COMPONENT_DETECTOR
def register_component_dataset():
    global _DATASETS_REGISTERED
    
    if _DATASETS_REGISTERED["components_only"]:
        # Return the already registered category names
        return MetadataCatalog.get("components_only_train").thing_classes
    
    """Register the component dataset (without PCB class)"""
    # Define category names for components (excluding PCB)
    component_names = [
        "relay", "led", "resistor", "capacitator electrolytic", "inductor", 
        "ferrite bead", "potentiometer", "diode", "transistor BC", "IC", 
        "voltage regulator", "transformer", "fuse", "sensor", "oscillator", 
        "rf module", "antenna", "connector", "switch", "display", 
        "varistor", "thermistor", "metal", "fuse holder", "Capacitator ceramic",
        "Cap pest", "FET", "mini PCBA", "Transistor ESBT", "Transistor PNP",
        "led panel", "coil", "fuse switch", "dc-dc converter", "pin",
        "diode bridge"  # No PCBA class
    ]
    
    # Register component dataset
    if "components_only_train" not in DatasetCatalog:
        register_coco_instances(
            "components_only_train",
            {"thing_classes": component_names},
            "datasets/pcba_crops/pcba_components.json",
            "datasets/pcba_crops/images"
        )
        print("Component-only dataset registered")
    else:
        # Update metadata even if already registered
        MetadataCatalog.get("components_only_train").thing_classes = component_names
        print("Component-only dataset already registered, updated metadata")
    _DATASETS_REGISTERED["components_only"] = True
    return component_names


def setup_component_detector_config():
    """Configure the component detector model (stage 2)"""
    cfg = get_cfg()
    
    # Use FPN for better small object detection
    config_path = model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    
    # Set to detect 36 component classes (no PCB)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 36
    
    # Small object detection improvements
    #cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32], [32, 64, 96], [96, 128, 192], [192 ,256, 384], [384 ,512, 768]]
    #cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16], [ 32, 64], [128, 192], [256, 384], [512, 768]]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16], [ 32, 64], [96, 128], [160, 192], [256, 384]] #these anchor boxes based on analysis of aspect ratios and sizes smaller boxes
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.33, 0.5, 1.0, 2.0, 3.0]] 
    
    # Input size adjustments
    cfg.INPUT.MIN_SIZE_TRAIN = 800
    cfg.INPUT.MAX_SIZE_TRAIN = 1333 
    cfg.INPUT.MIN_SIZE_TEST =  800
    cfg.INPUT.MAX_SIZE_TEST = 1333 
    
    # 3. Adjust RPN parameters for better small object proposals
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
    cfg.MODEL.RPN.NMS_THRESH = 0.7 #0.5->0.7 Back to default
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5 # 0.2->0.5 back to default
    cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE = "ciou"
    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = "ciou"
    cfg.MODEL.FPN.NORM = "GN"
    cfg.MODEL.ROI_BOX_HEAD.NORM = "GN"
    #cfg.MODEL.RESNETS.NORM = "GN" #Surprisingly does not increase performance

    
    ## 4. Better pooling for small objects
    cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 14 
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25 # 0.5->0.35->0.25 I have had problem with false positives
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 #lower for less memory 256->512  # for increased ability to detect small objects
    
    # 5. Lower score threshold for testing to catch more small objects
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
    
    # 6. Training parameters
    cfg.SOLVER.IMS_PER_BATCH = 1  # May need to reduce to 1 if OOM errors occurs 
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 15000 #for fast test
    cfg.SOLVER.STEPS = (8000,12500)
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.AMP.ENABLED = True
    
    cfg.DATALOADER.NUM_WORKERS = 4  # Adjust based on your CPU cores
    cfg.MODEL.DEVICE = "cuda"
    
    # Pin memory for faster CPU->GPU transfer
    cfg.DATALOADER.PIN_MEMORY = True
    
    # Output directory for component detector
    cfg.OUTPUT_DIR = "./output/component_detector"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    return cfg



def classification_mapper_component(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    annotations = dataset_dict.get("annotations", [])
    
    # Perform tiling
    from tiling import tile_image  # Local import
    tiles = tile_image(image, annotations, overlap=0.2, visualize=False)

    if not tiles:
        # Fallback in case no tiles were created
        return None
    
    # Randomly select one tile per iteration (stochastic)
    tile = random.choice(tiles)

    tile_image_crop = tile["image"]
    tile_annotations = tile["annotations"]
    
    # Augmentations after tiling
    augs = T.AugmentationList([
        T.ResizeShortestEdge(short_edge_length=(640,800), max_size=1333),
        T.RandomBrightness(0.8, 1.2),
        T.RandomContrast(0.8, 1.2),
        T.RandomSaturation(0.8, 1.2),
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        T.RandomRotation([-15, 15])
    ])
    aug_input = T.AugInput(tile_image_crop)
    transforms = augs(aug_input)
    image_transformed = aug_input.image

    dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image_transformed.transpose(2, 0, 1)))

    # Transform annotations
    annos_transformed = [
        utils.transform_instance_annotations(anno, transforms, image_transformed.shape[:2])
        for anno in tile_annotations
    ]
    
    filtered_annos = []
    for anno in annos_transformed:
        x1, y1, x2, y2 = anno["bbox"]
        w, h = x2 - x1, y2 - y1
        # Filter out boxes that are too small
        if w >= 1 and h >= 1:
            filtered_annos.append(anno)

    # Better handling of empty annotations
    if not filtered_annos:
        # Create a 'background' class or use a weighted sampling approach
        # depending on your application needs
        dataset_dict["instances"] = utils.annotations_to_instances(
            [], image_transformed.shape[:2]
        )
        return dataset_dict
    

    # For samples with valid annotations
    instances = utils.annotations_to_instances(
        filtered_annos, image_transformed.shape[:2]
    )

    dataset_dict["instances"] = instances
    
    return dataset_dict


# Custom trainer for classification only
class ComponentDetectorTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        dataset_name = cfg.DATASETS.TRAIN[0]
        if not hasattr(MetadataCatalog.get(dataset_name), "thing_classes"):
            MetadataCatalog.get(dataset_name).thing_classes = ["PCBA"]
            print(f"Registered classes with {dataset_name}")

        mapper = classification_mapper_component

        # Filter None outputs (empty tiles)
        def filtered_mapper(dataset_dict):
            result = mapper(dataset_dict)
            return result if result else None

        return build_detection_train_loader(cfg, mapper=filtered_mapper)


def train_component_detector(resume_from=None):
    """Train the second stage component detector with augmentations"""
    # Register component dataset
    register_component_dataset()
    
    # Setup config
    cfg = setup_component_detector_config()
    cfg.DATASETS.TRAIN = ("components_only_train",)
    cfg.DATASETS.TEST = ()
    
    if resume_from:
        cfg.MODEL.WEIGHTS = resume_from
    
    # Create trainer with component-specific augmentations
    trainer = ComponentDetectorTrainer(cfg)
    trainer.resume_or_load(resume=True if resume_from else False) # also loads lerningrate schedule and such things
    #trainer.resume_or_load(resume=False)  # only loads weights

    trainer.train()


def detect_components_in_region(cropped_image, model_path):
    """
    Second stage: Detect components in a cropped PCB region
    
    Args:
        cropped_image: Cropped PCB image
        model_path: Path to the trained component detector model
        
    Returns:
        dict: Component detection results with boxes, classes, and scores
    """
    # Register component dataset to get component names
    component_names = register_component_dataset()
    
    # Setup config
    cfg = setup_component_detector_config()
    cfg.MODEL.WEIGHTS = model_path
    
    # Create predictor
    predictor = DefaultPredictor(cfg)
    
    # Run component detection
    outputs = predictor(cropped_image)
    
    # Extract component detections
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy()
    classes = instances.pred_classes.numpy()
    scores = instances.scores.numpy()
    
    return {
        "boxes": boxes,
        "classes": classes,
        "scores": scores,
        "component_names": component_names
    }


def map_coordinates_to_original(component_detections, pcb_coordinates):
    """
    Map component coordinates from cropped PCB image back to original image
    
    Args:
        component_detections: Component detection results from detect_components_in_region
        pcb_coordinates: Coordinates of the PCB region in the original image (x1, y1, x2, y2)
        
    Returns:
        dict: Updated component detection results with coordinates in original image
    """
    x1_pcb, y1_pcb, _, _ = pcb_coordinates
    
    # Get component boxes
    boxes = component_detections["boxes"]
    
    # Map coordinates back to original image
    mapped_boxes = []
    for box in boxes:
        # Component coordinates in cropped image
        x1_comp, y1_comp, x2_comp, y2_comp = box
        
        # Map to original image
        x1_orig = x1_pcb + x1_comp
        y1_orig = y1_pcb + y1_comp
        x2_orig = x1_pcb + x2_comp
        y2_orig = y1_pcb + y2_comp
        
        mapped_boxes.append([x1_orig, y1_orig, x2_orig, y2_orig])
    
    # Update component detections with mapped coordinates
    mapped_detections = copy.deepcopy(component_detections)
    mapped_detections["boxes"] = np.array(mapped_boxes)
    
    return mapped_detections