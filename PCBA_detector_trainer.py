import torch
import numpy as np
import os
import copy
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.datasets import register_coco_instances
from detectron2.data.build import build_detection_train_loader
from detectron2 import model_zoo

_DATASETS_REGISTERED = {
    "pcb_only": False,
}

_PCB_DETECTOR = None

def get_pcb_detector(model_path):
    """Get PCB detector with singleton pattern"""
    global _PCB_DETECTOR
    
    if _PCB_DETECTOR is None:
        # Register dataset once
        register_pcb_only_dataset()
        
        # Setup config
        cfg = setup_pcb_detector_config()
        cfg.MODEL.WEIGHTS = model_path
        
        # Create predictor
        _PCB_DETECTOR = DefaultPredictor(cfg)
        print("PCB detector loaded from:", model_path)
    
    return _PCB_DETECTOR

def classification_mapper_pcb(dataset_dict):
    """
    Map the dataset dictionary to include only classification information
    """
    dataset_dict = copy.deepcopy(dataset_dict)
    
    # Read image
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    
    # Apply standard transformations (scaling, etc.)
    augs = T.AugmentationList([
        T.ResizeShortestEdge(
            short_edge_length=(640,800),
            max_size=1333
        ),
        T.RandomBrightness(0.8, 1.2),
        T.RandomContrast(0.8, 1.2),
        T.RandomSaturation(0.8, 1.2),
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),  
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),  
        T.RandomRotation([-15, 15]),
        T.RandomCrop("relative_range", (0.9, 1.0))
    ])
    aug_input = T.AugInput(image)
    transforms = augs(aug_input)
    image = aug_input.image
    
    # Handle image and basic annotations
    dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
    
    # Process annotations - without rotation information
    if "annotations" in dataset_dict:
        annos = []
        
        for anno in dataset_dict["annotations"]:
            # Make a copy of the annotation
            anno = copy.deepcopy(anno)
            
            # Transform bounding box for resizing, etc.
            utils.transform_instance_annotations(
                anno, transforms, (image.shape[0], image.shape[1])
            )
            
            annos.append(anno)
        
        # Convert annotations to instances
        instances = utils.annotations_to_instances(
            annos, (image.shape[0], image.shape[1])
        )
        
        dataset_dict["instances"] = instances
    
    return dataset_dict

class PCBDetectorTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        # Make sure to register metadata with the catalog
        dataset_name = cfg.DATASETS.TRAIN[0]
        if not hasattr(MetadataCatalog.get(dataset_name), "thing_classes"):
            category_names = [
                "PCBA"
            ]
            MetadataCatalog.get(dataset_name).thing_classes = category_names
            print(f"Registered {len(category_names)} classes with {dataset_name}")
            
        return build_detection_train_loader(cfg, mapper=classification_mapper_pcb)

def register_pcb_only_dataset():
    global _DATASETS_REGISTERED
    
    if _DATASETS_REGISTERED["pcb_only"]:
        # Return the already registered category names
        return MetadataCatalog.get("components_only_train").thing_classes
    """
    Register a dataset containing only PCB board annotations (class 36).
    This is used to train the first stage PCB detector.
    """
    # Define category names for PCB only (only one class)
    pcb_category_names = ["PCBA"]
    
    # Register PCB-only dataset
    if "pcb_only_train" not in DatasetCatalog:
        register_coco_instances(
            "pcb_only_train",
            {"thing_classes": pcb_category_names},
            "datasets/PCBA_data.json",  # This would need to be created
            "datasets/label_images"
        )
        print("PCB-only dataset registered")
    else:
        # Update metadata even if already registered
        MetadataCatalog.get("pcb_only_train").thing_classes = pcb_category_names
        print("PCB-only dataset already registered, updated metadata")
    _DATASETS_REGISTERED["pcb_only"] = True
    return pcb_category_names

def setup_pcb_detector_config():
    """Configure the PCB detector model (stage 1)"""
    cfg = get_cfg()
    
    # Use FPN for better detection of varying sizes
    config_path = model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    # Set to detect only one class (PCB)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    
    # 1. Adjust anchor sizes to better match your large objects
    #cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[192, 256, 352, 416, 512]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0, 3.0]]  # Default ratios
    
    # 2. Adjust input size (larger input size helps detect small objects)
    cfg.INPUT.MIN_SIZE_TRAIN = 780
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 780
    cfg.INPUT.MAX_SIZE_TEST = 1333
    
    # 5. Lower score threshold for testing to catch more small objects
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5 #post detection lower is more strict
    cfg.MODEL.RPN.NMS_THRESH = 0.6 #pre detection higher more lenient lower stricter
    
    # 6. Training parameters
    cfg.SOLVER.IMS_PER_BATCH = 1  # May need to reduce to 1 if OOM errors occur
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 10000
    cfg.SOLVER.STEPS = (7500, 9500)
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.AMP.ENABLED = True
    cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE = "ciou"
    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = "ciou"
    
    cfg.DATALOADER.NUM_WORKERS = 4  # Adjust based on your CPU cores
        
    # Pin memory for faster CPU->GPU transfer
    cfg.DATALOADER.PIN_MEMORY = True
    
    # Output directory for PCB detector
    cfg.OUTPUT_DIR = "./output/pcb_detector"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    return cfg

def train_pcb_detector(resume_from=None):
    """Train the first stage PCB detector with augmentations"""
    # Register PCB-only dataset
    register_pcb_only_dataset()
    
    # Setup config
    cfg = setup_pcb_detector_config()
    cfg.DATASETS.TRAIN = ("pcb_only_train",)
    cfg.DATASETS.TEST = ()
    
    if resume_from:
        cfg.MODEL.WEIGHTS = resume_from
    
    # Create trainer with augmentations
    trainer = PCBDetectorTrainer(cfg)
    trainer.resume_or_load(resume=True if resume_from else False)
    
    trainer.train()


def detect_pcb_regions(image_path, model_path):
    """
    First stage: Detect PCB regions in an image
    
    Args:
        image_path: Path to the input image
        model_path: Path to the trained PCB detector model
        
    Returns:
        dict: PCB detection results with boxes and scores
    """
    # Register PCB-only dataset for metadata
    #register_pcb_only_dataset()
    
    # Setup config
    cfg = setup_pcb_detector_config()
    cfg.MODEL.WEIGHTS = model_path
    
    # Create predictor
    predictor = DefaultPredictor(cfg)
    
    # Read image
    image = utils.read_image(image_path, format="BGR")
    
    # Run PCB detection
    outputs = predictor(image)
    
    # Extract PCB regions
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    
    # If no PCB detected, use whole image
    if len(boxes) == 0:
        boxes = np.array([[0, 0, image.shape[1], image.shape[0]]])
        scores = np.array([1.0])
    
    return {
        "boxes": boxes,
        "scores": scores,
        "image_shape": image.shape
    }


