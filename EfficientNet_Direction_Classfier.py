import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import os
import json
import cv2
import matplotlib.pyplot as plt

_DIRECTION_CLASSIFIER = None

def get_direction_classifier(model_path):
    """Get direction classifier with singleton pattern"""
    global _DIRECTION_CLASSIFIER
    
    if _DIRECTION_CLASSIFIER is None:
        _DIRECTION_CLASSIFIER = EfficientNetDirectionClassifier(model_path)
        print("Direction classifier loaded from:", model_path)
    
    return _DIRECTION_CLASSIFIER

class DirectionDataset(Dataset):
    """Dataset for component direction classification"""
    def __init__(self, json_data, image_dir, transform=None):
        self.data = json_data
        self.image_dir = image_dir
        self.transform = transform
        
        # Create a mapping from direction labels to indices
        self.direction_to_idx = {
            "N": 0, "NE": 1, "E": 2, "SE": 3, 
            "S": 4, "SW": 5, "W": 6, "NW": 7
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_dir, item["filename"])
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image and the label if the image can't be loaded
            image = Image.new("RGB", (64, 64), color=(0, 0, 0))
        
        label = item["label"]
        label_idx = self.direction_to_idx[label]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label_idx

class EfficientNetModel(nn.Module):
    """EfficientNet-based model for direction classification"""
    def __init__(self, model_name='efficientnet_b0', num_classes=8, pretrained=True):
        super().__init__()
        
        # Load the pre-trained EfficientNet
        self.model = timm.create_model(model_name, pretrained=pretrained)
        
        # Replace the classifier head
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, num_classes)
        
        # Optional: Add dropout for regularization
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.model.forward_features(x)
        # Apply global pooling if needed (depends on model architecture)
        if x.dim() > 2:
            x = self.model.global_pool(x)
            # Optional flatten operation if needed
            if x.dim() > 2:
                x = x.flatten(1)
        x = self.dropout(x)
        x = self.model.classifier(x)
        return x

def load_data(json_file):
    """Load the data from the JSON file"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading data from {json_file}: {e}")
        return []

def train_efficient_direction_classifier(
    json_file, 
    image_dir, 
    model_name='efficientnet_b0',
    batch_size=32, 
    num_epochs=30, 
    learning_rate=0.001,
    val_split=0.2,  # Added parameter for validation split
    use_validation=True,  # Added parameter to toggle validation
    output_path='output/direction/efficient_direction_classifier.pth'
):
    """
    Train an EfficientNet-based direction classifier
    
    Args:
        json_file: Path to the JSON file with training data
        image_dir: Directory containing the images
        model_name: Name of the EfficientNet model to use
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        val_split: Fraction of data to use for validation (if use_validation is True)
        use_validation: Whether to use a validation set
        output_path: Path to save the trained model
        
    Returns:
        model: Trained model
        class_names: List of class names
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for training EfficientNet direction classifier")
    
    # Load data
    data = load_data(json_file)
    if not data:
        print("No data loaded. Exiting.")
        return None, None
    
    print(f"Loaded {len(data)} samples from {json_file}")
    
    # Split data into training and validation sets if validation is enabled
    if use_validation:
        train_data, val_data = train_test_split(data, test_size=val_split, random_state=42)
        print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
    else:
        train_data = data
        val_data = []  # Empty validation set
        print(f"Using all {len(train_data)} samples for training (no validation)")
    
    # Define transformations with improved augmentations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation((-5, 5)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), interpolation=InterpolationMode.NEAREST),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = DirectionDataset(train_data, image_dir, transform=train_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Create validation loader if using validation
    if use_validation:
        val_dataset = DirectionDataset(val_data, image_dir, transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    model = EfficientNetModel(model_name=model_name, num_classes=8, pretrained=True)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Use AdamW optimizer which often works better with EfficientNets
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler with cosine annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=learning_rate/100)

    # Initialize arrays to store metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    # Initialize variables to track best model
    best_val_loss = float('inf')
    best_model_state = None
    class_names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    
    # Flag to determine if we're tracking validation loss
    track_val_loss = use_validation
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Calculate average training loss and accuracy
        avg_train_loss = train_loss / len(train_dataset)
        train_accuracy = 100.0 * train_correct / train_total
        
        # Store training metrics
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation phase (if enabled)
        if use_validation:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Calculate average validation loss and accuracy
            avg_val_loss = val_loss / len(val_dataset)
            val_accuracy = 100.0 * val_correct / val_total
            
            # Store validation metrics
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)
            
            # Print statistics
            print(f'Epoch {epoch+1}/{num_epochs}: '
                  f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
                  f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
            
            # Save the best model based on validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                print(f'New best model saved with validation loss: {avg_val_loss:.4f}')
        else:
            # If not using validation, just print training stats
            print(f'Epoch {epoch+1}/{num_epochs}: '
                  f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
            
            # When not using validation, save the model based on training loss
            # We could use other criteria like saving the last model or saving every N epochs
            if epoch == num_epochs - 1:  # Save the final model
                best_model_state = model.state_dict().copy()
                print(f'Final model saved with training loss: {avg_train_loss:.4f}')
        
        # Update learning rate
        scheduler.step()
    
    # Load the best model state (if validation was used)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Create plots based on available metrics
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    if use_validation:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training' + (' and Validation' if use_validation else '') + ' Loss')
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    if use_validation:
        plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training' + (' and Validation' if use_validation else '') + ' Accuracy')
    
    plt.tight_layout()
    plt.savefig('inferences/training_metrics.png')
    
    # Save the trained model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model_data = {
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'model_name': model_name
    }
    
    # Add validation metrics if available
    if use_validation:
        model_data['validation'] = {
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_val_loss': best_val_loss
        }
    
    torch.save(model_data, output_path)
    print(f"Model saved as '{output_path}'")
    
    return model, class_names

class EfficientNetDirectionClassifier:
    """Classifier for component directions using EfficientNet"""
    def __init__(self, model_path):
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device} for direction classifier")
        
        # Load the model
        checkpoint = torch.load(model_path, map_location=self.device)
        model_name = checkpoint.get('model_name', 'efficientnet_b0')
        
        # Create model with same architecture
        self.model = EfficientNetModel(model_name=model_name, num_classes=8)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.class_names = checkpoint.get('class_names', ["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
        
        # Define image transformation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Loaded EfficientNet direction classifier with classes: {self.class_names}")
    
    def predict_direction(self, image):
        # Ensure image is in PIL format
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Transform image - creates a tensor
        image_tensor = self.transform(image)
        # Now we can unsqueeze the tensor (not the PIL image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            prob, predicted = torch.max(probabilities, 1)
        
        direction = self.class_names[predicted.item()]
        
        confidence = prob.item()
        return direction, confidence
    

def extract_component_image(image, bbox):
    """Extract component image from full image using bounding box"""
    x1, y1, x2, y2 = map(int, bbox)
    
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
    
    return component_image

def classify_component_directions(image_path, detection_results, classifier, target_categories=None):
    """Classify the directions of detected components"""
    if target_categories is None:
        # Default to capacitor electrolytic (3) and connector (17)
        target_categories = [3, 17]
    
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get component detections
    component_boxes = detection_results["component_boxes"]
    component_classes = detection_results["component_classes"]
    component_scores = detection_results["component_scores"]
    
    # Initialize arrays for direction predictions
    direction_predictions = [None] * len(component_classes)
    direction_confidences = [0.0] * len(component_classes)
    
    # Classify directions for target components
    for i, (box, class_id, score) in enumerate(zip(component_boxes, component_classes, component_scores)):
        if class_id in target_categories:
            # Extract component image
            component_image = extract_component_image(image, box)
            
            # Predict direction
            direction, confidence = classifier.predict_direction(component_image)
            
            # Store prediction
            direction_predictions[i] = direction
            direction_confidences[i] = confidence
    
    # Update detection results
    updated_results = detection_results.copy()
    updated_results["component_directions"] = direction_predictions
    updated_results["direction_confidences"] = direction_confidences
    
    return updated_results

