import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class DirectionDataset(Dataset):
    """
    Dataset for component direction classification
    """
    def __init__(self, json_data, image_dir, transform=None):
        """
        Args:
            json_data: List of dictionaries with filename and label
            image_dir: Directory containing the images
            transform: Optional transform to be applied to the images
        """
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

class DirectionClassifier(nn.Module):
    """
    CNN model for direction classification
    """
    def __init__(self, num_classes=8):
        super(DirectionClassifier, self).__init__()
        
        # Use a pre-trained ResNet-18 model
        self.model = models.resnet18(pretrained=True)
        
        # Replace the last fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

def load_data(json_file):
    """
    Load the data from the JSON file
    """
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading data from {json_file}: {e}")
        return []

def train_direction_classifier(json_file, image_dir, batch_size=32, num_epochs=30, learning_rate=0.001):
    """
    Train a direction classifier using the provided data
    
    Args:
        json_file: Path to the JSON file containing the data
        image_dir: Directory containing the images
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate for the optimizer
    
    Returns:
        Trained model and evaluation results
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    data = load_data(json_file)
    if not data:
        print("No data loaded. Exiting.")
        return None, None
    
    print(f"Loaded {len(data)} samples from {json_file}")
    
    # Split data into training and validation sets
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
    
    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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
    val_dataset = DirectionDataset(val_data, image_dir, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    model = DirectionClassifier(num_classes=8)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    
    # Training loop
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    best_model_state = None
    
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
        
        # Validation phase
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
        
        # Calculate average loss and accuracy
        train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = 100.0 * train_correct / train_total
        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = 100.0 * val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Append metrics for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        # Print statistics
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            print(f'New best model saved with validation loss: {val_loss:.4f}')
    
    # Load the best model state
    model.load_state_dict(best_model_state)
    
    # Plot training and validation metrics
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('inferences/training_metrics.png')
    
    # Final evaluation on validation set
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Generate confusion matrix and classification report
    cm = confusion_matrix(all_labels, all_preds)
    class_names = ["N", "NW", "W", "SW", "S", "SE", "E", "NE"]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('inferences/confusion_matrix.png')
    
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    os.makedirs(os.path.dirname('output/direction/'), exist_ok=True)
    # Save the trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names
    }, 'output/direction/direction_classifier.pth')
    
    print("Model saved as 'direction_classifier.pth'")
    
    return model, report

def load_trained_model(model_path):
    """
    Load a trained model from a checkpoint file
    
    Args:
        model_path: Path to the model checkpoint
    
    Returns:
        Loaded model and class names
    """
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model = DirectionClassifier(num_classes=8)
    model.load_state_dict(checkpoint['model_state_dict'])
    class_names = checkpoint.get('class_names', ["N", "NW", "W", "SW", "S", "SE", "E", "NE"])
    return model, class_names