# src/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

from dataset import CustomImageDataset
from model import ConvNextCBAMClassifier
from loss import FocalLoss
from utils import load_config, load_metadata, get_file_paths, compute_alpha

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load configuration
config = load_config()

# Define transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Define metadata columns (update as necessary)
metadata_columns = [
    'age', 'smoke', 'drink', 'pesticide', 'gender', 'skin_cancer_history',
    'cancer_history', 'has_piped_water', 'has_sewage_system',
    'background_father_10', 'background_father_12', 'background_father_2',
    'background_father_4', 'background_father_6', 'background_father_7',
    'background_father_9', 'background_father_Other', 'background_mother_0',
    'background_mother_10', 'background_mother_2', 'background_mother_3',
    'background_mother_4', 'background_mother_7', 'background_mother_8',
    'background_mother_Other', 'region_0', 'region_1', 'region_10',
    'region_11', 'region_12', 'region_13', 'region_2', 'region_3',
    'region_4', 'region_5', 'region_6', 'region_7', 'region_8', 'region_9',
    'itch_1.0', 'grew_1.0', 'hurt_1.0', 'changed_1.0', 'bleed_1.0',
    'elevation_1.0', 'fitspatrick'
]

# --- Load Training, Validation, and Test Data ---
train_img_ids, train_metadata, train_labels = load_metadata(config['data']['train_csv'], metadata_columns)
val_img_ids, val_metadata, val_labels = load_metadata(config['data']['val_csv'], metadata_columns)
test_img_ids, test_metadata, test_labels = load_metadata(config['data']['test_csv'], metadata_columns)

train_paths = get_file_paths(train_img_ids, config['data']['train_img_dir'])
val_paths = get_file_paths(val_img_ids, config['data']['val_img_dir'])
test_paths = get_file_paths(test_img_ids, config['data']['test_img_dir'])

# Create Dataset instances
train_dataset = CustomImageDataset(train_paths, train_metadata, train_labels, transform=transform)
val_dataset = CustomImageDataset(val_paths, val_metadata, val_labels, transform=transform)
test_dataset = CustomImageDataset(test_paths, test_metadata, test_labels, transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'],
                          shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'],
                        shuffle=False, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'],
                         shuffle=False, num_workers=8, pin_memory=True)

# Compute alpha for Focal Loss
# For example: assign multiplier 4 for class 2 and 2 for class 4.
multiplier_dict = {2: 4, 4: 2}
num_classes = config['model']['num_classes']
alpha = compute_alpha(train_labels, num_classes, multiplier_dict)
print("Adjusted alpha:", alpha)

# Define Focal Loss with gamma and alpha
gamma = 2.5  # adjust as needed
criterion = FocalLoss(gamma=gamma, alpha=alpha, reduction='mean', device=device).to(device)

# Instantiate the model
model = ConvNextCBAMClassifier(
    num_classes=num_classes,
    metadata_input_size=config['model']['metadata_input_size'],
    metadata_output_size=config['model']['metadata_output_size']
)

# Use DataParallel if multiple GPUs are available
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)

# Setup optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'],
                        weight_decay=config['training']['weight_decay'])
scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['t_max'])

# --- Training and Evaluation Functions ---
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, metadata, labels in loader:
        images, metadata, labels = images.to(device), metadata.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images, metadata)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total

def evaluate_epoch(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, metadata, labels in loader:
            images, metadata, labels = images.to(device), metadata.to(device), labels.to(device)
            outputs = model(images, metadata)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    # Compute confusion matrix and classification report
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, digits=4, output_dict=True)
    return running_loss / total, correct / total, cm, report, all_labels, all_preds

# --- Training Loop with Early Stopping ---
num_epochs = config['training']['num_epochs']
patience = config['training']['patience']
cooldown = config['training']['cooldown']
min_delta = config['training']['min_delta']

best_val_loss = float('inf')
best_f1 = 0.0
patience_counter = 0
cooldown_counter = 0
val_loss_history = []
f1_history = []
window_size = 3
metrics = []
early_stop = False

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc, cm, report, _, _ = evaluate_epoch(model, val_loader, criterion)
    
    # Optionally print per-class metrics (report is a dict)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Scheduler step: using the val_loss
    scheduler.step(val_loss)
    
    # Save metrics
    weighted = report.get('weighted avg', {})
    current_f1 = weighted.get('f1-score', 0.0)
    metrics.append({
        'epoch': epoch+1,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'weighted_f1': current_f1
    })

    # Moving average for early stopping
    val_loss_history.append(val_loss)
    f1_history.append(current_f1)
    if len(val_loss_history) > window_size:
        val_loss_history.pop(0)
        f1_history.pop(0)
    avg_val_loss = sum(val_loss_history) / len(val_loss_history)
    avg_f1 = sum(f1_history) / len(f1_history)

    improved_loss = best_val_loss - avg_val_loss
    improved_f1 = avg_f1 - best_f1

    if improved_loss > min_delta or improved_f1 > min_delta:
        if improved_loss > min_delta:
            best_val_loss = avg_val_loss
        if improved_f1 > min_delta:
            best_f1 = avg_f1
        patience_counter = 0
        cooldown_counter = 0
        # Save the best model
        torch.save(model.state_dict(), 'best_model_convnextcbam.pth')
        best_cm = cm
        best_report = classification_report(*evaluate_epoch(model, val_loader, criterion)[4:6], digits=4)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            cooldown_counter += 1
            print(f"Patience exhausted. Cooldown counter: {cooldown_counter}/{cooldown}")
            if cooldown_counter >= cooldown:
                print("Early stopping triggered.")
                early_stop = True
                break

if early_stop:
    print("Training stopped due to early stopping.")
else:
    print("Training completed.")

# Save training metrics to CSV
pd.DataFrame(metrics).to_csv('training_metrics_convnextcbam.csv', index=False)

# Save the best confusion matrix and classification report
pd.DataFrame(best_cm).to_csv('validation_confusion_matrix_convnextcbam.csv', index=False)
with open('validation_classification_report_convnextcbam.txt', 'w') as f:
    f.write(best_report)

# --- Evaluate on Test Set ---
model.load_state_dict(torch.load('best_model_convnextcbam.pth'))
model = model.to(device)
test_loss, test_acc, cm, report, all_labels, all_preds = evaluate_epoch(model, test_loader, criterion)
print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
print("Test Classification Report:")
print(classification_report(all_labels, all_preds, digits=4))
pd.DataFrame(cm).to_csv('test_confusion_matrix_convnextcbam.csv', index=False)
with open('test_classification_report_convnextcbam.txt', 'w') as f:
    f.write(classification_report(all_labels, all_preds, digits=4))
