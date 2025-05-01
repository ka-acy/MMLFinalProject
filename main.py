import os
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor
from data import DataProcessor, VideoDataset, custom_collate_fn
from model import EnhancedClassificationHead, ImprovedVideoClipModel
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Unfreeze layers for fine-tuning
for name, param in clip_model.named_parameters():
    if "text_model.encoder.layers" in name or "vision_model" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Process data
data_processor = DataProcessor(
    segmented_manifest_path="/mnt/e/MML_RESEARCH/manifest/mlb-youtube-segmented.json",
    captions_manifest_path="/mnt/e/MML_RESEARCH/manifest/mlb-youtube-captions.json",
    segmented_folder="/mnt/e/MML_RESEARCH/SEGMENTED/segmented",
) # Your data paths
processed_data = data_processor.process_data()

train_df, valid_df = train_test_split(
    processed_data[processed_data["subset"] == "training"], test_size=0.2
)
test_df = processed_data[processed_data["subset"] == "testing"]

train_dataset = VideoDataset(train_df, processor)
valid_dataset = VideoDataset(valid_df, processor)
test_dataset = VideoDataset(test_df, processor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, collate_fn=custom_collate_fn)

# Initialize model
num_labels = len(processed_data["labels"].unique())
classification_head = EnhancedClassificationHead(input_dim=512, num_labels=num_labels).to(device)
video_clip_model = ImprovedVideoClipModel(
    clip_model=clip_model,
    classification_head=classification_head,
    dim=512  # Specify dimension explicitly
).to(device)

# Optimizer and scheduler
optimizer = torch.optim.AdamW(video_clip_model.parameters(), lr=5e-5, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

def evaluate_model(model, data_loader, loss_fn):
    """
    Evaluate the model on a dataset and compute accuracy, precision, recall, F1-score, and loss.

    Args:
        model: The trained model to evaluate.
        data_loader: DataLoader for the evaluation dataset.
        loss_fn: Loss function to compute validation loss.

    Returns:
        Metrics as a dictionary containing accuracy, precision, recall, F1-score, and loss.
    """
    model.eval()  # Set the model to evaluation mode
    all_predictions, all_labels = [], []
    num_correct, num_total = 0, 0
    total_loss = 0

    # Loop through the data loader
    for features, captions, labels in data_loader:
        features, labels = features.to(device), labels.to(device)

        with torch.no_grad():
            logits = model(features, captions)  # Get predictions
            predictions = torch.argmax(logits, dim=1)
            loss = loss_fn(logits, labels)  # Compute loss

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        num_correct += (predictions == labels).sum().item()
        num_total += len(labels)
        total_loss += loss.item() * len(labels)

    # Compute metrics
    accuracy = num_correct / num_total
    precision = precision_score(all_labels, all_predictions, average="weighted")
    recall = recall_score(all_labels, all_predictions, average="weighted")
    f1 = f1_score(all_labels, all_predictions, average="weighted")
    avg_loss = total_loss / num_total

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, Loss: {avg_loss:.4f}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "loss": avg_loss
    }

def plot_confusion_matrix(model, data_loader, class_names):
    """
    Create and plot confusion matrix for model predictions.
    
    Args:
        model: trained model
        data_loader: DataLoader containing validation/test data
        class_names: list of class names for labels
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, captions, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            logits = model(features, captions)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('confusion_matrix.png')
    plt.close()


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

# Training loop with early stopping
early_stopping = EarlyStopping(patience=5, verbose=True)

for epoch in range(20):  # Replace with num_epochs if variable
    video_clip_model.train()
    for features, captions, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = video_clip_model(features, captions)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
    
    scheduler.step()
    print(f"Epoch {epoch + 1}, Validation Metrics:")
    metrics = evaluate_model(video_clip_model, valid_loader, loss_fn)
    
    # Early stopping
    val_loss = metrics['loss']
    early_stopping(val_loss, video_clip_model)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break


# Load the last checkpoint with the best model
video_clip_model.load_state_dict(torch.load('checkpoint.pt'))
plot_confusion_matrix(
    video_clip_model, 
    test_loader,
    class_names=list(processed_data["labels"].unique())
)
# Test evaluation
print("Final Test Metrics:")
evaluate_model(video_clip_model, test_loader, loss_fn)
