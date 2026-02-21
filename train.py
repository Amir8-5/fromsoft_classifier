import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import time

# Import layout from model.py
from model import setup_model_and_loss

def train_model():
    # Configuration
    BATCH_SIZE = 32 
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 8
    # Set num_workers to 0 for Windows compatibility to avoid multiprocessing issues
    NUM_WORKERS = 0 
    
    DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE_NAME}")

    # Step 1: Data Augmentation & Loaders
    # Train Transforms
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Val/Test Transforms (Test not used in training loop, but useful to have)
    val_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets
    train_dir = os.path.join("dataset", "train")
    val_dir = os.path.join("dataset", "val")
    
    if not os.path.exists(train_dir):
        print(f"Dataset directory {train_dir} not found. Please run prepare_dataset.py first.")
        return

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    # DataLoaders
    # shuffle=True only for training
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"Training on {len(train_dataset)} images, Validating on {len(val_dataset)} images.")

    # Step 2: Initialize Training Components
    model, criterion = setup_model_and_loss(DEVICE_NAME)
    
    # Optimizer - only parameters that require gradients (transfer learning, base layers frozen)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    # Step 3: Training & Validation Loop
    CHECKPOINT_PATH = "best_model.pth"

    # Load best accuracy from existing checkpoint so we only overwrite if we improve on it
    best_val_acc = 0.0
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE_NAME)
        # Support both plain state_dict saves and dict-style checkpoints
        if isinstance(checkpoint, dict) and "val_acc" in checkpoint:
            best_val_acc = checkpoint["val_acc"]
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded existing checkpoint with val accuracy: {best_val_acc:.4f}")
        else:
            # Legacy plain state_dict — no accuracy recorded, start from 0
            print(f"Found existing checkpoint (no accuracy recorded). Will overwrite if any improvement is made.")

    print(f"Starting training for {NUM_EPOCHS} epochs...")
    
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 10)

        # Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(DEVICE_NAME)
            labels = labels.to(DEVICE_NAME)

            optimizer.zero_grad()

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            # Backward
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            if batch_idx % 100 == 0:
               print(f"Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE_NAME)
                labels = labels.to(DEVICE_NAME)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_dataset)
        val_acc = val_corrects.double() / len(val_dataset)

        epoch_time = time.time() - start_time
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} Time: {epoch_time:.0f}s")

        # Step 4: Early Stopping & Checkpointing
        if val_acc > best_val_acc:
            print(f"Validation accuracy improved from {best_val_acc:.4f} to {val_acc:.4f}. Saving model...")
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_acc": best_val_acc,
            }, CHECKPOINT_PATH)
        else:
            print(f"Val accuracy ({val_acc:.4f}) did not improve over best ({best_val_acc:.4f}). Model not saved.")
            
    print(f"\nTraining complete. Best Validation Accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"An error occurred: {e}")
        # Print full traceback for debugging
        import traceback
        traceback.print_exc()
