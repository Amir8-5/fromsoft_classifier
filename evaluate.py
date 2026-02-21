import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix

from model import setup_model_and_loss


def evaluate_model():
    # Configuration
    BATCH_SIZE = 32
    NUM_WORKERS = 0  # 0 for Windows compatibility
    CHECKPOINT_PATH = "best_model.pth"
    TEST_DIR = os.path.join("dataset", "test")
    DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {DEVICE_NAME}")

    # Data Loading
    test_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    if not os.path.exists(TEST_DIR):
        print(f"Test directory '{TEST_DIR}' not found. "
              "Please run prepare_dataset.py first.")
        return

    test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transforms)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    class_names = test_dataset.classes
    print(f"Evaluating on {len(test_dataset)} images across "
          f"{len(class_names)} classes: {class_names}")

    # Model Initialization
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint '{CHECKPOINT_PATH}' not found. "
              "Please run train.py first.")
        return

    model, _ = setup_model_and_loss(DEVICE_NAME)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE_NAME)
    # Support both the updated dict-style checkpoint and legacy plain state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        saved_acc = checkpoint.get("val_acc", "unknown")
        print(f"Loaded checkpoint (saved val accuracy: {saved_acc:.4f})"
              if isinstance(saved_acc, float) else
              f"Loaded checkpoint (saved val accuracy: {saved_acc})")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded legacy checkpoint (no accuracy metadata).")

    model.eval()

    # Inference Loop
    all_preds = []
    all_labels = []

    print("Running inference on test set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE_NAME)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # Metrics & Visualization
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix — FromSoftware Classifier", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_path = "confusion_matrix.png"
    plt.savefig(output_path, dpi=150)
    print(f"Confusion matrix saved to: {os.path.abspath(output_path)}")
    plt.close()


if __name__ == "__main__":
    try:
        evaluate_model()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
