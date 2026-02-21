import os
import shutil
import random
import json
import numpy as np
import kagglehub
from sklearn.utils.class_weight import compute_class_weight
from dotenv import load_dotenv

load_dotenv()

def prepare_dataset():
    # 1. Download/Get Dataset Path
    print("Step 1: Downloading/Verifying Dataset...")
    try:
        path = kagglehub.dataset_download("fraxle/images-from-fromsoftware-soulslikes")
        print("Dataset path:", path)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return

    # The actual images are in a subdirectory 'FromSoftwareImages' based on inspection
    source_dir = os.path.join(path, "versions", "1", "FromSoftwareImages")
    
    # Verify source directory exists
    if not os.path.exists(source_dir):
        # Fallback to checking without 'versions/1' if structure varies
        source_dir_alt = os.path.join(path, "FromSoftwareImages")
        if os.path.exists(source_dir_alt):
             source_dir = source_dir_alt
        else:
             print(f"Error: Source directory {source_dir} does not exist.")
             # List contents to help debugging
             print(f"Contents of {path}: {os.listdir(path)}")
             return

    # 2. Create Local Directory Structure
    base_dir = "dataset"
    splits = ["train", "val", "test"]
    # Get class names from source directory
    class_names = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    class_names.sort()

    print(f"Found classes: {class_names}")

    print("Step 2: Creating Directory Structure...")
    if os.path.exists(base_dir):
        print(f"Removing existing {base_dir}...")
        shutil.rmtree(base_dir) # Clean start
    os.makedirs(base_dir)

    for split in splits:
        for class_name in class_names:
            os.makedirs(os.path.join(base_dir, split, class_name), exist_ok=True)

    # 3. Stratified Split and Copy
    print("Step 3: Splitting and Copying Data...")
    split_ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
    
    # Valid image extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    for class_name in class_names:
        class_src_dir = os.path.join(source_dir, class_name)
        # Filter for valid image files
        files = [f for f in os.listdir(class_src_dir) 
                 if os.path.isfile(os.path.join(class_src_dir, f)) 
                 and os.path.splitext(f)[1].lower() in valid_extensions]
        
        random.shuffle(files)
        
        n_total = len(files)
        n_train = int(n_total * split_ratios["train"])
        n_val = int(n_total * split_ratios["val"])
        # n_test is the rest
        
        train_files = files[:n_train]
        val_files = files[n_train:n_train+n_val]
        test_files = files[n_train+n_val:]
        
        # Function to copy files
        def copy_files(file_list, destination_split):
            dest_dir = os.path.join(base_dir, destination_split, class_name)
            count = 0
            for f in file_list:
                src = os.path.join(class_src_dir, f)
                dst = os.path.join(dest_dir, f)
                try:
                    shutil.copy2(src, dst) # copy2 preserves metadata
                    count += 1
                except Exception as e:
                    print(f"Skipping corrupted/error file {f}: {e}")
            return count

        n_train_copied = copy_files(train_files, "train")
        n_val_copied = copy_files(val_files, "val")
        n_test_copied = copy_files(test_files, "test")
        
        print(f"Class '{class_name}': {n_train_copied} train, {n_val_copied} val, {n_test_copied} test")

    # 4. Calculate Class Weights
    print("Step 4: Calculating Class Weights...")
    train_dir = os.path.join(base_dir, "train")
    y_train = []
    class_counts = {}

    for class_name in class_names:
        class_path = os.path.join(train_dir, class_name)
        count = len(os.listdir(class_path))
        class_counts[class_name] = count
        y_train.extend([class_name] * count)

    # Calculate weights
    # classes are unique labels in y_train
    unique_classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train)
    class_weights_dict = {cls: float(weight) for cls, weight in zip(unique_classes, weights)}

    # 5. Export Configuration
    print("Step 5: Exporting Configuration...")
    output_file = "class_weights.json"
    with open(output_file, 'w') as f:
        json.dump(class_weights_dict, f, indent=4)

    print("\n--- Summary ---")
    print("Class Counts (Train Split):")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count}")
    print("\nCalculated Class Weights:")
    for cls, weight in class_weights_dict.items():
        print(f"  {cls}: {weight:.4f}")
    print(f"\nConfiguration saved to {output_file}")

if __name__ == "__main__":
    prepare_dataset()
