import os

dataset_path = r"C:\Users\amirb\.cache\kagglehub\datasets\fraxle\images-from-fromsoftware-soulslikes\versions\1\FromSoftwareImages"

if os.path.exists(dataset_path):
    print(f"Scanning dataset at: {dataset_path}\n")
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path):
            # Count files in the directory
            files = [f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))]
            print(f"{item}: {len(files)} images")
else:
    print(f"Directory not found: {dataset_path}")
