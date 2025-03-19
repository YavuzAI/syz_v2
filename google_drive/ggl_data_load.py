import os
import pandas as pd

# Path to dataset in Google Drive
DATASET_PATH = "/content/drive/MyDrive/datasets/brain_stroke_images/"

def read_image_df():
    """Load image paths and labels from Google Drive dataset."""
    data = []
    
    for label in ["0", "1"]:  # Assuming the dataset is organized in 0/ and 1/ subfolders
        label_path = os.path.join(DATASET_PATH, label)
        
        if not os.path.exists(label_path):
            print(f"Warning: Directory {label_path} not found!")
            continue
        
        for file in os.listdir(label_path):
            if file.lower().endswith('.dcm'):  # Ensure only DICOM files are read
                file_path = os.path.join(label_path, file)
                data.append({"file_path": file_path, "label": int(label)})
    
    df = pd.DataFrame(data)
    
    if df.empty:
        print("No images found! Check Google Drive path.")
    else:
        print(f"Loaded {len(df)} images from {DATASET_PATH}")
    
    return df

import pydicom

def read_metadata_df():
    """Read metadata from DICOM files in Google Drive dataset."""
    metadata_list = []
    
    for label in ["0", "1"]:  # Assuming the dataset is organized in 0/ and 1/ subfolders
        label_path = os.path.join(DATASET_PATH, label)
        
        if not os.path.exists(label_path):
            continue
        
        for file in os.listdir(label_path):
            if file.lower().endswith('.dcm'):
                file_path = os.path.join(label_path, file)
                
                try:
                    ds = pydicom.dcmread(file_path)
                    metadata_list.append({
                        "file_path": file_path,
                        "SliceThickness": float(ds.SliceThickness)
                    })
                except Exception as e:
                    print(f"Error reading metadata from {file_path}: {e}")
    
    return pd.DataFrame(metadata_list)

