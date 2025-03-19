import os
import pandas as pd
import pydicom
import numpy as np
from sklearn.model_selection import train_test_split
from process_img import preprocess_with_standard_windows

def read_image_df(data_dir):
    """Load image paths and labels into a DataFrame."""
    data = []
    for root, _, files in os.walk(data_dir):
        label = os.path.basename(root)
        if label not in ['0', '1']:
            continue
        for file in files:
            if file.lower().endswith('.dcm'):
                file_path = os.path.join(root, file)
                data.append({"file_path": file_path, "label": int(label)})
    return pd.DataFrame(data)


def read_metadata_df(data_dir):
    """Load metadata from DICOM files into a DataFrame."""
    metadata_list = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith('.dcm'):
                file_path = os.path.join(root, file)
                try:
                    ds = pydicom.dcmread(file_path)
                    metadata_list.append({
                        "file_path": file_path,
                        "SliceThickness": float(ds.SliceThickness),
                        "RescaleSlope": float(ds.RescaleSlope),
                        "RescaleIntercept": float(ds.RescaleIntercept),
                        "WindowCenter": [float(x) for x in ds.WindowCenter] if isinstance(ds.WindowCenter, pydicom.multival.MultiValue) else [float(ds.WindowCenter)],
                        "WindowWidth": [float(x) for x in ds.WindowWidth] if isinstance(ds.WindowWidth, pydicom.multival.MultiValue) else [float(ds.WindowWidth)]
                    })
                except Exception as e:
                    print(f"Error reading metadata from {file_path}: {e}")
    return pd.DataFrame(metadata_list)


def split_dataset(df):
    """Split dataset into train (70%), validation (20%), test (10%)."""
    train_val, test = train_test_split(df, test_size=0.10, stratify=df['label'], random_state=42)
    train, val = train_test_split(train_val, test_size=0.2222, stratify=train_val['label'], random_state=42)
    return train, val, test


def load_and_process_images(df, metadata_df):
    """Load and process images, ensuring alignment with metadata."""
    processed_images = []
    slice_thickness_list = []
    
    for _, row in df.iterrows():
        try:
            processed_image = preprocess_with_standard_windows(row['file_path'], metadata_df)
            processed_images.append(processed_image)
            slice_thickness = metadata_df.loc[metadata_df['file_path'] == row['file_path'], 'SliceThickness'].values
            slice_thickness_list.append(slice_thickness[0] if len(slice_thickness) > 0 else 0)
        except Exception as e:
            print(f"Error processing {row['file_path']}: {e}")
    
    processed_images = np.array(processed_images)
    slice_thickness_list = np.array(slice_thickness_list).reshape(-1, 1)
    
    if processed_images.shape[0] != slice_thickness_list.shape[0]:
        raise ValueError("Mismatch between processed images and metadata features. Check file paths and metadata alignment.")
    
    return processed_images, slice_thickness_list