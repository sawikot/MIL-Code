import glob
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import random

def get_mpp(slide):
    # Extract MPP values from the slide properties
    mpp_x = float(slide.properties.get('openslide.mpp-x', 0))
    mpp_y = float(slide.properties.get('openslide.mpp-y', 0))
    
    # Check if MPP values are the same for both dimensions
    if mpp_x == mpp_y:
        mpp = mpp_x
    else:
        print("MPP values differ for x and y dimensions. Please verify the image metadata.")
        mpp = (mpp_x, mpp_y)
    return mpp


SUPPORTED_EXTENSIONS = ['tiff', 'svs', 'mrxs']

def get_folder_file_list(folder_path, extensions=SUPPORTED_EXTENSIONS):
    folder_file_list = []

    for ext in extensions:
        file_paths = glob.glob(f"{folder_path}/**/*.{ext}", recursive=True)

        for path in file_paths:
            folder_name = os.path.basename(os.path.dirname(path))
            file_name = os.path.splitext(os.path.basename(path))[0]  # Remove the extension
            folder_file_list.append([folder_name, file_name])

    return folder_file_list





def get_wsi_splits(manifest_path: str, root_dir: str, seed: int = 42):
    # Set seed for reproducibility
    random.seed(seed)

    # Read CSV manifest
    df = pd.read_csv(manifest_path)

    # Initialize lists
    train_files = []
    val_files = []

    # Process training and validation splits
    for _, row in df.iterrows():
        class_dir = os.path.join(root_dir, row['label'])
        npz_path = os.path.join(class_dir, f"{row['slide']}.npz")
        
        if os.path.exists(npz_path):
            if row['type'] == 'training':
                train_files.append(npz_path)
            elif row['type'] == 'validation':
                val_files.append(npz_path)
    
    # Shuffle the file lists
    random.shuffle(train_files)
    random.shuffle(val_files)

    return train_files, val_files






def split_dataset_to_csv(root_dir, train_ratio=0.8, output_csv="dataset_split"):
    """
    Splits a dataset into training and validation sets and saves the result in a CSV file.

    Parameters:
        root_dir (str): The root directory containing class folders.
        train_ratio (float): The ratio of training data (e.g., 0.8 for 80% training, 20% validation).
        output_csv (str): The name of the output CSV file. Default is 'dataset_split.csv'.

    Returns:
        None
    """
    # Validate the training ratio
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be a float between 0 and 1.")

    # Initialize an empty list to store the data
    data = []

    # Loop through each class folder
    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)
        
        # Check if it's a directory
        if os.path.isdir(class_path):
            # Get all file names in the class folder
            files = os.listdir(class_path)
            
            # Split the files into training and validation sets
            train_files, val_files = train_test_split(files, test_size=1-train_ratio, random_state=42)
            
            # Add training files to the data list
            for file in train_files:
                # Remove the file extension
                slide_name = os.path.splitext(file)[0]
                data.append({"slide": slide_name, "type": "training", "label": class_name})
            
            # Add validation files to the data list
            for file in val_files:
                # Remove the file extension
                slide_name = os.path.splitext(file)[0]
                data.append({"slide": slide_name, "type": "validation", "label": class_name})

    # Convert the data list to a DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv+".csv", index=False)

    print(f"Dataset split saved to {output_csv}.csv")
