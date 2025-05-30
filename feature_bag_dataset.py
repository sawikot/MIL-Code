import os
import numpy as np
import torch
from torch.utils.data import Dataset

class FeatureBagDataset(Dataset):
    def __init__(self, npz_files: list, bag_size: int = 10):
        self.npz_files = npz_files
        self.bag_size = bag_size
        self.bags = []
        self.class_mapping = self._generate_class_mapping()
        self._create_bags()
    
    def _generate_class_mapping(self):
        """
        Dynamically generate a mapping of class names to class indices
        based on the folder names or file paths.
        """
        class_names = set()
        for path in self.npz_files:
            class_name = self._extract_class_name(path)
            class_names.add(class_name)
        
        # Sort class names to ensure consistent mapping
        sorted_class_names = sorted(class_names)
        return {name: idx for idx, name in enumerate(sorted_class_names)}
    
    def _extract_class_name(self, path):
        """
        Extract the class name from the file path.
        Assumes the class name is part of the folder or file name (e.g., 'class_00').
        """
        # Example: If path is 'data/class_00/sample.npz', this extracts 'class_00'
        return os.path.basename(os.path.dirname(path))
    
    def _create_bags(self):
        for npz_path in self.npz_files:
            data = np.load(npz_path)
            features = data['array1']
            
            # Create bags from this WSI's features
            for i in range(0, len(features), self.bag_size):
                bag_features = features[i:i + self.bag_size]
                
                if len(bag_features) < self.bag_size:
                    pad_size = self.bag_size - len(bag_features)
                    indices = np.random.choice(len(bag_features), pad_size)
                    bag_features = np.vstack([bag_features, bag_features[indices]])
                
                class_label = self._get_class_from_path(npz_path)
                self.bags.append((bag_features, len(bag_features), class_label))
    
    def _get_class_from_path(self, path):
        """
        Get the class label (integer) from the file path using the class mapping.
        """
        class_name = self._extract_class_name(path)
        if class_name not in self.class_mapping:
            raise ValueError(f"Unknown class in path: {path}")
        return self.class_mapping[class_name]
    
    @property
    def classes(self):
        """
        Return the list of class names in the dataset.
        """
        # Reverse the class_mapping to get class names in order of their indices
        return [class_name for class_name, _ in sorted(self.class_mapping.items(), key=lambda x: x[1])]
    
    def __len__(self):
        return len(self.bags)
    
    def __getitem__(self, idx):
        features, length, label = self.bags[idx]
        return (torch.FloatTensor(features), 
                torch.tensor(length, dtype=torch.long),
                torch.tensor(label, dtype=torch.long))