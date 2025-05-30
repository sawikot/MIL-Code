import tensorflow as tf
import numpy as np
import torch
import os
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import glob
from tqdm import tqdm


def get_folder_file_list(folder_path, file_type='tiff'):
    file_paths = glob.glob(f"{folder_path}/**/*."+file_type, recursive=True)
    folder_file_list = []

    for path in file_paths:
        folder_name = os.path.basename(os.path.dirname(path))
        file_name = os.path.splitext(os.path.basename(path))[0]  # Remove the .tiff extension
        folder_file_list.append([folder_name, file_name])

    return folder_file_list


class TFRecordParser:
    @staticmethod
    def parse_tfrecord_fn(example_proto):
        feature_description = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "x_start": tf.io.FixedLenFeature([], tf.int64),
            "y_start": tf.io.FixedLenFeature([], tf.int64),
        }
        
        example = tf.io.parse_single_example(example_proto, feature_description)
        image = tf.image.decode_png(example["image"], channels=3)
        x_start = example["x_start"]
        y_start = example["y_start"]
        
        return image, x_start, y_start


class FeatureExtractor:
    def __init__(self, model, transform):
        self.model = model
        self.transform = transform
    
    def extract_features(self, dataset,record_count, model_type):
        all_features = []
        all_start = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for img, x_start, y_start in tqdm(dataset, total=record_count, desc="Extracting features"):
            img_np = img.numpy()
            pil_image = Image.fromarray(img_np)
            image_tensor = self.transform(pil_image).unsqueeze(0).to(device)  # Add batch dimension

            with torch.no_grad():
                features = self.model(image_tensor)

            if model_type == "virchow" or model_type == "virchow2":
                class_token = features[:, 0]
                patch_tokens = features[:, 1:]
                embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1).squeeze().cpu().numpy()
                all_features.append(embedding)
            else:
                features_np = features.squeeze().cpu().numpy()
                all_features.append(features_np)

            all_start.append((x_start.numpy(), y_start.numpy()))

        
        return all_features, all_start

    def save_features(self, features, coordinates, output_file):
        np.savez(output_file, array1=features, array2=coordinates)
        print(f"Feature extraction complete! Saved to {output_file}")



def count_records_in_tfrecord(tfrecord_data):
    return sum(1 for _ in tfrecord_data)



class TFRecordProcessor:
    def __init__(self, input_dir, feature_extraction_model, transform, output_dir, model_type):
        self.input_dir = input_dir
        self.feature_extractor = FeatureExtractor(feature_extraction_model, transform)
        self.output_dir = output_dir
        self.model_type=model_type
        os.makedirs(output_dir, exist_ok=True)
    
    def process(self):
        file_paths = get_folder_file_list(self.input_dir, file_type='tfrecord')
        for class_type, image_name in file_paths:
            slide_path = os.path.join(self.input_dir, class_type, image_name + '.tfrecord')
            output_file = os.path.join(self.output_dir, class_type)
            os.makedirs(output_file, exist_ok=True)
            output_file = os.path.join(output_file, f"{image_name}.npz")
            dataset = tf.data.TFRecordDataset(slide_path)
            record_count = count_records_in_tfrecord(dataset)
            dataset = dataset.map(TFRecordParser.parse_tfrecord_fn)
            
            features, coordinates = self.feature_extractor.extract_features(dataset,record_count, self.model_type)
            self.feature_extractor.save_features(features, coordinates, output_file)







import os
import torch
from torchvision import transforms
import timm
import torch.nn as nn
from huggingface_hub import login
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked

class FeatureModel:
    def __init__(self, model_type, model_path=None, num_classes=4, image_size=(224, 224), hf_token=None):
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_type == "retccl":
            self.model = models.resnet50(num_classes=num_classes)
            pretext_model = torch.load(model_path, map_location=device)
            self.model.fc = nn.Identity()  # Remove the final fully connected layer
            self.model.load_state_dict(pretext_model, strict=True)
        elif model_type == "uni":
            self.model = timm.create_model(
                "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=num_classes, dynamic_img_size=True
            )
            pretext_model = torch.load(model_path, map_location=device)
            self.model.head = nn.Identity()
            self.model.load_state_dict(pretext_model, strict=True)
        elif model_type == "uni-2h":
            # Define the model parameters for UNI2-h
            timm_kwargs = {
                'model_name': 'vit_giant_patch14_224',
                'img_size': 224, 
                'patch_size': 14, 
                'depth': 24,
                'num_heads': 24,
                'init_values': 1e-5, 
                'embed_dim': 1536,
                'mlp_ratio': 2.66667 * 2,
                'num_classes': num_classes,  # Set to 0 for no classification head
                'no_embed_class': True,
                'mlp_layer': timm.layers.SwiGLUPacked, 
                'act_layer': torch.nn.SiLU, 
                'reg_tokens': 8, 
                'dynamic_img_size': True
            }
            self.model = timm.create_model(pretrained=False, **timm_kwargs)
            pretext_model = torch.load(model_path, map_location=device)
            self.model.head = nn.Identity()
            self.model.load_state_dict(pretext_model, strict=True)
        elif model_type == "virchow":
            if hf_token:
                login(token=hf_token)  # Authenticate with Hugging Face if a token is provided
            
            self.model = timm.create_model(
                "hf-hub:paige-ai/Virchow",
                pretrained=True,
                mlp_layer=SwiGLUPacked,
                act_layer=torch.nn.SiLU
            )
            self.model = self.model.to(device)
            self.model.eval()

            # Load transformation based on model configuration
            self.transform = create_transform(**resolve_data_config(self.model.pretrained_cfg, model=self.model))
        elif model_type == "virchow2":
            if hf_token:
                login(token=hf_token)  # Authenticate with Hugging Face if a token is provided
            
            self.model = timm.create_model(
                "hf-hub:paige-ai/Virchow2",
                pretrained=True,
                mlp_layer=SwiGLUPacked,
                act_layer=torch.nn.SiLU
            )
            self.model = self.model.to(device)
            self.model.eval()

            # Load transformation based on model configuration
            self.transform = create_transform(**resolve_data_config(self.model.pretrained_cfg, model=self.model))

        else:
            raise ValueError("Invalid model type. Choose either 'retccl', 'uni', 'uni-2h', or 'virchow'")

        self.model = self.model.to(device)
        self.model.eval()
    
    def get_model(self):
        return self.model
    
    def get_transform(self):
        return self.transform


