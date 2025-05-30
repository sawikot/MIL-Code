import os
import openslide
import numpy as np
import cv2
import math
import io
import tensorflow as tf
from skimage import filters, morphology
from utility import get_mpp, get_folder_file_list




class SlideProcessor:
    def __init__(self, slide, desired_width=1000):
        self.slide = slide
        self.desired_width = desired_width
        self.low_res_img, self.scale_factor = self.create_low_res_image()
        self.tissue_mask = self.create_tissue_mask()

    def create_low_res_image(self):
        """Create a low-resolution version of the slide for segmentation"""
        width, height = self.slide.dimensions
        scale_factor = self.desired_width / width
        low_res_size = (self.desired_width, int(height * scale_factor))
        low_res_img = self.slide.get_thumbnail(low_res_size)
        return np.array(low_res_img), scale_factor

    def create_tissue_mask(self):
        """Create tissue mask using Otsu's method with hole filling"""
        gray = cv2.cvtColor(self.low_res_img, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh_val = filters.threshold_otsu(blurred)
        binary = blurred > thresh_val
        
        kernel = np.ones((5,5), np.uint8)
        mask = binary.astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        binary_mask = mask > 0
        filled_mask = morphology.remove_small_holes(binary_mask, area_threshold=1000)
        cleaned_mask = morphology.remove_small_objects(filled_mask, min_size=500)
        
        final_mask = cleaned_mask.astype(np.uint8) * 255
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        
        return 255 - final_mask

    @staticmethod
    def create_tfrecord_example(image, x_start, y_start):
        """Create a TFRecord example from image and coordinates"""
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        image_bytes = image_bytes.getvalue()
        
        feature = {
            "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
            "x_start": tf.train.Feature(int64_list=tf.train.Int64List(value=[x_start])),
            "y_start": tf.train.Feature(int64_list=tf.train.Int64List(value=[y_start])),
        }
        
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def extract_tiles_and_visualize(self, output_dir, image_name, tile_px=256, tile_um=128, MPP=0.5, im_threshold=0.5):
        """Extract tiles from background areas and create visualization"""
        width, height = self.slide.dimensions
        tile_size_px = int(tile_um / MPP)
        n_tiles_x = math.ceil(width / tile_size_px)
        n_tiles_y = math.ceil(height / tile_size_px)
        
        vis_width = 2000
        scale_factor = vis_width / width
        vis_height = int(height * scale_factor)
        
        vis_img = np.array(self.slide.get_thumbnail((vis_width, vis_height)))
        vis_mask = cv2.resize(self.tissue_mask, (vis_width, vis_height), interpolation=cv2.INTER_NEAREST)
        
        os.makedirs(output_dir, exist_ok=True)
        tfrecord_path = os.path.join(output_dir, image_name + ".tfrecord")
        
        valid_tile_coords = []
        tiles_extracted = 0
        
        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            for y in range(n_tiles_y):
                for x in range(n_tiles_x):
                    x_start, y_start = x * tile_size_px, y * tile_size_px
                    if x_start >= width or y_start >= height:
                        continue
                    
                    actual_tile_size_x = min(tile_size_px, width - x_start)
                    actual_tile_size_y = min(tile_size_px, height - y_start)
                    
                    mask_x = int(x_start * self.tissue_mask.shape[1] / width)
                    mask_y = int(y_start * self.tissue_mask.shape[0] / height)
                    mask_width = int(actual_tile_size_x * self.tissue_mask.shape[1] / width)
                    mask_height = int(actual_tile_size_y * self.tissue_mask.shape[0] / height)
                    mask_region = self.tissue_mask[mask_y:mask_y+mask_height, mask_x:mask_x+mask_width]
                    
                    if np.mean(mask_region > 0) > im_threshold:
                        tile = self.slide.read_region((x_start, y_start), 0, (actual_tile_size_x, actual_tile_size_y)).convert('RGB')
                        example = self.create_tfrecord_example(tile, x_start, y_start)
                        writer.write(example.SerializeToString())
                        
                        vis_x_start, vis_y_start = int(x_start * scale_factor), int(y_start * scale_factor)
                        vis_tile_size_x, vis_tile_size_y = int(actual_tile_size_x * scale_factor), int(actual_tile_size_y * scale_factor)
                        valid_tile_coords.append((vis_x_start, vis_y_start, vis_tile_size_x, vis_tile_size_y))
                        tiles_extracted += 1

        vis_with_tiles = vis_img.copy()
        for (x_start, y_start, tile_size_x, tile_size_y) in valid_tile_coords:
            cv2.rectangle(vis_with_tiles, (x_start, y_start), (x_start + tile_size_x, y_start + tile_size_y), (0, 255, 0), 2)
        
        cv2.imwrite(os.path.join(output_dir, image_name + ".jpg"), cv2.cvtColor(vis_with_tiles, cv2.COLOR_RGB2BGR))
        return tiles_extracted




SUPPORTED_EXTENSIONS = ['.tiff', '.svs', '.mrxs']

def find_slide_path(input_dir, class_type, image_name):
    for ext in SUPPORTED_EXTENSIONS:
        potential_path = os.path.join(input_dir, class_type, image_name + ext)
        if os.path.isfile(potential_path):
            return potential_path
    return None

def generate_patch(input_dir, output_dir):
    file_paths = get_folder_file_list(input_dir)

    for class_type, image_name in file_paths:
        slide_path = find_slide_path(input_dir, class_type, image_name)
        if slide_path is None:
            print(f"Slide not found for {image_name} in {class_type}. Skipping.")
            continue

        try:
            slide = openslide.OpenSlide(slide_path)
        except Exception as e:
            print(f"Error opening slide {slide_path}: {e}")
            continue

        processor = SlideProcessor(slide)
        num_tiles = processor.extract_tiles_and_visualize(
            output_dir=os.path.join(output_dir, class_type),
            image_name=image_name,
            MPP=get_mpp(slide)
        )
        print(f"Extracted {num_tiles} background tiles for {image_name}.")