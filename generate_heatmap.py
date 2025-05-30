import numpy as np
import math
import torch 
import openslide 
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches 
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas 
import cv2  
import gc 
import os
import openslide
from utility import get_mpp



# def generate_heatmap(npz_file, slide_path, model, output_path, class_names, tile_px=256, tile_um=128):
#     """
#     Generate a heatmap based on the trained model's attention scores.

#     Args:
#         npz_file (str): Path to the .npz file containing features and tile coordinates.
#         slide_path (str): Path to the whole-slide image (WSI).
#         model (torch.nn.Module): Trained MIL model.
#         output_path (str): Path to save the generated heatmap.
#         tile_px (int): Tile size in pixels.
#     """
#     # Load the .npz file
#     data = np.load(npz_file)
#     features = data['array1']  # Extracted features
#     tile_coords = data['array2']  # Tile coordinates (x, y, width, height)

#     # Load the slide
#     slide = openslide.OpenSlide(slide_path)
#     slide_width, slide_height = slide.dimensions
#     MPP=get_mpp(slide)
#     tile_size_px = int(tile_um / MPP)
#     # Calculate number of tiles in each dimension
#     n_tiles_x = math.ceil(slide_width / tile_size_px)
#     n_tiles_y = math.ceil(slide_height / tile_size_px)

#     extent = [0, n_tiles_x*tile_size_px, n_tiles_y*tile_size_px, 0]

#     # Convert features to a PyTorch tensor
#     features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension


#     # Pass features through the model to get attention scores
#     model.eval()
#     with torch.no_grad():
#         logits, attention_scores = model(features_tensor, torch.tensor([features_tensor.shape[1]]), return_attention=True)
#     predicted_classes = torch.argmax(logits, dim=1)  # Class indices (0,1,2,3)

#     # Normalize attention scores
#     attention_scores = attention_scores.squeeze().numpy()  # Remove batch dimension
#     attention_scores = (attention_scores - attention_scores.min()) / (attention_scores.max() - attention_scores.min())
#     #print(attention_scores)

#     # Overlay the heatmap on the slide thumbnail
#     thumbnail = slide.get_thumbnail((slide_width, slide_height))
#     thumbnail = np.array(thumbnail)
#     wsi_with_borders = thumbnail.copy()

#     colors = [plt.cm.get_cmap("tab10")(i)[:3] for i in range(len(class_names))]  # Use 'tab10', 'tab20', or 'hsv'
#     colors_255 = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]

#     legend_patches=[]
#     for class_type in range(len(class_names)):
#         # Create a blank canvas for the heatmap
#         masked_grid = np.zeros((n_tiles_y, n_tiles_x))
#         # Map attention scores to the heatmap
#         for i, (x, y) in enumerate(tile_coords):
#             x_start, y_start = int(x/tile_size_px), int(y/tile_size_px)
#             #print(x_start, y_start)
#             max_index = np.argmax(attention_scores[i])
#             if max_index==class_type:
#                 masked_grid[y_start, x_start] = attention_scores[i][class_type]
    
#         #print(masked_grid)
#         #print(masked_grid.shape)
    
        
#         original_height, original_width = masked_grid.shape[:2] #[extent[1],extent[2]]
#         fig, ax = plt.subplots(figsize=(original_width / 100, original_height / 100), dpi=5000)

#         # Remove extra padding
#         ax.set_position([0, 0, 1, 1])
#         fig.patch.set_visible(False)
#         ax.set_frame_on(False)
        
#         #ax.imshow(thumbnail, zorder=1, extent=extent)
        
#         # Display the image with specific parameters
#         im = ax.imshow(
#             masked_grid,
#             zorder=10,              # Draw this on top of other plot elements
#             alpha=0.6,              # Set transparency
#             #extent=extent,          # Map the array onto the specified bounding box
#             interpolation='bicubic',  # Define how the image is interpolated [[3]]
#             cmap='hot'          # Pick a color map for your data [[5]]
#         )
        
#         # Add optional decorations
#         plt.axis('off')
#         plt.tight_layout()
#         #plt.show()
#         plt.close('all')
#         gc.collect()
        
#         # Convert the plot to a NumPy array
#         fig = ax.figure  # Get the parent figure
#         canvas = FigureCanvas(fig)
#         canvas.draw()
        
#         # Convert the rendered figure to a NumPy array
#         image_array = np.array(canvas.buffer_rgba())  # Shape: (height, width, 4) in RGBA format
#         gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)


#         gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
#         gray_normalized = gray.astype(float)/255.0
#         threshold_value = 0.1
#         mask = (gray_normalized > threshold_value).astype(np.uint8)
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
#         heatmap_h, heatmap_w = image_array.shape[:2]
#         wsi_h, wsi_w = thumbnail.shape[:2]
#         # Calculate scaling factors
#         scale_x = wsi_w / heatmap_w  # Horizontal scaling factor [[6]]
#         scale_y = wsi_h / heatmap_h  # Vertical scaling factor [[6]]
        
#         # Scale contours to WSI dimensions
#         scaled_contours = [contour * np.array([scale_x, scale_y]) for contour in contours]
#         scaled_contours = [contour.astype(np.int32) for contour in scaled_contours]  # Convert to integer coordinates [[9]]
        
#         # Create WSI copy and draw scaled contours
#         cv2.drawContours(wsi_with_borders, scaled_contours, -1, colors_255[class_type], 2)
#         legend_patches.append(mpatches.Patch(color=colors[class_type], label=class_names[class_type]))


#     # print(wsi_with_borders.shape)
#     plt.imshow(wsi_with_borders, extent=extent)
#     plt.legend(handles=legend_patches, loc='upper right', prop={'size': 3})
#     plt.axis('off')  # Hide axes
#     plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300, format='png')
#     plt.close('all')
#     print(output_path, "Predict Class : "+class_names[predicted_classes.item()])
#     gc.collect()




def generate_heatmap(npz_file, slide_path, model, output_path, class_names, tile_px=256, tile_um=128):
    """
    Generate a heatmap based on the trained model's attention scores.

    Args:
        npz_file (str): Path to the .npz file containing features and tile coordinates.
        slide_path (str): Path to the whole-slide image (WSI).
        model (torch.nn.Module): Trained MIL model.
        output_path (str): Path to save the generated heatmap.
        tile_px (int): Tile size in pixels.
    """
    # Load the .npz file
    data = np.load(npz_file)
    features = data['array1']  # Extracted features
    tile_coords = data['array2']  # Tile coordinates (x, y, width, height)

    # Load the slide
    slide = openslide.OpenSlide(slide_path)
    slide_width, slide_height = slide.dimensions
    MPP=get_mpp(slide)
    tile_size_px = int(tile_um / MPP)
    # Calculate number of tiles in each dimension
    n_tiles_x = math.ceil(slide_width / tile_size_px)
    n_tiles_y = math.ceil(slide_height / tile_size_px)

    extent = [0, (n_tiles_x*tile_size_px), (n_tiles_y*tile_size_px), 0]

    # Convert features to a PyTorch tensor
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    # Pass features through the model to get attention scores
    model.eval()
    with torch.no_grad():
        logits, attention_scores = model(features_tensor, torch.tensor([features_tensor.shape[1]]), return_attention=True)
    predicted_classes = torch.argmax(logits, dim=1)  # Class indices (0,1,2,3)


    # Normalize attention scores
    attention_scores = attention_scores.squeeze().numpy()  # Remove batch dimension
    attention_scores = (attention_scores - attention_scores.min()) / (attention_scores.max() - attention_scores.min())

    fig, axes = plt.subplots(1, len(class_names), figsize=(25*len(class_names), 25))
    # Flatten axes in case of a single row or column
    axes = axes.flatten()

    for class_type in range(len(class_names)):
        # Create a blank canvas for the heatmap
        masked_grid = np.zeros((n_tiles_y, n_tiles_x))
        # Map attention scores to the heatmap
        for i, (x, y) in enumerate(tile_coords):
            x_start, y_start = int(x/tile_size_px), int(y/tile_size_px)
            #print(x_start, y_start)
            max_index = np.argmax(attention_scores[i])
            if max_index==class_type:
                masked_grid[y_start, x_start] = 1.0 #attention_scores[i][class_type]
    
        #print(masked_grid)
        #print(masked_grid.shape)
    
        # Overlay the heatmap on the slide thumbnail
        thumbnail = slide.get_thumbnail((slide_width, slide_height))
        thumbnail = np.array(thumbnail)
    
        
        axes[class_type].imshow(thumbnail, zorder=1, extent=extent)
        
        # Display the image with specific parameters
        im = axes[class_type].imshow(
            masked_grid,
            zorder=10,              # Draw this on top of other plot elements
            alpha=0.6,              # Set transparency
            extent=extent,          # Map the array onto the specified bounding box
            interpolation='bicubic',  # Define how the image is interpolated [[3]]
            cmap='hot'          # Pick a color map for your data [[5]]
        )

        # Set title for each subplot
        axes[class_type].set_title(class_names[class_type], fontsize=14)
        axes[class_type].axis('off')  # Remove axes
        
    # Add optional decorations
    plt.tight_layout()
    #plt.show()

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close('all')
    print(output_path, "Predict Class : "+class_names[predicted_classes.item()])