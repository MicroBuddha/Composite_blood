#!/usr/bin/env python
# coding: utf-8

# # Red Blood Cell (RBC) Segmentation from Blood Smear Images
# 
# ## Overview
# 
# This notebook implements an automated pipeline for segmenting Red Blood Cells (RBCs) from microscopic blood smear images. The goal is to extract individual RBCs with transparent backgrounds from mixed blood cell images containing various cell types including White Blood Cells (WBCs), platelets, and other blood components.
# 
# ## Dataset Structure
# 
# The dataset is organized into multiple classes representing different types of blood cells:
# - **basophil** - A type of white blood cell
# - **eosinophil** - A type of white blood cell  
# - **erythroblast** - Immature red blood cells
# - **ig** - Immature granulocytes
# - **lymphocyte** - A type of white blood cell
# - **monocyte** - A type of white blood cell
# - **neutrophil** - A type of white blood cell
# - **platelet** - Blood clotting cells
# 
# Each class folder contains blood smear images (360×363 pixels) with approximately 6-12 complete RBCs per image.
# 
# ## Methodology
# 
# ### 1. RBC Detection Pipeline
# The segmentation process involves several key steps:
# 
# 1. **Image Preprocessing**
#    - Convert to grayscale for contour detection
#    - Apply binary thresholding to identify cell boundaries
# 
# 2. **Contour Detection**
#    - Find all potential cell contours using OpenCV
#    - Filter contours based on size and shape characteristics
# 
# 3. **White Blood Cell Exclusion**
#    - Use HSV color space to identify purple-stained WBCs
#    - Exclude regions with significant purple coloration
# 
# 4. **Quality Filtering**
#    - **Size filtering**: Remove artifacts and incomplete cells
#    - **Border filtering**: Exclude cells touching image edges
#    - **Shape analysis**: Filter based on circularity, aspect ratio, and convexity
#    - **Completeness check**: Ensure only complete, well-formed RBCs are kept
# 
# 5. **Individual RBC Extraction**
#    - Extract each valid RBC with transparent background
#    - Save as individual PNG files with alpha channel
# 
# ### 2. Data Consolidation
# After processing all class folders, individual RBCs from different classes are consolidated into a single `rbc` folder for downstream analysis or machine learning applications.
# 
# ## Expected Outcomes
# 
# - **Individual RBC images** with transparent backgrounds
# - **Quality-filtered dataset** containing only complete, well-segmented RBCs
# - **Consolidated RBC collection** from multiple blood cell classes
# - **Detailed processing statistics** for quality assessment
# 
# ## Applications
# 
# This segmented RBC dataset can be used for:
# - Training machine learning models for RBC analysis
# - Morphological studies of red blood cell characteristics
# - Automated blood cell counting systems
# - Research in hematology and medical diagnostics
# 
# ---
# 
# *Note: This segmentation approach focuses on extracting complete RBCs while maintaining high quality standards through multi-stage filtering.*

# ## Basophil

# In[31]:


import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

def segment_rbcs_only(image_path):
    """
    Segment only RBCs by detecting all cells and excluding WBCs (purple cells)
    """
    # Read the image
    img = cv2.imread(image_path)
    # Convert to RGB (from BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Find all cells using contour detection
    # Apply binary thresholding to catch cell edges
    _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    
    # Find all contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Step 2: Identify WBCs (purple cells) to exclude them
    # Convert to HSV for color-based detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define range for purple color in HSV (WBCs)
    lower_purple = np.array([120, 50, 50])
    upper_purple = np.array([160, 255, 255])
    
    # Create mask for purple regions (WBCs)
    purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
    
    # Clean up the purple mask
    kernel = np.ones((5, 5), np.uint8)
    purple_mask_cleaned = cv2.morphologyEx(purple_mask, cv2.MORPH_CLOSE, kernel)
    
    # Step 3: Filter contours to keep only RBCs (non-purple cells)
    rbc_contours = []
    
    for contour in contours:
        # Create a mask for this contour
        contour_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # Check if this contour overlaps significantly with purple regions
        overlap = cv2.bitwise_and(contour_mask, purple_mask_cleaned)
        overlap_ratio = np.sum(overlap > 0) / np.sum(contour_mask > 0)
        
        # If overlap is less than 30%, consider it an RBC
        if overlap_ratio < 0.3:
            # Additional size filtering to remove very small artifacts
            area = cv2.contourArea(contour)
            if area > 100:  # Adjust this threshold based on your image resolution
                rbc_contours.append(contour)
    
    # Step 4: Create visualization
    # Create a colored segmentation mask for RBCs only
    rbc_segmentation = np.zeros((*img_rgb.shape[:2], 3), dtype=np.uint8)
    cv2.drawContours(rbc_segmentation, rbc_contours, -1, (0, 255, 0), 2)
    
    # Create an overlay visualization
    rbc_overlay = img_rgb.copy()
    cv2.drawContours(rbc_overlay, rbc_contours, -1, (0, 255, 0), 2)
    
    return img_rgb, rbc_segmentation, rbc_overlay, len(rbc_contours)

def process_random_image(folder_path):
    """
    Process a random image from the specified folder
    """
    # Get all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    
    if not image_files:
        return None, "No image files found in the specified folder."
    
    # Select a random image
    random_image = random.choice(image_files)
    image_path = os.path.join(folder_path, random_image)
    
    # Process the image
    original, segmentation, overlay, rbc_count = segment_rbcs_only(image_path)
    
    return (original, segmentation, overlay, rbc_count), random_image

# Main execution
if __name__ == "__main__":
    image_folder = "/Users/afnanag/projects/DH307/yolov11/segment_rbc/data/basophil/images"
    
    try:
        (original, segmentation, overlay, rbc_count), image_name = process_random_image(image_folder)
        
        # Display the results
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.title(f"Original: {image_name}")
        plt.imshow(original)
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title(f"RBC Segmentation ({rbc_count} RBCs)")
        plt.imshow(segmentation)
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title("RBC Overlay")
        plt.imshow(overlay)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Successfully processed image: {image_name}")
        print(f"Number of RBCs detected: {rbc_count}")
        
    except Exception as e:
        print(f"Error: {str(e)}")


# In[33]:


import os
import cv2
import numpy as np

def segment_rbcs(image_path):
    """
    Segment all RBCs in an image and save each one individually with transparent background
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return None
    
    # Convert to grayscale for contour detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Find all cells using contour detection
    # Apply binary thresholding to catch cell edges
    _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    
    # Find all contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Step 2: Identify WBCs (purple cells) to exclude them
    # Convert to HSV for color-based detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for purple color in HSV (WBCs)
    lower_purple = np.array([120, 50, 50])
    upper_purple = np.array([160, 255, 255])
    
    # Create mask for purple regions (WBCs)
    purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
    
    # Clean up the purple mask
    kernel = np.ones((5, 5), np.uint8)
    purple_mask_cleaned = cv2.morphologyEx(purple_mask, cv2.MORPH_CLOSE, kernel)
    
    # Step 3: Filter contours to keep only complete RBCs (non-purple cells)
    rbc_contours = []
    image_height, image_width = gray.shape
    
    for contour in contours:
        # Create a mask for this contour
        contour_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # Check if this contour overlaps significantly with purple regions
        overlap = cv2.bitwise_and(contour_mask, purple_mask_cleaned)
        overlap_ratio = np.sum(overlap > 0) / np.sum(contour_mask > 0)
        
        # If overlap is less than 30%, consider it potentially an RBC
        if overlap_ratio < 0.3:
            # Filter 1: Size filtering to remove very small artifacts
            area = cv2.contourArea(contour)
            if area < 200 or area > 5000:  # Adjust based on expected RBC size
                continue
            
            # Filter 2: Check if contour touches image borders (incomplete RBCs)
            x, y, w, h = cv2.boundingRect(contour)
            border_margin = 15  # pixels from edge
            
            if (x <= border_margin or y <= border_margin or 
                x + w >= image_width - border_margin or 
                y + h >= image_height - border_margin):
                continue  # Skip RBCs too close to borders
            
            # Filter 3: Circularity check - RBCs should be roughly circular
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.5:  # Should be close to 1 for perfect circle
                continue
            
            # Filter 4: Aspect ratio check - RBCs should be roughly round
            aspect_ratio = float(w) / h
            if aspect_ratio < 0.6 or aspect_ratio > 1.7:  # Allow some elongation
                continue
            
            # Filter 5: Convexity check - complete RBCs should be fairly convex
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            solidity = area / hull_area
            if solidity < 0.7:  # Should be fairly solid/convex
                continue
            
            rbc_contours.append(contour)
    
    if not rbc_contours:
        print(f"No RBCs found in {os.path.basename(image_path)}")
        return None
    
    # Step 4: Create individual segmented RBCs with transparent backgrounds
    segmented_rbcs = []
    
    for i, contour in enumerate(rbc_contours):
        # Create an empty mask and fill the contour
        rbc_mask = np.zeros_like(gray)
        cv2.drawContours(rbc_mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # Get the bounding rectangle of the contour to crop the image closely
        x, y, w, h = cv2.boundingRect(contour)
        
        # Add padding to the bounding box
        padding = 5  # Reduced padding since we're filtering edge cases
        x_pad = max(0, x - padding)
        y_pad = max(0, y - padding)
        w_pad = min(image.shape[1] - x_pad, w + 2*padding)
        h_pad = min(image.shape[0] - y_pad, h + 2*padding)
        
        # Create a transparent 4-channel image (RGBA) with the size of the padded bounding rectangle
        segmented_cropped = np.zeros((h_pad, w_pad, 4), dtype=np.uint8)
        
        # Create a mask for the cropped region
        cropped_mask = rbc_mask[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
        
        # Copy RGB values from original image to the cropped region where mask is active
        cropped_image = image[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
        segmented_cropped[:, :, 0:3][cropped_mask == 255] = cropped_image[cropped_mask == 255]
        
        # Set alpha channel - transparent (0) where inactive, opaque (255) where active
        segmented_cropped[:, :, 3] = cropped_mask
        
        segmented_rbcs.append(segmented_cropped)
    
    return segmented_rbcs, os.path.basename(image_path)

# Main execution
if __name__ == "__main__":
    # Folder containing images
    image_folder = "/Users/afnanag/projects/DH307/yolov11/segment_rbc/data/basophil/images"
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(image_folder), "segmented_rbcs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    
    # Count for progress tracking
    total_images = len(image_files)
    successful_segments = 0
    failed_segments = 0
    total_rbcs_saved = 0
    
    print(f"Found {total_images} images to process...")
    
    # Process each image
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_folder, image_file)
        print(f"[{i+1}/{total_images}] Processing {image_file}...")
        
        # Segment the image
        result = segment_rbcs(image_path)
        
        if result is not None:
            segmented_rbcs, original_filename = result
            
            # Save each RBC individually
            base_name = os.path.splitext(original_filename)[0]
            rbcs_saved_for_this_image = 0
            
            for j, rbc_img in enumerate(segmented_rbcs):
                # Create filename for individual RBC
                output_filename = f"rbc_{base_name}_{j+1:03d}.png"
                output_path = os.path.join(output_dir, output_filename)
                
                # Save to PNG to preserve transparency
                cv2.imwrite(output_path, rbc_img)
                rbcs_saved_for_this_image += 1
                total_rbcs_saved += 1
            
            print(f" ✓ Saved {rbcs_saved_for_this_image} RBCs from {original_filename}")
            successful_segments += 1
        else:
            print(f" ✗ Failed to segment RBCs in {image_file}")
            failed_segments += 1
    
    print(f"\nRBC Segmentation complete!")
    print(f"Successfully processed: {successful_segments}/{total_images} images")
    print(f"Failed to process: {failed_segments}/{total_images} images")
    print(f"Total RBCs saved: {total_rbcs_saved}")
    print(f"Segmented RBCs saved to: {output_dir}")


# ## Eosinophil, Erythroblast, Lymphocyte

# In[48]:


import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

def segment_rbcs_only(image_path):
    """
    Segment only RBCs by detecting all cells and excluding WBCs (purple cells)
    """
    # Read the image
    img = cv2.imread(image_path)
    # Convert to RGB (from BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Find all cells using contour detection
    # Apply binary thresholding to catch cell edges
    _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    
    # Find all contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Step 2: Identify WBCs (purple cells) to exclude them
    # Convert to HSV for color-based detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define range for purple color in HSV (WBCs)
    lower_purple = np.array([120, 50, 50])
    upper_purple = np.array([160, 255, 255])
    
    # Create mask for purple regions (WBCs)
    purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
    
    # Clean up the purple mask
    kernel = np.ones((5, 5), np.uint8)
    purple_mask_cleaned = cv2.morphologyEx(purple_mask, cv2.MORPH_CLOSE, kernel)
    
    # Step 3: Filter contours to keep only RBCs (non-purple cells)
    rbc_contours = []
    
    for contour in contours:
        # Create a mask for this contour
        contour_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # Check if this contour overlaps significantly with purple regions
        overlap = cv2.bitwise_and(contour_mask, purple_mask_cleaned)
        overlap_ratio = np.sum(overlap > 0) / np.sum(contour_mask > 0)
        
        # If overlap is less than 30%, consider it an RBC
        if overlap_ratio < 0.3:
            # Additional size filtering to remove very small artifacts
            area = cv2.contourArea(contour)
            if area > 100:  # Adjust this threshold based on your image resolution
                rbc_contours.append(contour)
    
    # Step 4: Create visualization
    # Create a colored segmentation mask for RBCs only
    rbc_segmentation = np.zeros((*img_rgb.shape[:2], 3), dtype=np.uint8)
    cv2.drawContours(rbc_segmentation, rbc_contours, -1, (0, 255, 0), 2)
    
    # Create an overlay visualization
    rbc_overlay = img_rgb.copy()
    cv2.drawContours(rbc_overlay, rbc_contours, -1, (0, 255, 0), 2)
    
    return img_rgb, rbc_segmentation, rbc_overlay, len(rbc_contours)

def process_random_image(folder_path):
    """
    Process a random image from the specified folder
    """
    # Get all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    
    if not image_files:
        return None, "No image files found in the specified folder."
    
    # Select a random image
    random_image = random.choice(image_files)
    image_path = os.path.join(folder_path, random_image)
    
    # Process the image
    original, segmentation, overlay, rbc_count = segment_rbcs_only(image_path)
    
    return (original, segmentation, overlay, rbc_count), random_image

# Main execution
if __name__ == "__main__":
    image_folder = "/Users/afnanag/projects/DH307/yolov11/segment_rbc/data/eosinophil/images"
    
    try:
        (original, segmentation, overlay, rbc_count), image_name = process_random_image(image_folder)
        
        # Display the results
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.title(f"Original: {image_name}")
        plt.imshow(original)
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title(f"RBC Segmentation ({rbc_count} RBCs)")
        plt.imshow(segmentation)
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title("RBC Overlay")
        plt.imshow(overlay)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Successfully processed image: {image_name}")
        print(f"Number of RBCs detected: {rbc_count}")
        
    except Exception as e:
        print(f"Error: {str(e)}")


# In[50]:


import os
import cv2
import numpy as np

def segment_rbcs(image_path):
    """
    Segment all RBCs in an image and save each one individually with transparent background
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return None
    
    # Convert to grayscale for contour detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Find all cells using contour detection
    # Apply binary thresholding to catch cell edges
    _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    
    # Find all contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Step 2: Identify WBCs (purple cells) to exclude them
    # Convert to HSV for color-based detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for purple color in HSV (WBCs)
    lower_purple = np.array([120, 50, 50])
    upper_purple = np.array([160, 255, 255])
    
    # Create mask for purple regions (WBCs)
    purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
    
    # Clean up the purple mask
    kernel = np.ones((5, 5), np.uint8)
    purple_mask_cleaned = cv2.morphologyEx(purple_mask, cv2.MORPH_CLOSE, kernel)
    
    # Step 3: Filter contours to keep only complete RBCs (non-purple cells)
    rbc_contours = []
    image_height, image_width = gray.shape
    
    for contour in contours:
        # Create a mask for this contour
        contour_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # Check if this contour overlaps significantly with purple regions
        overlap = cv2.bitwise_and(contour_mask, purple_mask_cleaned)
        overlap_ratio = np.sum(overlap > 0) / np.sum(contour_mask > 0)
        
        # If overlap is less than 30%, consider it potentially an RBC
        if overlap_ratio < 0.3:
            # Filter 1: Size filtering to remove very small artifacts
            area = cv2.contourArea(contour)
            if area < 200 or area > 5000:  # Adjust based on expected RBC size
                continue
            
            # Filter 2: Check if contour touches image borders (incomplete RBCs)
            x, y, w, h = cv2.boundingRect(contour)
            border_margin = 15  # pixels from edge
            
            if (x <= border_margin or y <= border_margin or 
                x + w >= image_width - border_margin or 
                y + h >= image_height - border_margin):
                continue  # Skip RBCs too close to borders
            
            # Filter 3: Circularity check - RBCs should be roughly circular
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.5:  # Should be close to 1 for perfect circle
                continue
            
            # Filter 4: Aspect ratio check - RBCs should be roughly round
            aspect_ratio = float(w) / h
            if aspect_ratio < 0.6 or aspect_ratio > 1.7:  # Allow some elongation
                continue
            
            # Filter 5: Convexity check - complete RBCs should be fairly convex
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            solidity = area / hull_area
            if solidity < 0.7:  # Should be fairly solid/convex
                continue
            
            rbc_contours.append(contour)
    
    if not rbc_contours:
        print(f"No RBCs found in {os.path.basename(image_path)}")
        return None
    
    # Step 4: Create individual segmented RBCs with transparent backgrounds
    segmented_rbcs = []
    
    for i, contour in enumerate(rbc_contours):
        # Create an empty mask and fill the contour
        rbc_mask = np.zeros_like(gray)
        cv2.drawContours(rbc_mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # Get the bounding rectangle of the contour to crop the image closely
        x, y, w, h = cv2.boundingRect(contour)
        
        # Add padding to the bounding box
        padding = 5  # Reduced padding since we're filtering edge cases
        x_pad = max(0, x - padding)
        y_pad = max(0, y - padding)
        w_pad = min(image.shape[1] - x_pad, w + 2*padding)
        h_pad = min(image.shape[0] - y_pad, h + 2*padding)
        
        # Create a transparent 4-channel image (RGBA) with the size of the padded bounding rectangle
        segmented_cropped = np.zeros((h_pad, w_pad, 4), dtype=np.uint8)
        
        # Create a mask for the cropped region
        cropped_mask = rbc_mask[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
        
        # Copy RGB values from original image to the cropped region where mask is active
        cropped_image = image[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
        segmented_cropped[:, :, 0:3][cropped_mask == 255] = cropped_image[cropped_mask == 255]
        
        # Set alpha channel - transparent (0) where inactive, opaque (255) where active
        segmented_cropped[:, :, 3] = cropped_mask
        
        segmented_rbcs.append(segmented_cropped)
    
    return segmented_rbcs, os.path.basename(image_path)

# Main execution
if __name__ == "__main__":
    # Folder containing images
    image_folder = "/Users/afnanag/projects/DH307/yolov11/segment_rbc/data/eosinophil/images"
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(image_folder), "segmented_rbcs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    
    # Count for progress tracking
    total_images = len(image_files)
    successful_segments = 0
    failed_segments = 0
    total_rbcs_saved = 0
    
    print(f"Found {total_images} images to process...")
    
    # Process each image
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_folder, image_file)
        print(f"[{i+1}/{total_images}] Processing {image_file}...")
        
        # Segment the image
        result = segment_rbcs(image_path)
        
        if result is not None:
            segmented_rbcs, original_filename = result
            
            # Save each RBC individually
            base_name = os.path.splitext(original_filename)[0]
            rbcs_saved_for_this_image = 0
            
            for j, rbc_img in enumerate(segmented_rbcs):
                # Create filename for individual RBC
                output_filename = f"rbc_{base_name}_{j+1:03d}.png"
                output_path = os.path.join(output_dir, output_filename)
                
                # Save to PNG to preserve transparency
                cv2.imwrite(output_path, rbc_img)
                rbcs_saved_for_this_image += 1
                total_rbcs_saved += 1
            
            print(f" ✓ Saved {rbcs_saved_for_this_image} RBCs from {original_filename}")
            successful_segments += 1
        else:
            print(f" ✗ Failed to segment RBCs in {image_file}")
            failed_segments += 1
    
    print(f"\nRBC Segmentation complete!")
    print(f"Successfully processed: {successful_segments}/{total_images} images")
    print(f"Failed to process: {failed_segments}/{total_images} images")
    print(f"Total RBCs saved: {total_rbcs_saved}")
    print(f"Segmented RBCs saved to: {output_dir}")


# In[51]:


# Main execution
if __name__ == "__main__":
    # Folder containing images
    image_folder = "/Users/afnanag/projects/DH307/yolov11/segment_rbc/data/erythroblast/images"
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(image_folder), "segmented_rbcs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    
    # Count for progress tracking
    total_images = len(image_files)
    successful_segments = 0
    failed_segments = 0
    total_rbcs_saved = 0
    
    print(f"Found {total_images} images to process...")
    
    # Process each image
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_folder, image_file)
        print(f"[{i+1}/{total_images}] Processing {image_file}...")
        
        # Segment the image
        result = segment_rbcs(image_path)
        
        if result is not None:
            segmented_rbcs, original_filename = result
            
            # Save each RBC individually
            base_name = os.path.splitext(original_filename)[0]
            rbcs_saved_for_this_image = 0
            
            for j, rbc_img in enumerate(segmented_rbcs):
                # Create filename for individual RBC
                output_filename = f"rbc_{base_name}_{j+1:03d}.png"
                output_path = os.path.join(output_dir, output_filename)
                
                # Save to PNG to preserve transparency
                cv2.imwrite(output_path, rbc_img)
                rbcs_saved_for_this_image += 1
                total_rbcs_saved += 1
            
            print(f" ✓ Saved {rbcs_saved_for_this_image} RBCs from {original_filename}")
            successful_segments += 1
        else:
            print(f" ✗ Failed to segment RBCs in {image_file}")
            failed_segments += 1
    
    print(f"\nRBC Segmentation complete!")
    print(f"Successfully processed: {successful_segments}/{total_images} images")
    print(f"Failed to process: {failed_segments}/{total_images} images")
    print(f"Total RBCs saved: {total_rbcs_saved}")
    print(f"Segmented RBCs saved to: {output_dir}")


# In[52]:


# Main execution
if __name__ == "__main__":
    # Folder containing images
    image_folder = "/Users/afnanag/projects/DH307/yolov11/segment_rbc/data/lymphocyte/images"
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(image_folder), "segmented_rbcs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    
    # Count for progress tracking
    total_images = len(image_files)
    successful_segments = 0
    failed_segments = 0
    total_rbcs_saved = 0
    
    print(f"Found {total_images} images to process...")
    
    # Process each image
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_folder, image_file)
        print(f"[{i+1}/{total_images}] Processing {image_file}...")
        
        # Segment the image
        result = segment_rbcs(image_path)
        
        if result is not None:
            segmented_rbcs, original_filename = result
            
            # Save each RBC individually
            base_name = os.path.splitext(original_filename)[0]
            rbcs_saved_for_this_image = 0
            
            for j, rbc_img in enumerate(segmented_rbcs):
                # Create filename for individual RBC
                output_filename = f"rbc_{base_name}_{j+1:03d}.png"
                output_path = os.path.join(output_dir, output_filename)
                
                # Save to PNG to preserve transparency
                cv2.imwrite(output_path, rbc_img)
                rbcs_saved_for_this_image += 1
                total_rbcs_saved += 1
            
            print(f" ✓ Saved {rbcs_saved_for_this_image} RBCs from {original_filename}")
            successful_segments += 1
        else:
            print(f" ✗ Failed to segment RBCs in {image_file}")
            failed_segments += 1
    
    print(f"\nRBC Segmentation complete!")
    print(f"Successfully processed: {successful_segments}/{total_images} images")
    print(f"Failed to process: {failed_segments}/{total_images} images")
    print(f"Total RBCs saved: {total_rbcs_saved}")
    print(f"Segmented RBCs saved to: {output_dir}")


# ## IG

# In[53]:


# Main execution
if __name__ == "__main__":
    # Folder containing images
    image_folder = "/Users/afnanag/projects/DH307/yolov11/segment_rbc/data/ig/images"
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(image_folder), "segmented_rbcs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    
    # Count for progress tracking
    total_images = len(image_files)
    successful_segments = 0
    failed_segments = 0
    total_rbcs_saved = 0
    
    print(f"Found {total_images} images to process...")
    
    # Process each image
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_folder, image_file)
        print(f"[{i+1}/{total_images}] Processing {image_file}...")
        
        # Segment the image
        result = segment_rbcs(image_path)
        
        if result is not None:
            segmented_rbcs, original_filename = result
            
            # Save each RBC individually
            base_name = os.path.splitext(original_filename)[0]
            rbcs_saved_for_this_image = 0
            
            for j, rbc_img in enumerate(segmented_rbcs):
                # Create filename for individual RBC
                output_filename = f"rbc_{base_name}_{j+1:03d}.png"
                output_path = os.path.join(output_dir, output_filename)
                
                # Save to PNG to preserve transparency
                cv2.imwrite(output_path, rbc_img)
                rbcs_saved_for_this_image += 1
                total_rbcs_saved += 1
            
            print(f" ✓ Saved {rbcs_saved_for_this_image} RBCs from {original_filename}")
            successful_segments += 1
        else:
            print(f" ✗ Failed to segment RBCs in {image_file}")
            failed_segments += 1
    
    print(f"\nRBC Segmentation complete!")
    print(f"Successfully processed: {successful_segments}/{total_images} images")
    print(f"Failed to process: {failed_segments}/{total_images} images")
    print(f"Total RBCs saved: {total_rbcs_saved}")
    print(f"Segmented RBCs saved to: {output_dir}")


# ## Monocyte

# In[55]:


# Main execution
if __name__ == "__main__":
    # Folder containing images
    image_folder = "/Users/afnanag/projects/DH307/yolov11/segment_rbc/data/monocyte/images"
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(image_folder), "segmented_rbcs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    
    # Count for progress tracking
    total_images = len(image_files)
    successful_segments = 0
    failed_segments = 0
    total_rbcs_saved = 0
    
    print(f"Found {total_images} images to process...")
    
    # Process each image
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_folder, image_file)
        print(f"[{i+1}/{total_images}] Processing {image_file}...")
        
        # Segment the image
        result = segment_rbcs(image_path)
        
        if result is not None:
            segmented_rbcs, original_filename = result
            
            # Save each RBC individually
            base_name = os.path.splitext(original_filename)[0]
            rbcs_saved_for_this_image = 0
            
            for j, rbc_img in enumerate(segmented_rbcs):
                # Create filename for individual RBC
                output_filename = f"rbc_{base_name}_{j+1:03d}.png"
                output_path = os.path.join(output_dir, output_filename)
                
                # Save to PNG to preserve transparency
                cv2.imwrite(output_path, rbc_img)
                rbcs_saved_for_this_image += 1
                total_rbcs_saved += 1
            
            print(f" ✓ Saved {rbcs_saved_for_this_image} RBCs from {original_filename}")
            successful_segments += 1
        else:
            print(f" ✗ Failed to segment RBCs in {image_file}")
            failed_segments += 1
    
    print(f"\nRBC Segmentation complete!")
    print(f"Successfully processed: {successful_segments}/{total_images} images")
    print(f"Failed to process: {failed_segments}/{total_images} images")
    print(f"Total RBCs saved: {total_rbcs_saved}")
    print(f"Segmented RBCs saved to: {output_dir}")


# ## Neutrophil

# In[56]:


# Main execution
if __name__ == "__main__":
    # Folder containing images
    image_folder = "/Users/afnanag/projects/DH307/yolov11/segment_rbc/data/neutrophil/images"
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(image_folder), "segmented_rbcs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    
    # Count for progress tracking
    total_images = len(image_files)
    successful_segments = 0
    failed_segments = 0
    total_rbcs_saved = 0
    
    print(f"Found {total_images} images to process...")
    
    # Process each image
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_folder, image_file)
        print(f"[{i+1}/{total_images}] Processing {image_file}...")
        
        # Segment the image
        result = segment_rbcs(image_path)
        
        if result is not None:
            segmented_rbcs, original_filename = result
            
            # Save each RBC individually
            base_name = os.path.splitext(original_filename)[0]
            rbcs_saved_for_this_image = 0
            
            for j, rbc_img in enumerate(segmented_rbcs):
                # Create filename for individual RBC
                output_filename = f"rbc_{base_name}_{j+1:03d}.png"
                output_path = os.path.join(output_dir, output_filename)
                
                # Save to PNG to preserve transparency
                cv2.imwrite(output_path, rbc_img)
                rbcs_saved_for_this_image += 1
                total_rbcs_saved += 1
            
            print(f" ✓ Saved {rbcs_saved_for_this_image} RBCs from {original_filename}")
            successful_segments += 1
        else:
            print(f" ✗ Failed to segment RBCs in {image_file}")
            failed_segments += 1
    
    print(f"\nRBC Segmentation complete!")
    print(f"Successfully processed: {successful_segments}/{total_images} images")
    print(f"Failed to process: {failed_segments}/{total_images} images")
    print(f"Total RBCs saved: {total_rbcs_saved}")
    print(f"Segmented RBCs saved to: {output_dir}")


# ## Platelet

# In[57]:


# Main execution
if __name__ == "__main__":
    # Folder containing images
    image_folder = "/Users/afnanag/projects/DH307/yolov11/segment_rbc/data/platelet/images"
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(image_folder), "segmented_rbcs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    
    # Count for progress tracking
    total_images = len(image_files)
    successful_segments = 0
    failed_segments = 0
    total_rbcs_saved = 0
    
    print(f"Found {total_images} images to process...")
    
    # Process each image
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_folder, image_file)
        print(f"[{i+1}/{total_images}] Processing {image_file}...")
        
        # Segment the image
        result = segment_rbcs(image_path)
        
        if result is not None:
            segmented_rbcs, original_filename = result
            
            # Save each RBC individually
            base_name = os.path.splitext(original_filename)[0]
            rbcs_saved_for_this_image = 0
            
            for j, rbc_img in enumerate(segmented_rbcs):
                # Create filename for individual RBC
                output_filename = f"rbc_{base_name}_{j+1:03d}.png"
                output_path = os.path.join(output_dir, output_filename)
                
                # Save to PNG to preserve transparency
                cv2.imwrite(output_path, rbc_img)
                rbcs_saved_for_this_image += 1
                total_rbcs_saved += 1
            
            print(f" ✓ Saved {rbcs_saved_for_this_image} RBCs from {original_filename}")
            successful_segments += 1
        else:
            print(f" ✗ Failed to segment RBCs in {image_file}")
            failed_segments += 1
    
    print(f"\nRBC Segmentation complete!")
    print(f"Successfully processed: {successful_segments}/{total_images} images")
    print(f"Failed to process: {failed_segments}/{total_images} images")
    print(f"Total RBCs saved: {total_rbcs_saved}")
    print(f"Segmented RBCs saved to: {output_dir}")


# In[58]:


import os
import shutil

def copy_all_rbcs():
    """
    Copy all segmented RBCs from different class folders into a single 'rbc' folder
    """
    # Base path
    base_path = "/Users/afnanag/projects/DH307/yolov11/segment_rbc/data"
    
    # List of classes (excluding lymphoblast)
    classes = ['basophil', 'eosinophil', 'erythroblast', 'ig', 'lymphocyte', 'monocyte', 'neutrophil', 'platelet']
    
    # Create destination folder
    destination_folder = os.path.join(base_path, "rbc")
    os.makedirs(destination_folder, exist_ok=True)
    
    total_files_copied = 0
    files_per_class = {}
    
    print("Starting to copy RBC files...")
    print("-" * 50)
    
    for class_name in classes:
        source_folder = os.path.join(base_path, class_name, "segmented_rbcs")
        
        if not os.path.exists(source_folder):
            print(f"⚠️  Source folder not found: {source_folder}")
            files_per_class[class_name] = 0
            continue
        
        # Get all PNG files in the source folder
        rbc_files = [f for f in os.listdir(source_folder) if f.lower().endswith('.png')]
        
        if not rbc_files:
            print(f"⚠️  No RBC files found in: {class_name}")
            files_per_class[class_name] = 0
            continue
        
        # Copy each file
        files_copied_for_class = 0
        for rbc_file in rbc_files:
            source_path = os.path.join(source_folder, rbc_file)
            
            # Create new filename with class prefix to avoid naming conflicts
            new_filename = f"{class_name}_{rbc_file}"
            destination_path = os.path.join(destination_folder, new_filename)
            
            # Copy the file
            try:
                shutil.copy2(source_path, destination_path)
                files_copied_for_class += 1
                total_files_copied += 1
            except Exception as e:
                print(f"❌ Error copying {rbc_file} from {class_name}: {e}")
        
        files_per_class[class_name] = files_copied_for_class
        print(f"✅ {class_name}: {files_copied_for_class} files copied")
    
    print("-" * 50)
    print("Copy operation completed!")
    print(f"Total files copied: {total_files_copied}")
    print(f"Destination folder: {destination_folder}")
    
    print("\nSummary by class:")
    for class_name, count in files_per_class.items():
        print(f"  {class_name}: {count} RBCs")
    
    return destination_folder, total_files_copied

if __name__ == "__main__":
    destination, total = copy_all_rbcs()
    print(f"\nAll RBC files have been copied to: {destination}")
    print(f"Total RBC files: {total}")

