import os
import cv2
import time
import numpy as np

def find_overlap(image_path1, image_path2):
  """
  Finds the overlap area and percentage between two images.

  Args:
      image_path1 (str): Path to the first image file.
      image_path2 (str): Path to the second image file.

  Returns:
      tuple: A tuple containing the overlap area (pixels) and overlap percentage (float).
  """

  image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
  image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

  if image1 is None or image2 is None:
    print(f"Error: Could not load image files - {image_path1}, {image_path2}")
    return None, None

  correlation_output = cv2.filter2D(image1, cv2.CV_32F, image2, borderType=cv2.BORDER_CONSTANT)
  correlation_output = cv2.normalize(correlation_output, None, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX)
  overlap_mask = correlation_output > 0.95

  overlap_area = np.sum(overlap_mask)
  total_area1 = image1.shape[0] * image1.shape[1]

  overlap_percentage = 1 - (overlap_area / total_area1) * 100

  return overlap_area, overlap_percentage

def compare_all_images(image_folder):
    start = time.time()
    image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) 
                 if os.path.isfile(os.path.join(image_folder, filename))]

    for i in range(len(image_paths)):
        for j in range(i + 1, len(image_paths)):
            image_path1 = image_paths[i]
            image_path2 = image_paths[j]
            
            overlap_area, overlap_percentage = find_overlap(image_path1, image_path2)
            
            if overlap_area is not None:
                if overlap_percentage > 0.95:
                    print(f"Image 1: {image_path1}")
                    print(f"Image 2: {image_path2}")
                    print(f"Overlap Area: {overlap_area} pixels")
                    print(f"Overlap Percentage: {overlap_percentage:.2f}")

    end = time.time()
    print(f"Overlap search spent: {round((end - start), 2)} s")

image_folder = "C:/Users/flori/Downloads/image"  
compare_all_images(image_folder)
