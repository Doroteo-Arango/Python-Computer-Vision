###############################################################
# GOALS
# Setup image & template image
# Apply template matching
# Improve template matching algorithm for sample image
###############################################################


# SETUP DATA


# Relevant Libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from google.colab import drive, files
from skimage import io, color, feature, transform
from skimage.color import rgb2gray
from skimage.feature import match_template, peak_local_max

# Mount & setup paths to data
drive.mount('/content/gdrive', force_remount=True)
dir = '/content/gdrive/dir'
img_dir = os.path.join(dir, 'images')

# Load image & template to match
img = io.imread(os.path.join(img_dir, 'letters.jpg'))
template1 = io.imread(os.path.join(img_dir, 'template_T.jpg'))
template2 = io.imread(os.path.join(img_dir, 'template_L.jpg'))

# Convert both images to grayscale
img_gs = color.rgb2gray(img)
template1_gs = color.rgb2gray(template1)
template2_gs = color.rgb2gray(template2)


###############################################################


# TEMPLATE MATCHING

# Apply template matching
result1 = match_template(img_gs, template1_gs)
result1 = match_template(img_gs, template2_gs)


###############################################################


# DISPLAY RESULTS

fig, ax = plt.subplots(1, 2, figsize = (15, 15))

ax[0].imshow(img_gs, cmap='gray')
ax[1].imshow(img, cmap='gray')

patch_width1, patch_height1 = template1_gs.shape 
patch_width2, patch_height2 = template2_gs.shape 

ax[0].set_title('Image', fontsize = 15)
ax[1].set_title('Template Matched', fontsize = 15)

###############################################################


# TEMPLATE IDENTIFICATION

# For each template matched, identify with rectangles
# Matching for template1
for x, y in peak_local_max(result1, threshold_abs = 0.8):
     rect = plt.Rectangle((y, x), patch_height1, patch_width1, color = 'r', fc = 'none')
     ax[1].add_patch(rect)

# Matching for template2
for x, y in peak_local_max(result2, threshold_abs = 0.8):
     rect = plt.Rectangle((y, x), patch_height2, patch_width2, color = 'b', fc = 'none')
     ax[1].add_patch(rect)

  
###############################################################


# ROTATE IMAGES FOR A SERIES OF ANGLES & APPLY TEMPLATE MATCHING

# Chosen angles are between -30° to +30°, in steps of 5°
# range(start, stop, step)
angles = range(-30, 31, 10)
for angle in angles:
  # 'resize = True' ensures full image is visible in display
  img_rotated = transform.rotate(img_gs, angle, resize = True)

  # Apply template matching
  result1 = match_template(img_rotated, template1_gs)
  result2 = match_template(img_rotated, template2_gs)

  # Create a figure for each iteration
  plt.figure(figsize = (8, 6))
  # Print rotated image onto the figure
  io.imshow(img_rotated, cmap = 'gray')
  plt.title(f'Rotated Image (Angle: {angle} degrees)')
  plt.axis('off')

  # Add rectangles for template matching identification
  patch_width1, patch_height1 = template1_gs.shape 
  patch_width2, patch_height2 = template2_gs.shape

  # Matching for T
  for x, y in peak_local_max(result1, min_distance=1, threshold_abs = 0.8):
    rect = plt.Rectangle((y, x), patch_height1, patch_width1, color = 'r', fc = 'none')
    # 'plt.gca()' is a MatPlotlib function 'Get Current Axes' that returns the axes of the current iterated figure
    plt.gca().add_patch(rect)
  
   # Matching for L
  for x, y in peak_local_max(result2, min_distance=1, threshold_abs = 0.8):
    rect = plt.Rectangle((y, x), patch_height2, patch_width2, color = 'b', fc = 'none')
    plt.gca().add_patch(rect)

plt.show()
