################################################################
# GOALS
# Import an image of a monkey using Google drive & Google Colab
# Apply Gaussian noise
# Experiment with different edge detection algorithms
# Compare results
###############################################################


# SETTING UP NOISY IMAGE FOR EXPERIMENTATION

# Relevant Libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive, files
from skimage import io, color, feature, util, transform, filters
from skimage.util import random_noise

# Mount & Setup Paths to Data
drive.mount('/content/gdrive', force_remount=True)
dir = '/content/gdrive/dir'
img_dir = os.path.join(dir, 'images')

# Load jpg image
img = io.imread(os.path.join(img_dir, 'monkey.jpg'))

# IPA techniques perform better with a resized, greyscale image
resized_img = transform.resize(img, (512, 512))
gs_img = color.rgb2gray(resized_img)

# Function to add random noise of various types (Gaussian) to a floating-point image
# Keep default value for variance of distribution
sigma = 0.01
img_noise = random_noise(gs_img, mode= 'gaussian', mean = 0.0, var = sigma)


###############################################################


# ROBERTS EDGE DETECTION ALGORITHM

img_roberts = color.rgb2gray(img_noise)


###############################################################


# SOBEL EDGE DETECTION ALGORITHM

img_sobel = filters.sobel(img_noise)


###############################################################


# CANNY EDGE DETECTION ALGORITHM

# Compute the Canny filter for two values of sigma
canny1 = feature.canny(img_noise)
canny2 = feature.canny(img_noise, sigma = 3)


###############################################################


# DISPLAY RESULTS


# Initialise plots
fig, ax = plt.subplots(ncols=4, figsize=(24, 9))

ax[0] = plt.subplot(1, 4, 1)
ax[1] = plt.subplot(1, 4, 2, sharex = ax[0], sharey = ax[0])
ax[2] = plt.subplot(1, 4, 3, sharex = ax[0], sharey = ax[0])
ax[3] = plt.subplot(1, 4, 4, sharex = ax[0], sharey = ax[0])

# Populate plots
ax[0].imshow(img_roberts, cmap=plt.cm.gray)
ax[0].set_title('Roberts Edge Detection')
ax[0].axis('on')

ax[1].imshow(img_sobel, cmap=plt.cm.gray)
ax[1].set_title('Sobel Edge Detection')
ax[1].axis('on')

ax[2].imshow(canny1, cmap=plt.cm.gray)
ax[2].set_title('Canny Edge Detection w/ sigma = 1')

ax[3].imshow(canny2, cmap=plt.cm.gray)
ax[3].set_title('Canny Edge Detection w/ sigma = 3')

fig.tight_layout()

# To save figure as jpg file
# plt.savefig(os.path.join(img_dir, 'filename.jpg'))

plt.show()
