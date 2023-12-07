# Python Computer-Vision w/ *scikit-learn*

## Introduction
### Brief
Computer vision is a major application of Python in modern technology.
The purpose of this repository is to demonstrate image processing techniques using a Python library named *scikit-learn*, aided by other data manipulation libraries like *matplotlib* & *NumPy*.
The acquisition of images & processes involved in analysis are fundamental to many technologies. 
Computer vision, machine learning & scientific research are examples of fields in which skills in Image Processing & Analysis (IPA) are sought after within.

This repository can be viewed as an excellent introduction to Python computer vision practices for an interested engineer.

### Technique
Google Colaboratory was employed during this assignment to simplify the IPA process. 
Colab allows for the execution of code from within a browser. 
It offers a wide range of free-to-use Python libraries that are often used in IPA & data science.


## filters.py
### Preamble
It is important to understand the core difference between RGB and grayscale images:

Images are simply collections of red, green & blue (RGB) values. 
Videos are sequences of many images stored together. RGB images are represented by a series of three values, each pertaining to a shade of red, green, or blue, which can only be between 1 and 255, with 1 being the lightest shade, and 255 being the darkest shade of such colour. 
An RGB image can be visualised as a 3-dimensional array of RGB values, the specific value of these colours dictates the final colour. 
The representation of images as numerical values is a necessary step in many IPA applications. RGB is the most well-known colour model. Grayscale is the most simple model. 

Instead of a 3-dimensional array, a 2-dimensional one is used. This array defines the lightness or darkness of a single component, hence the name grayscale. 
The result is a single colour that can only be described as white, black or somewhere in-between. 
Grayscale images are common in IPA since they require less computer memory, and are much more computationally affordable.

Before implementing most IPA algorithms, images are usually converted to a 2-D array of greyscale pixel values, and are also resized for simpler iteration.

### Goals
1. Set up image
2. Introduce Gaussian Noise
3. Diagnose which edge detection algorithm is superior 

### Random Noise
#### Gaussian Noise
Specifically, Gaussian noise was added to the image with a mean value of 0.0 and a variance of 0.01. 
Gaussian noise is a common type of random noise occurring in imaging systems, characterised by its Gaussian probability distribution.
The addition of such noise serves as a basepoint example of a noisy image, which will eventually be optimised in various ways in future analysis. 
These kinds of algorithms are essential in IPA applications, enabling for the identification of sharp edges & areas of noise.


### Edge Detection Algorithms
#### Roberts
Roberts edge detection involves the use of two 2x2 convolution kernels, while Sobel edge detection uses two 3x3 convolution kernels. 
Roberts edge detection works by finding the edge magnitude using Robert’s cross operator.
This operator approximates the gradient of an image through discrete differentiation, the important factor being the sum of the squares of the differences between adjacent pixels.

#### Sobel
Sobel edge detection works by another discrete differentiation operator, computing the approximation of the gradient of image intensity, or simply the gradient magnitude. 
The gradient represents the change of intensity of image values, and is higher at locations of greater intensity change, i.e. edges. 

When the Sobel operator is applied to a noisy image, horizontal & vertical kernels are convolved with the image separately, the result of each convolution is used. 
These gradients are used to calculate the magnitude & direction of each gradient. What results is a gradient magnitude representing edge density for Roberts edge detection. 
For Sobel edge detection, there is both a gradient magnitude and direction. Sobel is therefore a more accurate method of edge detection. 

#### Roberts vs Sobel
It is clear to see that the Sobel operator out-performs the Roberts operator under the influence of a standard level of Gaussian noise. 
This result aligns successfully with the previous hypothesis. The Roberts edge detector employs smaller kernels than the Sobel detector, in order to approximate gradients. 
The smaller size of the kernel may identify abrupt intensity changes as edges, leading to false positives. 
Compared to the larger kernel size of Sobel, smoothening of intensity is applied, which leads to less false positives.

#### Canny
Canny is a highly effective & more computationally-expensive edge detection technique than previously described.
A familiarity in how these techniques respond to noise is crucial in the successful implementation of IPA to real, practical activities. 
According to the scikit-image documentation, the “Canny filter is a multi-stage edge detector. 
It uses a filter based on the derivative of a Gaussian in order to compute the intensity of the gradients”.

Canny is a superior edge detection algorithm due to its complexity. It involves:
1. Smoothening
2. Intensity Gradient Calculation
3. Suppression
4. Double-thresholding
5. Edge Tracking

## template_matching.py
### Preamble
Template matching is an image processing technique that employs an algorithm to identify the occurrence of small parts of an image, when given a template of said image. 

The main points of interest in a template matching algorithm are:
1. Occlusion detection
2. Non-rigid transformation detetction
3. Illumination sensitivity
4. Background clutter
5. Scale variations
6. Speed & efficiency of computation

*template_matching.py* uses normalised cross-correlation to find the best match between the template & the input image. 
The result is a map indicating similarities between the template & input image. Normalised cross-correlation is a mathematical technique used to measure the similarity between two signals or images. 
The normalised cross-correlation provides a value that represents how alike two images are, despite intensity differences or brightness variations.

### Goals
1. Set up image & template image
2. Apply template matching algorithm
3. Improve template matching algorithm

### Template Matching
Template matching by cross-correlation works by the following:
1. Input image & template are imported & converted to greyscale.
2. A loop iterates over the 2-D array of intensity values. Areas of high intensity are compared to the template.
3. Thresholding sets the minimum & maximum basis for feature extraction.

*template_matching.py* was developed to tackle a more complex task.
All instances of two separate templates are detected within the same input image.

To test the robustness of this algorithm. The template images used were rotated at varying angles between -30° and +30°.
It was observed that at original image orientation, threshold_abs = 0.8 identifies all templates and identifies some of the templates at offsets of ±5-10°, with increasing inaccuracy at higher image rotation offsets. 
However, for threshold_abs = 0.75, all templates were identified at offsets of ±5-10°, with a slight accuracy increase for angles higher than 10°. 
The drawback when using this threshold value is that in the original image orientation, all templates are identified alongside other letters like ‘I’ that resemble ‘T’ & ‘L’. 
This conclusion highlights the importance of thresholding & which value for thresholding is suitable for different applications.
