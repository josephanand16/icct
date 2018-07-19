# icct
Image corruption classification tool

# image_aug.py

The script can be used to generate augmented images.
Pass a source path of the image.
Mentioned the type of corruption without extra spaces.
A folder will be created in an output directory name same as the class of corruption.

## Implemented Noisy Filters:

### Original Image

![Original Image](LanRed4.jpg)

### 1. Salt and Pepper Noise

![Salt and Pepper Noise](saltPepper_10rgb.jpg)

### 2. Speckle Noise 

![Speckle Noise](speckle_1rgb.jpg)

### 3. Image Glitching

![Glitched Image](glitched_1_0.jpg)

## Need to do

### 1. Export data to CSV file "class of corruption : Image name"
### 2. Add more augmentation techniques on color,hue saturation.
