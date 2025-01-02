import numpy as np
import cv2
import os

# Load the grayscale image from the specified path
image_path = 'images/greyscaleimage.png'
frame = cv2.imread(image_path)  # Read the image using OpenCV

# Define paths to the model files
protofile = "../model/models_colorization_deploy_v2.prototxt"  # Path to the model architecture
weightsfile = "../model/colorization_release_v2.caffemodel"    # Path to the model weights
pts_in_hull = np.load("../model/pts_in_hull.npy")              # Load points for colorization

# Load the pre-trained model using Caffe framework
net = cv2.dnn.readNetFromCaffe(protofile, weightsfile)

# Prepare points for colorization by reshaping them to match the expected input format
pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1)

# Set the blobs for the network layers with prepared points
net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull.astype(np.float32)]

# Set a constant blob for another layer to adjust colorization output
net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

# Define input dimensions for resizing images (224x224 is commonly used for models)
W_in = 224
H_in = 224

# Convert the image from RGB to Lab color space for processing
img_rgb = (frame[:, :, [2, 1, 0]] * 1.0 / 255).astype(np.float32)  # Normalize RGB values to [0, 1]
img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)                  # Convert RGB to Lab color space
img_l = img_lab[:, :, 0]                                            # Extract L channel (lightness)

# Resize and normalize the L channel to fit the model input size
img_l_rs = cv2.resize(img_l, (W_in, H_in))                         # Resize L channel to (224, 224)
img_l_rs -= 50                                                      # Normalize L channel by subtracting 50

# Set input to the network and perform a forward pass to get color predictions
net.setInput(cv2.dnn.blobFromImage(img_l_rs))                      # Prepare input blob for network
av_dec = net.forward()[0, :, :, :].transpose((1, 2, 0))           # Forward pass and transpose output

# Resize predicted ab channels back to original image dimensions
(H_orig, W_orig) = img_rgb.shape[:2]                                # Get original image dimensions
ab_dec_us = cv2.resize(av_dec, (W_orig, H_orig))                   # Resize predicted ab channels

# Concatenate L channel with predicted ab channels to form Lab image
img_lab_out = np.concatenate((img_l[:, :, np.newaxis], ab_dec_us), axis=2)

# Convert Lab image back to BGR color space for display/saving
img_bgr_out = np.clip(cv2.cvtColor(img_lab_out, cv2.COLOR_Lab2BGR), 0, 1)  # Clip values between [0, 1]

# Save the output colored image as a new file
outputfile = os.path.splitext(image_path)[0] + '_colored.png'      # Create output filename
img_bgr_process = (img_bgr_out * 255).astype(np.uint8)              # Convert back to uint8 format for saving
cv2.imwrite(outputfile, img_bgr_process)                            # Save the processed image

print('Image saved as ' + outputfile)                                # Print confirmation message
