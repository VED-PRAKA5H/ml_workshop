# ml_workshop
This is a machine learning workshop repo.
## Part 1: EDA
* We will use the iris dataset.
* And doing the exploratory data analysis on iris dataset.

## Part 2: Face Detection with Dlib and OpenCV
* This project uses OpenCV and Dlib to perform real-time face detection and landmark identification from a webcam feed.
* The detected faces are highlighted with bounding boxes, and facial landmarks are marked with circles.

## Part 3: Coloring of Black & White Image
### The Zhang Algorithm
**Credit**: The colorization algorithm was developed by Zhang, et al, and is detailed here: [click here](http://richzhang.github.io/colorization/)
* An overview of image processing and recognition.
* Build one such working model from scratch.
* Learn the basics of Classification & Colorization Problems, Neural Networks, etc.
* Models:
  * `models_colorization_deploy_v2.prototxt`: This file defines the architecture of the convolutional neural network (CNN) used for colorization. It specifies how layers are connected, including convolutional layers, activation functions, and any normalization layers. The model is designed to take a grayscale image's L channel as input and predict corresponding a and b channels, which represent color information. This architecture is based on a modified VGG network tailored for the task of image colorization.
  * `colorization_release_v2.caffemodel`: This file contains the trained weights and parameters of the CNN defined in the `.prototxt` file. It represents a pre-trained model that has been trained on a large dataset of color images to learn how to map grayscale images to their colored counterparts. During inference, this model uses the weights stored in this file to perform predictions based on input images processed through the network architecture specified in the `.prototxt` file.
  * `pts_in_hull.npy`: This file contains the cluster centers for the ab channels in the Lab color space. The centers represent quantized values that correspond to different colors, allowing the model to map grayscale images to color outputs effectively. The values in this file are used as convolution kernels in the neural network to predict the ab channel values based on the input lightness (L channel) from grayscale images.