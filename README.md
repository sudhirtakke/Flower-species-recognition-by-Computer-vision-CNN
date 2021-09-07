# Flower-species-recognition-by-Computer-vision-CNN
Image classification based computer vision model CNN

![image](https://user-images.githubusercontent.com/49444353/128812725-11466206-402b-4973-aa47-cad30879e74a.png)

### Introduction:

We will deal with image classification problem namely Flower Species Recognition, which is a hard problem because there are millions of flower species around the world. As we know machine learning is all about learning from past data, we need huge dataset of flower images to perform real-time flower species recognition

This is typically a supervised learning problem where we must provide training data (set of images along with its labels) to the machine learning model so that it learns how to discriminate each image (by learning the pattern behind each image) with respect to its label

We will not train whole convolutional network ourselves but we will use pre-trained network i.e. VGGNnet that is trained on Imagenet dataset. We will see whether to use this network for feature extractor or as an initial network for fine tune. VGGNet is great because it's simple and has great performance.

If we use VGGNet for feature extraction, it means we have to use models in two parts, Convolutional base and classifier.
 
 ### Problem Statement:
 
Plant or Flower Species Classification is one of the most challenging and difficult problems in Computer Vision due to a variety of reasons e.g. collecting plant/flower dataset is time consuming task, flower like sunflower may look similar to Daffodil causing inter-class variation problem, sometimes single sunflower image might have differences within its class itself causing intra-class problem. 

Again segmenting the flower region from an image is a challenging task for CNN because it need to understand to remove unwanted background and take only foreground object for learning.

Considering all above problems, suppose we have a dataset given, covering images of five classes of flowers: daisy (chamomile), tulip, rose, sunflower, dandelion, we are asked to build a CNN model that will be trained with this dataset and should predict the label/class of the flower accurately.

### Data pre-processing:

We have sufficient data points for the 5 classes of flower species. Firstly, we create separate folders for Train, Validation and Test data set for each class in disk. Total 2000 samples for train, 750 each for validation and test sets.

We will then pass all these images from created folders through convolutional base  for feature extractions however before this, we use ImageDataGenerator utility for image augmentation purpose to help model become robust as it will get transformed images as input while training, validation. It will output image size as 224, 224 as VGG16 use this input size as default. 

Output of pr-processed image data in the form of tuples (x, y) where x is a numpy array containing batch of images (in our case 32, 224, 224, 3) and y is numpy array with corresponding labels (32, 5) will be fed to VGG16 model from directory.

VGG16 conv base model with multiple layers of Conv2d, MaxPooling2d will generate final output of (32, 7, 7, 512), 32 being batch size, 7 x 7 feature map size and 512 is depth of the feature map (different features of the image like, edge, corners, curves etc.)

Numpy array for Label ‘y’ will have same shape i.e. 32, 5 (32 being batch size and 5 the number of classes)

So whole process of data augmentation and feature extraction will run together for each of train, validation and test data sets and this will be the most time taking part from computational perspective.

So this is all a part of transfer learning approach wherein we have used existing pre-trained model VGG16 for feature extraction instead of training new model from scratch.  
 
### Model building for classifier 

We will build a classifier based on fully-connected layers. This classifier will use the features extracted from the convolutional base.

This sequential model will take input as (7, 7, 512), flatten layer will flatten into 25088 units, 

Dense layer of 256 units and ‘Relu’ as activation function will process this flattened output to give 256 one dimensional output units, dropout of 0.3 is applied to avoid overfitting.

Finally it is a dense layer of 5 units (corresponding to 5 number of classes) using ‘softmax’ as activation function, this being multi-class problem.

Model further will be compiled with optimizer as ‘Adam’ and loss as ‘categorical_crossentropy’  

Model will be further fitted into train features and train labels and also with validation data. With the use of batch size as 32 and epoch as 20, validation accuracy comes near to 78%. 

### Model Evaluation 

![image](https://user-images.githubusercontent.com/49444353/128813401-bb547c63-ce2f-4655-add2-734de994a855.png)

Training accuracy is increasing with number of epochs and reaching max to 99.50 and that is sign of overfitting. However validation accuracy is ranging from 70% to 79% and finally settling to around 78% after 20 epochs

### Model predictions 

 We will build a function to predict the class of flower and visualize it from random samples from the test data set
 
Firstly, we extract features of the randomly selected image using conv base VGG16 feature extractor model

Then, we will be using classifier model to get the class probabilities and then use numpy.argmax function to get the index of maximum probability value from the array. 

Here are the images and predictions made for few randomly selected images….

![image](https://user-images.githubusercontent.com/49444353/128813578-b82adff1-9bec-4234-bdfb-db6dae68856a.png)


[Jupyter Notebook](./Image classification.ipynb)









 




