# SDC Behavioural Cloning Project

## Aim
The goal of the Behavioual Cloning Project was to navigate a simulated vehicle  around a test track by using a  Convolutional Neural Network to predict steering angles.

## Solution Files
My solution to this project consists of the following source code files that are needed to develop the model:

 - [train.py](train.py): Contains code for the training of a CNN using
 - a CSV file of the steering angles, throttled image files from centre,
   left and right vehicle cameras.  This program takes arguments to control the training of the model.
  - [nvidia.py](models/nvidia.py): This contains the code implementing the model used to predict the steering angles.  It is called NVIDIA as the model is based upon the research NVIDIA did for using deep learning to control a self driving car (see their Aug 2016 blog article [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/deep-learning-self-driving-cars/).
  - [datagen.py](models/generator.py): Contains the code for my data generator class [ImageGenerator()](/utils/datagen.py#L12-L111) that implements the file selection, reading from disk and image augmentation of the training images.
  - [drive.py](drive.py): If the Udacity provided utility for loading the trained model and interfacing it with the Udacity Car Simulator.  The final version is largely unmodified from that provided apart form setting the speed form 9 to 25.

Additional files of interest which are aviailable are the:
- [logs/train_20190317-170803.log](logs/train_20190317-170803.log): A log of the runtime training session.
- [trained/nvidia_20190317-170803.h5](trained/nvidia_20190317-170803.h5): the resultant model that successfully predicts steering angles for Track 1 of the car simulator.  This can be downloaded and used with the Udacity car simulator to drive the vehicle around Track 1 with the _drive.py_ program (ie `python drive.py trained/nvidia_20190317-170803.h5`).

### Running the Model
The model can be run to navigate the vehicle around the Udacity Car simulator Track 1 simply upon the following command:
```python drive.py trained/nvidia_20190317-170803.h5
```
The development environment the model was developed in was an Ubuntu 18.04 host loaded with:
- _Python 3.6._ 
- _Tensorflow r1.13_ compiled for GPU with NIVIDIA CUDA 10.0
- _Keras 2.2.4_
The model is saved in HDF5 format using _h5py version 2.9.0_


## Model Architecture and Training Strategy
### Model Architecture Overview
The final model selected is based upon the NVIDIA End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) research/demonstration paper. 

My implementation of the model is in the source file [nvidia.py](models/nvidia.py) has 5 convolution layers and 3 Fully connected layers. Additionally embedded in the model input layers apply pre-procesing to the RGB input images of dimensions 160 _high_ x 320 _wide_ :
 1. a __colourspace conversion__ layer changing RGB color space to YUV (160,320)
 2. a __normalisation layer__ to scale the images values in the range of [-1, 1]; 
 3. a __cropping layer__ to crop the input images to  a region interest of the road that excludes the Horizon and bonnet of the car.

A diagramatic representation of the model architecture is shown below.  It was created using a Keras callback to Tensorboard (see [train.py](train.py#L148) logging the training and validation losses.


## Training Model
