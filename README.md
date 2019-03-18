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

My implementation of the model is in the source file [nvidia.py](models/nvidia.py) has _5 Convolution Layers_ and _3 Fully Connected Layers_. Additionally embedded in the model [input  layers](#L27-33) apply pre-procesing to the RGB input images of dimensions 160 _high_ x 320 _wide_ :
 1. a __colourspace conversion__ layer changing RGB color space to YUV (160,320)
 2. a __normalisation layer__ to scale the images values in the range of [-1, 1]; 
 3. a __cropping layer__ to crop the input images to  a region interest of the road that excludes the Horizon and bonnet of the car.

A diagrammatic representation of the model architecture is shown below.  This has been illustrated using[NN-SVG](http://alexlenail.me/NN-SVG/index.html).

![Architecture Layers Diagram](doc/architecture_layers.svg)

 I verified the architecture of the designed vs actual model implementation conducted with Tensorboard to view the graph. I utilised in training a Keras callback to Tensorboard, see [train.py](train.py#L148), to logging the training and validation loss as well as store a copy of the [neural network architecture graph](doc/architecture_graph.png).   This verification I found invaluable in the 'Traffic Signs Classifier' project where I used the resultant graph visualised in Tensorboard to identify I incorrectly implemented a layer due to a silly typo, hence I have made it standard practice to visualise the actual implementation vs desired deigned implementation.

Each training run also writes out a summary the model architecture as text See [logs/20190317-130803.log#L68-L134] lines 68-134 by using the Keras command [`model.summary()`](train.py#L142). 

The convolution layers reduce the cropped YUV image from __80x160@3__ input to a __2x32@64__ filter output with each layer progressively increasing in depth capturing different features.
 
The first [three convolution layers](train#L35-53) use the same layout of:
 1. __5x5 filter sizes__  moving across the filter slices with a __2x2 stride__
 2. Non linear __ELU Activation Function__ 
 3. __2x2 Max Pooling Layer__
 4. __Batch Normalizaion__ regularization function
 
 The next [two convolution layers](train.py#L66-74) use the same layout of:
 1. __3x3 filter sizes__  moving across the filter slices with a __1x1 stride__
 2. Non linear __ELU Activation Function__ 
 3. __Batch Normalizaion__ regularization function
 
 The output of the convolution layer is flattened to a __1x4096__ and fed into __3 x Fully Connected Layers__. Again each layer progressively reduces the output from the flattened __4096__ output of the convolution filter layer to a fully connected __128 > 64 > 16 > 1__ single value decimal output for the steering angle. 

### Over-fitting and Regularisation Considerations
 I added to the model regularisation in the forms of Batch Normalisation and Dropout to aid the models ability to produce 'alternate representations', generalisation and reduce over fitting.  My method of applying regularisation was to use "Batch Normalisation" in in the _Convolution Layers_ after non-linear activation and Max Pooling, whilst in Fully Connected Layers. Dropout was set to a 50% keep rate. The dropout rate greatly affected the number or training iterations required going form a consistent 8 epochs to requiring 30+ epochs. 

### Model Parameter Tuning
The model didn't need a lot of parameter tuning as the chosen optimised in the implementation was the [Adam Optimiser](train#L126) . The main considerations were were the **initial learning rate** which was chosen to be **0.0002**. This was established though several runs as higher values in the order of _1e-2_ and _1e-3_ would cause a high rate of drop in the loss, though there were clear periodic  oscillation around a minima. Choosing a smaller learning rate in the order of _1e-4_ meant mode epochs were required for the loss to decrease to an asymptote plateau, though the validation loss was more. The learning rate was easily changed through a command line parameter `--learn_rate 0.0002` being passed to `train.py` I also included a callback to a learning rate scheduler to force the decrease of the learning rate by half to a limit of _1e-5_ when there was no change in the loss.  The feature of the Adam optimiser, Ternsorboard callback to view losses and the learning rate scheduler reduced the need to perform the number of runs to the the model parameters. 

### Training Data
The training data utilised to train the model was the sample Udcaity data provided with the project that provided **8036  steering angles with camera views** of _centre, left and right camera_. **I augmented this data** by recording additional data from the simulator by:
- Driving around the track in reverse recording an additional **4292 steering angles with camera views**. The strategy behind driving reverse around the track was to provide additional steering angle data to aid in creating a balanced data set where there is not a bias to turning left or right.
- Short recovery driving session to record recovery angles where the car would drive off the track unless there was a steering correction input.  This was done around several points on the track where there were different surface transitions on the edge (eg, road -> read-white chicane, road -> ripple strip, road -> lines, road -> dirt) where the steering angles for correction were established and then the recording for the recovery made.  This added an additional **1259 steering angles with camera views**.
-   
Thus the data set used for training and validation totalled **13587 steering angles**

## Training Strategy
My model development and training strategy was iterative. I initially built basic models like what was illustrated in the lectures like
- [simple.py](models/simple.py) : A model where the cropped RGB images were flattened and fed into a Dense output layer 
- [lenet.py](models/lenet.py): A simple LeNet style model with 2 x Convolution followed by 2 x Fully Connected layers eith RELU activation.

These models formed the basis for developing the [train.py](train.py) training harness  where the python library _argparse_ was used to set default and enable variation in the training modes (eg dropout, early stopping, left right camera offset correction, multi=processing).

What was evident in these early stages of development was that my initial data generator that read batches of training  images the _drive_log,csv_ from disk was inefficient.
