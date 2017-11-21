# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the Nvidia architecture that is present in the following [paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

Below are the layers that are part of my model in the mentioned order:

1. Cropping2D layer to crop top and bottom portion of the image since they do not contain any useful information about the road to train the model.

2. Lambda layer to perform normalization on the data. This would help the model to converge faster.

3. This is followed by 5 layers of Convolution + Relu.

    * Depth - 24, Kernel - (5, 5), Stride - (2, 2), Padding - 'VALID'
    * Depth - 36, Kernel - (5, 5), Stride - (2, 2), Padding - 'VALID'
    * Depth - 48, Kernel - (5, 5), Stride - (2, 2), Padding - 'VALID'
    * Depth - 64, Kernel - (3, 3), Stride - (1, 1), Padding - 'VALID'
    * Depth - 64, Kernel - (3, 3), Stride - (1, 1), Padding - 'VALID'

4. After the features are exctracted using the convolutional network, we have fully connceted layers to make the steering prediction.

#### 2. Attempts to reduce overfitting in the model

I have added in a couple of Dropout layers in between the FC layers so that the model does not overfit.
Trained the model for 3 epochs and observed that validation accuracy did not increase which is a good sign that the model is not overfitting.

#### 3. Appropriate training data

Initially I used the sample data that has been provided. Later I collected my own data for traning the model.
I go into more details on this later in the writeup.

### Model Architecture and Training Strategy

Initially I trained the model using the sample dataset that has been provided.
But there were couple of issues with my model and as a result of this, it was not performing well in some turns - Car was going off the road.

Problem was that I did not add activation functions after the fully connected layers. So I fixed this and added
in Dropout layers to avoid over-fitting. After doing this the car was successfully able to go around the track.

Then I started collecting my own data as follows.
* Two laps around the track with the car staying in the center of the road all the time.
* Same as above but going around the track in opposite direction.
* Couple more laps around the track to collect recovery data.

All this data was split into train and validation with a ratio of 4:1.

Data augmentation is performed as part of the generator function itself.
Every time the generator is called, it picks up a batch of samples from the shuffled data.
After this it randomly picks up the image data from one of the cameras and adds a correction factor to the steering angle. I also added a randomly image flip into this pipeline to help with augmentation.

With my final trained model, car is able to drive successfully around the track.
It is able to handle every turn in the track without any wobbles.
But there is a little wobble in the straight portions of the road and on the bridge.
I have collected training data using keyboard, so this would be one reason for the wobble since the steering measurement would not be precise. Also I believe that adding more balanced training data for the bridge portions of the road would fix the wobble.