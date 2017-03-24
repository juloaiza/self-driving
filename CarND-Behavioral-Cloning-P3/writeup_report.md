# **Behavioral Cloning**
---

** Behavioral Cloning Project **

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output/cnn_architecture.png "Model Visualization"
[image2]: ./output/center.png "Good Driving"
[image3]: ./output/r1.png "Recovery Image"
[image4]: ./output/r2.png "Recovery Image"
[image5]: ./output/r3.png "Recovery Image"
[image6]: ./output/image_original.png "Normal Image"
[image7]: ./output/image_flipped.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network based on this [paper from Nvidia](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf), below some details (model.py lines 94-112),

![alt text][image1]

* Data normalization using a Keras lambda layer (code line 95).
* 3 Convolution layers, with 5x5 filter, stride of 2x2 and depths of 24, 36 and 48
* 2 Convolution layers, with 3x3 filter and depths of 64
* To introduce nonlinearity (code lines 98, 100, 102 and 104), the model includes RELU layers.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 107,109 & 111).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 138).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Also I added the left and right side cameras with a correction factor to help steer back to the center

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to predict the steering wheel angle based on the images generate by the cameras installed on the car.

My first step was to use a convolution neural network model similar to Nvidia architecture. I thought this model might be appropriate because it was already tested by Nvidia team
In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model adding three dropout layers among the fully connected layers

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track so to improve the driving behavior in these cases, I augmented the data flipping the image and also crop the image to speed up the training process.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 94-112) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 160x320x3 RGB image   							|
| Normalization        		| Using Lambda function  							|
| Cropping2D       		| Cropping image from 160x320x3 to 65x320x3 |
| Convolution 5x5     	| 2x2 stride, valid padding 	|
| RELU					|	Activation function	|
| Convolution 5x5     	| 2x2 stride, valid padding 	|
| RELU					|	Activation function	|
| Convolution 5x5     	| 2x2 stride, valid padding 	|
| RELU					|	Activation function	|
| Convolution 3x3     	| 1x1 stride, valid padding 	|
| RELU					|	Activation function											|
| Convolution 3x3     	| 1x1 stride, valid padding 	|
| RELU					|	Activation function											|
|Flatten layer|Function flattens into two dimensions|
| Fully connected		| 1164 inputs -> 100 outputs							|
| dropout(0.5) | Reduce overfitting. Turn on only during training|
| Fully connected		|100 inputs -> 50 outputs 	|
| dropout(0.5) | Reduce overfitting. Turn on only during training|
| Fully connected		|50 inputs -> 10 outputs 		|
| dropout(0.5) | Reduce overfitting. Turn on only during training|
| Fully connected		|10 inputs -> 1 outputs			|


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn how to recover from the the edge of the road. These images show what a recovery looks like starting from left side to the center:

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would help to generalize the model. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 8246 number of data points(Track1 + Track2). I then preprocessed this data by a generator to avoid to store in memory all at once.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

The training data was 80% for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as any more does not reduce the MSE much at all. I used an adam optimizer so that manually training the learning rate wasn't necessary.
