# **Traffic Sign Recognition**

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output/traffic_sign.png "Visualization"
[image2]: ./output/sign_distribution.png "Distribution"
[image3]: ./output/extra_traffic_sign.png "Extra Traffic Sign"
[image4]: ./output/top5.png "Top 5 Softmax Probability"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

**Answer:**
You're reading it! and here is a link to my [project code](https://github.com/juloaiza/self-driving/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

**Answer:**
The code for this step is contained in the second code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ? 34799
* The size of test set is ? 12630
* The shape of a traffic sign image is ? 32x32x3(RGB)
* The number of unique classes/labels in the data set is ? 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

**Answer:**
The code for this step is contained in the third and forth code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. First display some traffic sign images

![alt text][image1]


Bar chart showing the signs distribution

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

**Answer:**
The code for this step is contained in the fifth code cell of the IPython notebook.

I applied mean normalization to normalize the image data because it will help to converge much faster any optimization algorithm.

I didn't use grayscale conversion as color is very important feature that CNN require to learn to identify traffic sign image.

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

**Answer:**
The code is contained in the sixth code cell of the IPython notebook.  

Training data was shuffle using sklearn library

My final training set had 34799 number of images. My validation set and test set had 4410 and 12630 number of images.


#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

**Answer:**
The code for my final model is located in the eigth cell of the ipython notebook.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x32 	|
| RELU					|	Activation function											|
| Max pooling	      	| 2x2 kernel size, 1x1 stride,  outputs 29x29x32 				|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 27x27x64 	|
| RELU					| Activation function|
| Max pooling	      	| 2x2 kernel size, 1x1 stride,  outputs 26x26x64 				|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 24x24x128 	|
| RELU					|	Activation function											|
| Max pooling	      	| 2x2 kernel size, 2x2 stride,  outputs 12x12x128 				|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 10x10x256 	|
| RELU					|	Activation function											|
| Max pooling	      	| 2x2 kernel size, 2x2 stride,  outputs 5x5x256 				|
|Flatten layer|Function flattens a Tensor into two dimensions|
| Fully connected		| 6200 inputs -> 1200 outputs							|
| Fully connected		|1200 inputs -> 400 outputs 	|
| dropout | Reduce overfitting. Turn on only during training|
| Fully connected		|400 inputs -> 120 outputs 		|
| dropout | Reduce overfitting. Turn on only during training|
| Fully connected		|120 inputs -> 43 outputs			|
| Softmax				| softmax_cross_entropy_with_logits       									|
| Regularization| L2 Regularization|



#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

**Answer:**
The code for training the model is located in the seventh cell and nine cell of the ipython notebook.

To train the model, I used an Adam optimizer with a learning rate 1e-3. Due to the size of my model batch size is 64 and epochs 20

Besides, it was added a L2 Regularization with beta 5e-6 to improve performance on validation / test accuracy, around 3% gain.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

**Answer:**
The code for calculating the accuracy of the model is located in the tenth and eleventh cells of the Ipython notebook.

My final model results were:
* validation set accuracy of 0.985
* test set accuracy of 0.977

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?<br>
**Answer:** The [LeNet-5 network](https://github.com/udacity/CarND-LeNet-Lab/blob/master/LeNet-Lab-Solution.ipynb) designed for handwritten. It was chosen to provide a performance baseline before to try different architecture.

* What were some problems with the initial architecture?<br>
**Answer:** Poor accuracy during training below 90% and slow to converge (more of 100 iterations)

* How was the architecture adjusted and why was it adjusted?<br>
**Answer:**
For this problem it was required to create a model with more freedom parameters to capture image patterns, but with the risk of poor generalization. The original network was using 3 convolutional layer with kernel size 5x5 and 3 fully connected layer, however it was added an extra convolutional layer. All the kernel in each layer modified 3x3.

Same for fully connected layer, from 3 to 4. Between the last two fully connected layer were added two dropout  as a part of the regularization.

For the first two maxpool layer the stride was modified from 2x2 to 1x1. The reason was to capture more information of each image.

* Which parameters were tuned? How were they adjusted and why?<br>
**Answer:** Sign traffic images are very complex because there are different color, form and shapes so decreasing the filter size and increasing depth helped to capture more characteristics and patterns of each image.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?<br>
**Answer:**
Adding dropout to help reduce overfitting issues. L2 regularization to penalize large activations leading to improve accuracy of the model.

If a well known architecture was chosen:
* What architecture was chosen?<br>
**Answer:** This is an image recognition problem so the best fit is to use Convolutional Neuron Network (CNN).
CNN use shared parameters across space to extract patterns over an image.
We can also use a just fully connected network but the performance will be always poor due to amount of parameters required, leading to more noise addition during the training process.

* Why did you believe it would be relevant to the traffic sign application?<br>
**Answer:** The traffic sign is a multi-class single image classification challenge.


* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?<br>
**Answer:** I can't use only accuracy to evaluate my network. I need to add more measurement like precision/recall to get more details of the network and also try with a complete different dataset.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

**Answer:**
Here are 12 German traffic signs that I found on the web:

Image size 32x32xRGB

![alt text][image3]

The first time I had a problem with children crossing sign due to very poor image quality. The image was replaced with a better quality and is working fine.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

**Answer:**
The code for making predictions on my final model is located in the 13th cell of the Ipython notebook.

Here are the results of the prediction:

|Actual Image | Prediction |
|:--:|:--:|
| Yield | Yield |
| Speed limit (60km/h) | Speed limit (60km/h) |
| Stop | Stop |
| Ahead only | Ahead only |
| Road work | Road work |
| Roundabout mandatory | Roundabout mandatory |
| Children crossing | Children crossing |
| Beware of ice/snow | Beware of ice/snow |
| Speed limit (30km/h) | Speed limit (30km/h) |
| Slippery road | Slippery road |
| No passing | No passing |
| Speed limit (50km/h) | Speed limit (50km/h) |



The model was able to correctly guess 12 of the 12 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 97.70

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

**Answer:**
The code for making predictions on my final model is located in the 14th cell of the Ipython notebook.

In general, the model is running very well to predict the image with probability of 1.0. I defined my own probability confidence threshold (>0.9) where there was a image below this threshold (30-0.83). Here the Precision/Recall,

Confidence Threshold > 0.9

Benchmarking my CNN based Softmax Probabilities

| | Actual True| Actual False|
|:-------------: |:-------------:| :-----:|
| Predicted True| 11         | 0  |
| Predicted False| 1 | 0 |

#### Precision = 1

#### Recall = 0.916

#### F1 score = 0.956

The top five soft max probabilities were

![alt text][image4]
