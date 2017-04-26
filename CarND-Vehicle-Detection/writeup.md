**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image21]: ./output_images/color_space25.png
[image22]: ./output_images/color_space_31.png
[image23]: ./output_images/HOG_image.png
[image3]: ./output_images/training_phase.png
[image4]: ./output_images/sliding_windows.png
[image5]: ./output_images/sliding_window.png
[image6]: ./output_images/bboxes_and_heat.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 3rd code cell of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I explored different color spaces and the car pixel looks to cluster well using YCrCb space. The code is contained in the fifth code cell of the IPython notebook. Below some images comparison between different color space,

![alt text][image21]

![alt text][image22]

Finally, I applied `skimage.hog()` to an car image with the following parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. Here is an example using the `Luv` color space and HOG :


![alt text][image23]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters to get the best training accuracy. The code is contained in the 7th code cell of the IPython notebook, below are my final settings,

| Parameters         		| Value   |  Description	        					|
|:--------------------:|:--------:|:-------------------------------:|
|color_space| YCrCb | Can be RGB, HSV, LUV, HLS, YUV, YCrCb|
|orient | 9 |  HOG orientations|
|pix_per_cell| 8 | HOG pixels per cell|
|cell_per_block|  2 | HOG cells per block|
|hog_channel| ALL | Can be 0, 1, 2, or ALL|
|spatial_size| 323x32 | Spatial binning dimensions|
|hist_bins| 16 | Number of histogram bins|

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

For the training phase this is how my pipeline looks,

![alt text][image3]

I trained a linear SVM using the python libraries from scikit-learn. The code is contained in the 7th code cell of the IPython notebook.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I followed the lecture suggestion to scale the entire image  and extract HOG features just once for the entire region of interest in each full image or video frame. Using this method is more efficient to find vehicle. You can verify the code in cell 9th of the IPython notebook.

![alt text][image4]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image5]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code in cell 11th of the IPython notebook.I saved the position of each bounding boxes detected during the find_cars step. From the detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Below an example image:

![alt text][image6]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Initially, I had problem with the classifier to detect vehicle. I had to adjust the color space. Also, I need more time to build a classifier more robust. I'm thinking to try to use a deep learning neural network.    
