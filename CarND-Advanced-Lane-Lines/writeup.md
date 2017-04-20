## **Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistor_chessboard.png "Undistorted"
[image2]: ./output_images/undistor_car.png "Road Transformed"
[image3]: ./output_images/binary_morphological.png "Binary Image and Morphological Transformations"
[image4]: ./output_images/warped_image.png "Warp Image"
[image5]: ./output_images/color_fit_lines.jpg "Fit Visual"
[image6]: ./output_images/final_output.png "Output"
[video1]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained from 4th to 7th code cells of the IPython notebook located in "P4.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

###Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
Here comparison between original image and image after applying distortion-corrected
![alt text][image2]
#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color (S-channel) and gradient thresholds to generate a binary image, thresholding steps at cell 9 in P4.ipynb.  Here's an example of my output for this step.

Also after binary image generated, it was applied morphological transformations to improve the quality of the lines. In this way, it's easier to detect the center of the line using the histogram method.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `four_point_transform()`, which appears in 2nd cell of the Ipython notebook. The `four_point_transform()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I choose to hardcode the source and destination points in the following manner:

```
tl = (imshape_max_x * (.5-mid_width/2), imshape_max_y*height_pct)
tr = (imshape_max_x * (.5+mid_width/2), imshape_max_y*height_pct)
br = (imshape_max_x * (.5+bot_width/2),imshape_max_y*bottom_trim)
bl = (imshape_max_x * (.5-bot_width/2),imshape_max_y*bottom_trim)
src = np.array([tl, tr, br, bl], dtype = "float32")

dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 461, 504      | 320, 0        |
| 819, 504      | 960, 0      |
| 1139, 673     | 960, 720      |
| 140, 673      | 320, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

 To identify lane-line pixels I'm detecting the peaks in a histogram to find the left and right line center. From that point, I can use a sliding window, placed around the line centers, to find and follow the lines up to the top of the frame. Then I extract left and right line pixel positions to fit my lane lines with a 2nd order polynomial.

 The code is avaliable in cell 13 of the Ipython notebook

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did the radius of curvature in the 17th code cell of the Ipython notebook. The pixel values of the lane are scaled into meters using a scaling factors. Then, these values were used to determine new polynomial coefficients in meters. Finally, using the formula provided in the lectures, it was computed the radius of curvature.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this in the 18th code cell of the IPython notebook.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My pipeline is weak during shadows or lack of contrast condition. Also I need to improve my algorithm to be more robust to detect in efficient way the center lines.
