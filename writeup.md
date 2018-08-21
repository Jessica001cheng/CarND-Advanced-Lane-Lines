## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

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

[image1]: ./output_images/undistort.png "Undistorted"
[image2]: ./output_images/testImagedistort.png "Road Transformed"
[image3]: ./output_images/yellowLab.png "Binary Example"
[image4]: ./output_images/whiteLuv.png "Binary Example"
[image5]: ./output_images/YellowWhiteLine.png "Binary Example"
[image6]: ./output_images/SobelXThres.png "Binary Example"
[image7]: ./output_images/SobelYThres.png "Binary Example"
[image8]: ./output_images/combineGradiantAndLABLUV.png "Binary Example"
[image9]: ./output_images/source_line_drawed.png "Warp Example"
[image10]: ./output_images/Perspective_transformed.png "Warp Example"
[image11]: ./output_images/polynomial_line.png "Fit Visual"
[image12]: ./output_images/LaneOnTestImages.png "Output"
[video1]: ./video_output/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines #14 through #54 of the file called `camera_calibrate.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

First, as there is one frame from the project_video.mp4 cause lane detection fail. I took this frame as a test image, named "screenshot.jpg".

Then I tried a lot of way to create a thresholded image. And reviewer(Thanks for the review) give me the suggestion as below:

. Applying color threshold to the B(range:145-200 in LAB for shading & brightness changes and R in RGB in final pipeline can also help in detecting the yellow lanes.

. And thresholding L (range: 215-255) of Luv for whites.

The code is (thresholding steps at lines #12 through #121 in `threshold_binary.py`).  Here's an example of my output for this step. 

Detect Yellow Line in B channel in LAB colorspace with yellow mask

low: [145]

hight: [200]

![alt text][image3]

Detect White Line in L channel in LUV colorspace with white mask

low: [215]

hight: [255]


![alt text][image4]

Combine Yellow Line and White Line together:

![alt text][image5]

Then I convert image into HLS colorspace and do Sobel detection:

Sobel X on S channel:

![alt text][image6]
Sobel Y on S channel threshold:

![alt text][image7]
Combine Sobel X/Y oand Yellow and White Line:

![alt text][image8]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `perspective.py`. The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
bottomY = 720
topY = 500
left1 = (201, bottomY)
left1_x, left1_y = left1
left2 = (528, topY)
left2_x, left2_y = left2

right1 = (768, topY)
right1_x, right1_y = right1
right2 = (1100, bottomY)
right2_x, right2_y = right2
src = np.float32([
                  [left2_x, left2_y],
                  [right1_x, right1_y],
                  [right2_x, right2_y],
                  [left1_x, left1_y]
dst = np.float32([
                  [offset, 0],
                  [img_size[0]-offset, 0],
                  [img_size[0]-offset, img_size[1]],
                  [offset, img_size[1]]
                  ])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 528, 500      | 200, 0        | 
| 201, 720      | 200, 720      |
| 1100, 720     | 1080, 720      |
| 768, 500      | 1080, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image9]
warped result on test image:

![alt text][image10]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image11]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines #188 through #203 in my code in `advanced_lane.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines #156 through #178 in my code in `advanced_lane.py` in the function `drawLine()`.  Here is an example of my result on a test image:

![alt text][image12]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./video_output/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. 

1. I tried a lot of way to create a thresholded image. And finally decided to use a combination of detect Yellow and White Line in HSV colorspace and gradient thresholds to generate a binary image.

2. I use the distance of between white dash line to calcuate ym_per_pix. As this is more accurate on my perspective image.

3. I use reviewer's suggestion to detect yellow line in LAB colorspace and white line in LUV colorspace. it reduce a lot of noises.

4. I save previous 10 frames polynomial parameters. when get the current frame polynomial, I check it with sanityCheck(Line 305~326 in advanced_lane.py). If current frame is not valid, I will use the last frame. If current is valid, I will avarage with last 10 frames and get a new polynomial parameters.

5. I clip the video from 0:39~0:42 to test1.mp4. With my above changes, the test1_out.mp4 can detect lane smoothly.

6. I check the distance of start/middle/end of the road. If the distance is not in a range, I will use the last frame.