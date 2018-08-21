import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
from helper import showImages, showSidebySide
from camera_calibrate import undistortImages
from threshold_binary import combineGradients, combineGradientsOnS, combineGradientsAndColor,combineYellowWhiteOnLabLuv, combineGradientsAndLUV
from moviepy.editor import VideoFileClip
import pdb

class Lane():
    def __init__(self):
        self.fit = None
        self.fit_m = None
        self.curvature = None
        self.prevfit = []
        self.prevfit_m = []
        self.discardFrameNo = 0
    def add_fit(self, fit):
        self.prevfit.append(fit)
        if len(self.prevfit) > 10:
            self.prevfit = self.prevfit[len(self.prevfit)-10:]
            self.fit = np.mean(self.prevfit, axis=0)
        else:
            self.fit = fit
        return self.fit
            
    def add_fit_m(self, fit_m):
        self.prevfit_m.append(fit_m)
        if len(self.prevfit_m) > 10:
            self.prevfit_m = self.prevfit_m[len(self.prevfit_m)-10:]
            self.fit_m = np.mean(self.prevfit_m, axis=0)
        else:
            self.fit_m = fit_m
        return self.fit_m

# Init LeftLane and rightLane for pipeline use
leftLane = Lane()
rightLane = Lane()
# Define conversions in x and y from pixels space to meters
## as perspective transform, the offset = 200
offset = 200
image_width = 1280
image_height = 720
## the line width is 3.7m
## 3.7m = xm_per_pix * (1280 - 2*200)
between_left_right_Line = 3.7
xm_per_pix = between_left_right_Line/(image_width - 2*offset) # meters per pixel in x dimension
##  Each dashed line measures 10 feet, and the empty spaces in-between measure 30 feet.
between_dashline_m = 9.14 ##     30 feet
between_dashline_start_y = 100
between_dashline_end_y = 670
ym_per_pix = between_dashline_m/(between_dashline_end_y - between_dashline_start_y) # meters per pixel in y dimension

# Define M as 0 at begin
M = 0

# Define if need convert BGR to RGB for cv image reading
needBGR2RGB = True

# Define show image row and columns
imageRow = 4
imageCol = 3

# Define the number of Frames that discarded
discardFrameNo = 0

def adjustPerspective(image, M=M):
    """
        Adjust the `image` using the transformation matrix `M`.
        """
    img_size = (image.shape[1], image.shape[0])
    warped = cv2.warpPerspective(image, M, img_size)
    return warped

def getDiscardFrameNo():
    return discardFrameNo

def getDiscardFrameNo():
    return discardFrameNo

def findLines(binary_warped, nwindows=9, margin=110, minpix=50):
    """
        Find the polynomial representation of the lines in the `image` using:
        - `nwindows` as the number of windows.
        - `margin` as the windows margin.
        - `minpix` as minimum number of pixes found to recenter the window.
        - `ym_per_pix` meters per pixel on Y.
        - `xm_per_pix` meters per pixels on X.
        
        Returns (left_fit, right_fit, left_lane_inds, right_lane_inds, out_img, nonzerox, nonzeroy)
        """
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    try:
        ## if can not find valid point, use the previous value
        left_fit = np.polyfit(lefty, leftx, 2)
        left_fit_m = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    except TypeError:
        ##pdb.set_trace()
        leftLane.discardFrameNo = leftLane.discardFrameNo + 1
        plt.imshow(binary_warped)
        plt.savefig("notdectedLeft_" + str(leftLane.discardFrameNo) + ".png")
        left_fit = None
        left_fit_m = None
        pass

    try:
        ## if can not find valid point, use the previous value
        right_fit = np.polyfit(righty, rightx, 2)
        right_fit_m = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    except TypeError:
        #pdb.set_trace()
        rightLane.discardFrameNo = rightLane.discardFrameNo + 1
        plt.imshow(binary_warped)
        plt.savefig("notdectedRight_" + str(rightLane.discardFrameNo) + ".png")
        right_fit = None
        right_fit_m = None
        pass

    return left_fit, right_fit, left_fit_m, right_fit_m, left_lane_inds, right_lane_inds, out_img, nonzerox, nonzeroy

def visualizeLanes(image, ax):
    """
        Visualize the windows and fitted lines for `image`.
        Returns (`left_fit` and `right_fit`)
        """
    left_fit, right_fit, left_fit_m, right_fit_m, left_lane_inds, right_lane_inds, out_img, nonzerox, nonzeroy = findLines(image)
    # Visualization
    ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    ax.imshow(out_img)
    ax.plot(left_fitx, ploty, color='yellow')
    ax.plot(right_fitx, ploty, color='yellow')
    return ( left_fit, right_fit, left_fit_m, right_fit_m )

def showLaneOnImages(images, imagesName, cols = 2, rows = 4, figsize=(15,13)):
    """
        Display `images` on a [`cols`, `rows`] subplot grid.
        Returns a collection with the image paths and the left and right polynomials.
        """
    imgLength = len(images)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    indexes = range(cols * rows)
    imageAndFit = []
    for ax, index in zip(axes.flat, indexes):
        if index < imgLength:
            imagePathName = imagesName[index]
            image = images[index]
            left_fit, right_fit, left_fit_m, right_fit_m = visualizeLanes(image, ax)
            ax.set_title(imagePathName)
            ax.axis('off')
            imageAndFit.append( ( imagePathName, left_fit, right_fit, left_fit_m, right_fit_m ) )
    saveName = "./output_images/polynomial_line.png"
    fig.savefig(saveName)
    return imageAndFit

def drawLine(img, left_fit, right_fit):
    """
        Draw the lane lines on the image `img` using the poly `left_fit` and `right_fit`.
        """
    yMax = img.shape[0]
    ploty = np.linspace(0, yMax - 1, yMax)
    color_warp = np.zeros_like(img).astype(np.uint8)
    
    # Calculate points.
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    return cv2.addWeighted(img, 1, newwarp, 0.3, 0)

def drawLaneOnImage(img):
    """
        Find and draw the lane lines on the image `img`.
        """
    left_fit, right_fit, left_fit_m, right_fit_m, _, _, _, _, _ = findLines(img)
    output = drawLine(img, left_fit, right_fit)
    return cv2.cvtColor( output, cv2.COLOR_BGR2RGB )

def calculateCurvature(yRange, fit_cr):
    """
        Returns the curvature of the polynomial `fit` on the y range `yRange`.
        """
    return ((1 + (2*fit_cr[0]*yRange*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])

def calculateCenter(img, left_fit_m, right_fit_m):
    # Calculate vehicle center
    xMax = img.shape[1]*xm_per_pix
    yMax = img.shape[0]*ym_per_pix
    vehicleCenter = xMax / 2
    lineLeft = left_fit_m[0]*yMax**2 + left_fit_m[1]*yMax + left_fit_m[2]
    lineRight = right_fit_m[0]*yMax**2 + right_fit_m[1]*yMax + right_fit_m[2]
    lineMiddle = lineLeft + (lineRight - lineLeft)/2
    diffFromVehicle = lineMiddle - vehicleCenter
    return diffFromVehicle

# Function to draw the line on the images
def laneCurveProcess(binary_warped, img):
    left_fit, right_fit, left_fit_m, right_fit_m, _, _, _, _, _ = findLines(binary_warped)
    #print("left_fit_m in left lane, right_fit_m in right lane: ", leftLane.fit_m, rightLane.fit_m)
    ## need to get some default value for left lane and right lane. Otherwise, None will cause issue.
    leftLane.add_fit(left_fit)
    leftLane.add_fit_m(left_fit_m)
    rightLane.add_fit(right_fit)
    rightLane.add_fit_m(right_fit_m)
    output = drawLine(img, left_fit, right_fit)
    if needBGR2RGB:
        output = cv2.cvtColor( output, cv2.COLOR_BGR2RGB )
        print("need convert")
    # Calculate curvature
    yRange = img.shape[0] - 1
    leftCurvature = calculateCurvature(yRange, left_fit_m)
    rightCurvature = calculateCurvature(yRange, right_fit_m)
    print("actural left curv:, right curv: ",leftCurvature, rightCurvature)
    
    # Calculate vehicle center
    diffFromVehicle = calculateCenter(img, left_fit_m, right_fit_m)
    if diffFromVehicle > 0:
        message = '{:.2f} m right'.format(diffFromVehicle)
    else:
        message = '{:.2f} m left'.format(-diffFromVehicle)
    
    # Draw info
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontColor = (255, 255, 255)
    fontScale = 2
    cv2.putText(output, 'Left curvature: {:.0f} m'.format(leftCurvature), (50, 50), font, fontScale, fontColor, 2)
    cv2.putText(output, 'Right curvature: {:.0f} m'.format(rightCurvature), (50, 120), font, fontScale, fontColor, 2)
    cv2.putText(output, 'Vehicle is {} of center'.format(message), (50, 190), font, fontScale, fontColor, 2)
    return output

## sanity check for polynomials
def sanityCheck(img, left_fit, right_fit, left_fit_m, right_fit_m):
    discard = False
    xMax = img.shape[1]*xm_per_pix
    yMax = img.shape[0]*ym_per_pix
    yMid = yMax/2
    yMin = yMax/8
    yRange = img.shape[0] - 1
    try:
        lineLeftMax = left_fit_m[0]*yMax**2 + left_fit_m[1]*yMax + left_fit_m[2]
        lineRightMax = right_fit_m[0]*yMax**2 + right_fit_m[1]*yMax + right_fit_m[2]
        
        lineLeftMid = left_fit_m[0]*yMid**2 + left_fit_m[1]*yMid + left_fit_m[2]
        lineRightMid = right_fit_m[0]*yMid**2 + right_fit_m[1]*yMid + right_fit_m[2]
        
        lineLeftMin = left_fit_m[0]*yMin**2 + left_fit_m[1]*yMin + left_fit_m[2]
        lineRightMin = right_fit_m[0]*yMin**2 + right_fit_m[1]*yMin + right_fit_m[2]
        ## check width between line left and line right
        delta_width = abs((lineRightMax - lineLeftMax) - between_left_right_Line)/between_left_right_Line
        if delta_width > 0.05:
            print("start width of lane wrong")
            discard = True
        else:
            delta_width_mid = abs((lineRightMid - lineLeftMid) - (lineRightMax - lineLeftMax))/(lineRightMax - lineLeftMax)
            if delta_width_mid > 0.1:
                print("middle width of lane wrong")
                discard = True
            else:
                delta_width_min = abs((lineRightMin - lineLeftMin) - (lineRightMid - lineLeftMid))/(lineRightMid - lineLeftMid)
                if delta_width_min > 0.1:
                    print(" end width of lane wrong")
                    discard = True
                else:
                    leftCurvature = calculateCurvature(yRange, left_fit_m)
                    rightCurvature = calculateCurvature(yRange, right_fit_m)
                    if leftCurvature > 10000:
                        print("discard fail frame as left curve")
                        discard = True
                    if rightCurvature > 10000:
                        print("discard fail frame as right curve")
                        discard = True
    
    except TypeError:
        print("typeError")
        discard = True

    if discard == True:
        # use previous frame data
        left_fit = leftLane.fit
        left_fit_m = leftLane.fit_m
        right_fit = rightLane.fit
        right_fit_m = rightLane.fit_m
    else:
        # valid data and add in leftLane/rightLane prev_fit
        left_fit = leftLane.add_fit(left_fit)
        left_fit_m = leftLane.add_fit_m(left_fit_m)
        right_fit = rightLane.add_fit(right_fit)
        right_fit_m = rightLane.add_fit_m(right_fit_m)

    return left_fit, right_fit, left_fit_m, right_fit_m

## pipe line to deal with movie
def pipeline(img):
    undistort = undistortImages(img,mtx, dist)
    #hls = cv2.cvtColor(undistort, cv2.COLOR_BGR2HLS)
    combine = combineGradientsAndLUV(img,threshSobel = (70,160))
    binary_warped = adjustPerspective(combine, M)
    left_fit, right_fit, left_fit_m, right_fit_m, _, _, _, _, _ = findLines(binary_warped)
    #pdb.set_trace()
    left_fit, right_fit, left_fit_m, right_fit_m = sanityCheck(img, left_fit, right_fit, left_fit_m, right_fit_m)
    output = drawLine(img, left_fit, right_fit)
    if needBGR2RGB:
        output = cv2.cvtColor( output, cv2.COLOR_BGR2RGB )
        print("need convert")
    # Calculate curvature
    yRange = img.shape[0] - 1
    leftCurvature = calculateCurvature(yRange, left_fit_m)
    rightCurvature = calculateCurvature(yRange, right_fit_m)
    
    #print("actural left curv:, right curv: ",leftCurvature, rightCurvature)
    ## first frame set
    if leftLane.curvature is None:
        print("first frame")
        leftLane.curvature = leftCurvature
    if rightLane.curvature is None:
        print("first frame")
        rightLane.curvature = rightCurvature
    ## Compare with old_left_fit & old_right_fit to decide if update or not
    ## Also check deviation from last frame. If deviation > 5%, discard this frame.
    #leftDeviate = abs((left_fit[0] - leftLane.fit[0])/leftLane.fit])
    #rightDeviate = abs((right_fit[0] - rightLane.fit[0])/rightLane.fit[0])
    if leftCurvature > 10000:
        print("discard fail frame as left curve")
        left_fit = leftLane.fit
        left_fit_m = leftLane.fit_m
        leftCurvature = leftLane.curvature
    else:
        leftLane.fit = left_fit
        leftLane.fit_m = left_fit_m
        leftLane.curvature = leftCurvature

    if rightCurvature > 10000:
        print("discard fail frame as right curve")
        right_fit = rightLane.fit
        right_fit_m = rightLane.fit_m
        rightCurvature = rightLane.curvature
    else:
        rightLane.fit = right_fit
        rightLane.fit_m = right_fit_m
        rightLane.curvature = rightCurvature


    # Calculate vehicle center
    diffFromVehicle= calculateCenter(img, left_fit_m, right_fit_m)
    if diffFromVehicle > 0:
        message = '{:.2f} m right'.format(diffFromVehicle)
    else:
        message = '{:.2f} m left'.format(-diffFromVehicle)
    
    # Draw info
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontColor = (255, 255, 255)
    fontScale = 2
    cv2.putText(output, 'Left curvature: {:.0f} m'.format(leftCurvature), (50, 50), font, fontScale, fontColor, 2)
    cv2.putText(output, 'Right curvature: {:.0f} m'.format(rightCurvature), (50, 120), font, fontScale, fontColor, 2)
    cv2.putText(output, 'Vehicle is {} of center'.format(message), (50, 190), font, fontScale, fontColor, 2)
    
    return output

def videoClip(inputVideo, outputVideo, start, end):
    clip = VideoFileClip(inputVideo)
    clip2 = clip.subclip("0:0:" + str(start), "0:0:" + str(end))
    clip2.write_videofile(outputVideo, audio=False)

def videoPipeline(inputVideo, outputVideo):
    """
        Process the `inputVideo` frame by frame to find the lane lines, draw curvarute and vehicle position information and
        generate `outputVideo`
        """
    myclip = VideoFileClip(inputVideo)
    ## use pipeline to deal with each frame
    clip = myclip.fl_image(pipeline)
    clip.write_videofile(outputVideo, audio=False)

## test code
# Loading camera calibration
cameraCalibration = pickle.load( open('./pickled_data/camera_calibration.p', 'rb' ) )
mtx, dist = map(cameraCalibration.get, ('mtx', 'dist'))

# Load test images.
testImages = list(map(lambda imageFileName: cv2.imread(imageFileName),
                      glob.glob('./test_images/*.jpg')))
testImagesName = glob.glob('./test_images/*.jpg')
print("test image shape: ", testImages[1].shape)
image_width = testImages[1].shape[1]
image_height = testImages[1].shape[1]
undistImages = list(map(lambda img: undistortImages(img,mtx, dist),testImages))
## Finally use combination of color detection on HSV and Sobel on S channel of HLS
combineImages = list(map(lambda img: combineGradientsAndLUV(img,threshSobel = (70,160)), undistImages))
## show warped images in the test images
#showImages(combineImages, testImagesName,figTitle ='Filter on test images', cols=imageRow,rows=imageCol,cmap='gray',figName = "FilteredTestImages")
#plt.show()

## Apply perspective transform
transMatrix = pickle.load( open('./pickled_data/perspective_transform.p', 'rb' ) )
M, Minv = map(transMatrix.get, ('M', 'Minv'))

warpedImages = list(map(lambda img: adjustPerspective(img, M), combineImages))
## show warped images in the test images
#showImages(warpedImages, testImagesName,figTitle ='Perspective transform on the test images', cols=imageRow,rows=imageCol,cmap='gray',figName = "PerspectiveTestImages")
#plt.show()

## show lane-line pixels and fit their positions with a polynomial
#polyImages = showLaneOnImages(warpedImages, testImagesName, cols=imageRow,rows=imageCol)
#plt.show()
discardFrameNo = 0

processedImages = []
for i in range(len(testImagesName)):
    print("image name: ", testImagesName[i])
    processed = laneCurveProcess(warpedImages[i], testImages[i])
    processedImages.append(processed)
## show lane in the test images
#showImages(processedImages, testImagesName,figTitle ='Detected Lines on the test images', cols=imageRow,rows=imageCol,cmap='gray', figName = "LaneOnTestImages")
#plt.show()

## Use pipeline to deal with the project_video file
## no need convert BGR to RGB
needBGR2RGB = not needBGR2RGB
print("needBGR2RGB: ", needBGR2RGB)
#videoClip('project_video.mp4', 'test1.mp4', 37,43)
#videoClip('project_video.mp4', 'test2.mp4', 21,25)
#videoPipeline('test1.mp4', 'video_output/test1_out.mp4')
#videoPipeline('test2.mp4', 'video_output/test2_out.mp4')
videoPipeline('project_video.mp4', 'video_output/project_video_out.mp4')
