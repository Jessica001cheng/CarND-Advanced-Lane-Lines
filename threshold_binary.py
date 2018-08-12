import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
from helper import showImages, showSidebySide
from camera_calibrate import undistortImages


# Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def sThresh(s, thresh=(0, 255)):
    mask = (s >= thresh[0]) & (s <= thresh[1])
    # 3) Return a binary image of threshold result
    binary_output = np.zeros_like(s)
    binary_output[mask] = 1
    return binary_output

def absSobelThresh(img, orient = "x", sobel_kernel=3, thresh = (10,100)):
    """
        Calculate the Sobel gradient on the direction `orient` and return a binary thresholded image
        on [`thresh_min`, `thresh_max`]. Using `sobel_kernel` as Sobel kernel size.
    """
    if orient == 'x':
        yorder = 0
        xorder = 1
    else:
        yorder = 1
        xorder = 0
    sobel = cv2.Sobel(img, cv2.CV_64F, xorder, yorder, ksize=sobel_kernel) # Take the derivative in x
    abs_sobel = np.absolute(sobel) # Absolute  derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    # Threshold x gradient
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return sbinary

def magThresh(img, sobel_kernel=3, thresh = (10,100)):
    """
        Calculate the Sobel magnitude gradient and return a binary thresholded image
        on [`thresh_min`, `thresh_max`]. Using `sobel_kernel` as Sobel kernel size.
    """
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    binary = np.zeros_like(img)
    binary[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1
    return binary

def dirThresh(img, sobel_kernel=3, thresh = (10,100)):
    """
        Calculate the Sobel direction gradient and return a binary thresholded image
        on [`thresh_min`, `thresh_max`]. Using `sobel_kernel` as Sobel kernel size.
    """
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary = np.zeros_like(img)
    binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary

def combineGradients(img,thresh = (10,100)):
    """
        Compute the combination of Sobel X and Sobel Y or Magnitude and Direction
        """
    s=img[:,:,2]
    l=img[:,:,1]
    sobelX = absSobelThresh(l, thresh=thresh)
    sobelY = absSobelThresh(l, thresh=thresh, orient = "y")
    sthresh = sThresh(s, thresh=thresh)
    combined = np.zeros_like(sobelX)
    combined[((sobelX == 1) & (sobelY == 1)) | (s == 1)] = 1
    return combined

def combineGradientsOnS(img,thresh = (10,100)):
    """
        Compute the combination of Sobel X and Sobel Y or Magnitude and Direction
        """
    s=img[:,:,2]
    sobelX = absSobelThresh(s, thresh=thresh)
    sobelY = absSobelThresh(s, thresh=thresh, orient = "y")
    sthresh = sThresh(s, thresh=thresh)
    combined = np.zeros_like(sobelX)
    combined[((sobelX == 1) & (sobelY == 1)) | (s == 1)] = 1
    return combined

if __name__ == '__main__':
    cameraCalibration = pickle.load( open('./pickled_data/camera_calibration.p', 'rb' ) )
    mtx, dist = map(cameraCalibration.get, ('mtx', 'dist'))
    # Load test images.
    testImages = list(map(lambda imageFileName: cv2.imread(imageFileName),
                          glob.glob('./test_images/*.jpg')))
    testImagesName = glob.glob('./test_images/*.jpg')
    #print("testImages name: ", testImagesName)
    undistImages = list(map(lambda img: undistortImages(img,mtx, dist),testImages))
    # Convert to HLS color space and
    s = list(map(lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:,:,2],undistImages))
    l = list(map(lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:,:,1],undistImages))
    sChannel = list(map(lambda img: sThresh(img, thresh=(100, 255)),s))
    showImages(sChannel, testImagesName,figTitle='S channel Threshold', cols=2,rows=4,cmap='gray',
               figName = "SChanelThres")
    ## Sobel X on L channel
    withSobelX = list(map(lambda img: absSobelThresh(img, thresh=(10, 160)), s))
    showImages(withSobelX, testImagesName,figTitle ='Sobel X on S channel', cols=2,rows=4,cmap='gray',
               figName = "SobelXThres")
    plt.show()
    ## Sobel Y on L channel
    withSobelY = list(map(lambda img: absSobelThresh(img, thresh=(10, 160), orient = "y"), s))
    showImages(withSobelY, testImagesName,figTitle ='Sobel Y on S channel', cols=2,rows=4,cmap='gray',
               figName = "SobelYThres")
    plt.show()
    ## Sobel Magnitude
    magnitude = list(map(lambda img: magThresh(img, thresh=(10, 160)), s))
    showImages(magnitude, testImagesName,figTitle ='Magnitude on S channel', cols=2,rows=4,cmap='gray')
    plt.show()
    ## Sobel Direction gradient
    direction = list(map(lambda img: dirThresh(img, thresh=(0.79,1.20)), s))
    showImages(direction, testImagesName,figTitle ='Direction gradient on S channel', cols=2,rows=4,cmap='gray')
    plt.show()
    ## Combine Sobel X,y and S channel threshold
    # Convert to HLS color space
    hls = list(map(lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2HLS),undistImages))
    combine = list(map(lambda img: combineGradients(img, thresh=(10, 160)), hls))
    showImages(combine, testImagesName,figTitle ='Combine Sobel on L channel and S threshold', cols=2,rows=4,cmap='gray')
    plt.show()
    combineS = list(map(lambda img: combineGradientsOnS(img, thresh=(10, 160)), hls))
    showImages(combine, testImagesName,figTitle ='Combine Sobel on S channel and S threshold', cols=2,rows=4,cmap='gray',
               figName = "CombineS_SobelThres_Schannel")
    plt.show()
