import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
from helper import showImages, showSidebySide

objpoints = []
imgpoints = []
outimages = []
originalImages = []


def findCorner(calibrationImages):
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) # x and y cordinates.

    for imageAndFile in calibrationImages:
        fileName, image = imageAndFile
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
            
            img_points = cv2.drawChessboardCorners(image.copy(), (9,6), corners, ret)
            outimages.append(img_points)
            originalImages.append(image)
    return outimages,originalImages

def undistortImages(image, mtx, dist):
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    return undist

# Load calibration images.
if __name__ == '__main__':
    calibrationImages = list(map(lambda imageFileName: (imageFileName, cv2.imread(imageFileName)), glob.glob('./camera_cal/c*.jpg')))
    calibrationImagesName = glob.glob('./camera_cal/c*.jpg')
    print("test image number: ", len(calibrationImages))
    #showImages(calibrationImages, calibrationImagesName, 4, 5, (15, 13))
    outimages, originalImages = findCorner(calibrationImages)
    ## calibration and undistort
    index = 5
    original = originalImages[index]
    print("gray shape: ", original.shape[0:2])
    chessPoint = outimages[index]
    #showSidebySide(original, chessPoint, "original", "with points")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, original.shape[0:2], None, None)
    undist = undistortImages(original, mtx, dist)
    showSidebySide(original, undist, "original", "undistort")
    ## save the calibration parameters
    pickle.dump( { 'mtx': mtx, 'dist': dist }, open('./pickled_data/camera_calibration.p', 'wb'))
    testImages = list(map(lambda imageFileName: cv2.imread(imageFileName),
                          glob.glob('./test_images/*.jpg')))
    testImagesName = glob.glob('./test_images/*.jpg')
    testIndex = 1
    testImage = testImages[testIndex]
    testName = testImagesName[testIndex]
    undist = undistortImages(testImage, mtx, dist)
    ## need to convert to RGB for matplot show
    undist = cv2.cvtColor(undist, cv2.COLOR_BGR2RGB )
    showSidebySide(cv2.cvtColor(testImage, cv2.COLOR_BGR2RGB ), undist, testName, "testImagedistort")

