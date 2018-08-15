import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
from helper import showImages, showSidebySide
from camera_calibrate import undistortImages

cameraCalibration = pickle.load( open('./pickled_data/camera_calibration.p', 'rb' ) )
mtx, dist = map(cameraCalibration.get, ('mtx', 'dist'))
## Read test image
testImages = list(map(lambda imageFileName: cv2.imread(imageFileName),
                      glob.glob('./test_images/st*.jpg')))
testImagesName = glob.glob('./test_images/st*.jpg')
print("test images num:", len(testImages))
index = 1
print("test images name:", testImagesName[index])
## Convert to RGB image
## test4.img to test
#testImage = cv2.imread('./test_images/test4.jpg')

original = cv2.cvtColor(testImages[index],cv2.COLOR_BGR2RGB)
undist = cv2.undistort(original, mtx, dist, None, mtx)

xSize, ySize, _ = undist.shape
copy = undist.copy()

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

color = [255, 0, 0]
w = 2
cv2.line(copy, left1, left2, color, w)
cv2.line(copy, left2, right1, color, w)
cv2.line(copy, right1, right2, color, w)
cv2.line(copy, right2, left1, color, w)
showSidebySide(undist, copy, "original", "source_line_drawed")

gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
src = np.float32([
                  [left2_x, left2_y],
                  [right1_x, right1_y],
                  [right2_x, right2_y],
                  [left1_x, left1_y]
                  ])
nX = gray.shape[1]
nY = gray.shape[0]
img_size = (nX, nY)
offset = 200
dst = np.float32([
                  [offset, 0],
                  [img_size[0]-offset, 0],
                  [img_size[0]-offset, img_size[1]],
                  [offset, img_size[1]]
                  ])
img_size = (gray.shape[1], gray.shape[0])
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)
warped = cv2.warpPerspective(undist, M, img_size)
showSidebySide(undist, warped, "original", "Perspective_transformed")

#pickle.dump( { 'M': M, 'Minv': Minv }, open('./pickled_data/perspective_transform.p', 'wb'))
print(M)
print(Minv)

