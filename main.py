import numpy as np
import pandas as pd
import os
import csv
import cv2
from matplotlib import pyplot as plt

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score

img = cv2.imread('cot.jpg',0)

# Initiate SIFT detector
sift = cv2.SIFT_create(nfeatures=1000)

# find the keypoints with SIFT
keypoints = sift.detect(img,None)

# compute the descriptors with SIFT
keypoints, descriptors = sift.compute(img, keypoints)

# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(img,keypoints,descriptors, color=(0,255,0), flags=0, )
plt.imshow(img2),plt.show()

print ("No of keypoints :" , len(keypoints))
print ("length of descriptor:" , len(descriptors))
print ("length of descripbed keypoint:" , len(descriptors[1]))
print ("Elements stored in descriptor" , descriptors[0])

img = cv2.imread('cot2.jpg',0)

# Initiate SIFT detector
sift = cv2.SIFT_create(nfeatures=1000)

# find the keypoints with SIFT
keypoints = sift.detect(img,None)

# compute the descriptors with SIFT
keypoints, descriptors = sift.compute(img, keypoints)

# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(img,keypoints,descriptors, color=(0,255,0), flags=0, )
plt.imshow(img2),plt.show()

print ("No of keypoints :" , len(keypoints))
print ("length of descriptor:" , len(descriptors))
print ("length of descripbed keypoint:" , len(descriptors[1]))
print ("Elements stored in descriptor" , descriptors[0])



img = cv2.imread('cot.jpg',0)

# Initiate ORB detector
orb = cv2.ORB_create(nfeatures=1000)

# find the keypoints with ORB
keypoints = orb.detect(img,None)

# compute the descriptors with ORB
keypoints, descriptors = orb.compute(img, keypoints)

# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(img,keypoints,descriptors, color=(0,255,0), flags=0, )
plt.imshow(img2),plt.show()

print ("No of keypoints :" , len(keypoints))
print ("length of descriptor:" , len(descriptors))
print ("length of descripbed keypoint:" , len(descriptors[1]))
print ("Elements stored in descriptor" , descriptors[0])


img = cv2.imread('cot2.jpg',0)

# Initiate ORB detector
orb = cv2.ORB_create(nfeatures=1000)

# find the keypoints with ORB
keypoints = orb.detect(img,None)

# compute the descriptors with ORB
keypoints, descriptors = orb.compute(img, keypoints)

# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(img,keypoints,descriptors, color=(0,255,0), flags=0, )
plt.imshow(img2),plt.show()

print ("No of keypoints :" , len(keypoints))
print ("length of descriptor:" , len(descriptors))
print ("length of descripbed keypoint:" , len(descriptors[1]))
print ("Elements stored in descriptor" , descriptors[0])


from skimage.feature import daisy

img = cv2.imread('./cot2.jpg',0)
descs, descs_img = daisy(img, step=180, radius=58, rings=10, histograms=6, orientations=8, visualize=True)

fig, ax = plt.subplots()
ax.axis('off')
ax.imshow(descs_img)
descs_num = descs.shape[0] * descs.shape[1]
ax.set_title('%i DAISY descriptors extracted:' % descs_num)
plt.show()

from skimage.feature import daisy

img = cv2.imread('./cot2.jpg',0)
descs, descs_img = daisy(img, step=180, radius=20, rings=20, histograms=6, orientations=8, visualize=True)

fig, ax = plt.subplots()
ax.axis('off')
ax.imshow(descs_img)
descs_num = descs.shape[0] * descs.shape[1]
ax.set_title('%i DAISY descriptors extracted:' % descs_num)
plt.show()



