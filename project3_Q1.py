"""
ESE 568 Project 3
@author: Yun Chen
@ID# 114702519
Fall 2022
"""
import numpy as np
from skimage import io
import matplotlib.pyplot as plt 
import cv2

img = io.imread('101.jpg')
# img = cv2.resize(img, dsize=(256,256), interpolation=cv2.INTER_CUBIC)
(M,N,o) = img.shape
redChannel = img[:,:,0] 
greenChannel = img[:,:,1] 
blueChannel = img[:,:,2] 
allBlack = np.zeros((M, N), dtype=np.uint8)
justRed = np.stack((redChannel, allBlack, allBlack), axis=2)
justGreen = np.stack((allBlack, greenChannel, allBlack),axis=2)
justBlue = np.stack((allBlack, allBlack, blueChannel),axis=2)

AG = justRed[:,:,0]/3 + justGreen[:,:,1]/3 + justBlue[:,:,2]/3

# Q1 Histogram Equalization
hist=np.zeros(256)              
for i in range(M) :
    for j in range(N) :
        hist[img[i,j,0]] += 1

pdf=np.zeros(256)
for k in range(256):
    pdf[k] = hist[k]/(M*N)
    
cdf=np.zeros(256)
for k in range(256):
    for i in range(k):
        cdf[k] += pdf[i]

h_trans = np.zeros((M, N), dtype=np.uint8)
for i in range(M) :
    for j in range(N) :
        h_trans[i,j] = 255*cdf[int(AG[i,j])]

plt.title("Q1_Histogram Equalization")
plt.imshow(h_trans, cmap=plt.cm.gray)
plt.show()
io.imsave('./output/Q1_Histogram Equalization.jpg' , h_trans)
