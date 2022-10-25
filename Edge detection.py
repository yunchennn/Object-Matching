"""
@author: Yun Chen
"""
import cv2
import numpy as np
from skimage import io
from matplotlib import pyplot as plt
import math
import os
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


def gaussianKernel(size, sigma, threshold):
    outborder = int(np.ceil(size/2)) #5
    g = np.zeros(size, dtype=float)
    summ = 0
    #1-D Gaussian
    for k in range(size):
        g[k] = math.exp(-((k-outborder)**2) / (2*sigma*sigma))
        summ += g[k]
   
    # normalization (sum as 1.0)
    for k in range (size):
        g[k] = g[k]/summ
    
    # Smooth between 5th row, 5th column, (N-5)th row and (N-5)th column
    h_1 = np.zeros((M,N), float)
    for i in range(M):
        for j in range(outborder, N-outborder):
            summ = 0.0
            for k in range(size):
                summ = summ+g[k]*AG[i][j-(k-outborder)]
                # print(summ)
            h_1[i][j] = summ
    
    h_2 = np.copy(h_1)
    for j in range(N):
        for i in range(outborder, M-outborder):
            summ = 0.0
            for k in range(size):
                summ = summ+g[k]*h_1[i-(k-outborder)][j]
            h_2[i][j] = summ

    h_2gm = np.copy(h_2)
    for i in range(1, M-1):
        for j in range(1, N-1):
            Gy = h_2[i+1,j]-h_2[i,j]
            Gx = h_2[i,j+1]-h_2[i,j]
            h_2gm[i][j] = np.sqrt(np.power(Gy,2)+np.power(Gx,2))

    '''
    If the gradient magnitude is higher than the threshold at a pixel (i,j), 
    then set the value of the pixel in the output image to be 255
    '''
    h2_thres = np.zeros((M,N), dtype=float)
    for i in range(outborder, M-outborder):
        for j in range(outborder, N-outborder):
            if h_2gm[i][j] > threshold:
                h2_thres[i][j] = 255
            else:
                h2_thres[i][j] = 0
    
    plt.title("Q3_sigma={}, threshold={}".format(sigma, threshold))
    plt.imshow(h2_thres, cmap=plt.cm.gray)
    plt.show()
    # io.imsave("./output/Q3_Edge_detection_sigma_{}_threshold_{}.jpg".format(int(sigma), threshold), h2_thres)
    


if __name__ == "__main__":

    sigma = [1.0, 2.0, 3.0]
    thresholds = [20,30,40,50]

    for i in sigma:
        for j in thresholds:
            gaussianKernel(9, i, j) 
#    sigma (1.0, 2.0, and 3.0) and different thresholds (20, 30, 40, and 50)
