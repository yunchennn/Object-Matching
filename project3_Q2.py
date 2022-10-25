"""
ESE 568 Project 3
@author: Yun Chen
@ID# 114702519
Fall 2022
"""
import numpy as np
from skimage import io
import matplotlib.pyplot as plt 
import math

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

def ImageFlitering(filternumber):

    (krn0, krn1) =filternumber[1].shape
    filter = filternumber[1]

    h = np.zeros((M,N), dtype=float)


    for k in range(krn0):
            for l in range(krn1):
                for i in range(3,M-3):
                    for j in range(3,N-3):
                        h[i,j] += filter[k][l]*AG[i-(k-3)][j-(l-3)]

    newh = [[0]*(M-6) for _ in range(N-6)]  
    for i in range(3,M-3):
            for j in range(3,M-3):
                newh[i-3][j-3] = h[i][j]

    hmin, hmax = np.array(newh).min(), np.array(newh).max()
    R = np.zeros((M,N), dtype=np.uint8)
    for i in range(3,  M-3):
        for j in range(3,  N-3):
            R[i][j] = math.floor(h[i][j]-hmin)*255/(hmax-hmin)
    
    plt.title("Q2_Image_filter_{}".format(filternumber[0]))
    plt.imshow(R, cmap=plt.cm.gray)
    plt.show()
    io.imsave("./output/Q2_Image_filter_{}.jpg".format(filternumber[0]), R)
            
    


if __name__ == "__main__":

    filter1 = np.array([[0, 0, -1, 0, 0],
                        [0, -1, -2, -1, 0],
                        [-1, -2, 16, -2, -1],
                        [0, -1, -2, -1, 0],
                        [0, 0, -1, 0, 0]])

    filter2 = np.array([[0.04,0.04,0.04,0.04,0.04],
                        [0.04,0.04,0.04,0.04,0.04],
                        [0.04,0.04,0.04,0.04,0.04],
                        [0.04,0.04,0.04,0.04,0.04],
                        [0.04,0.04,0.04,0.04,0.04]])
    filter_1 = [1,filter1]
    filter_2 = [2,filter2]
    
    ImageFlitering(filter_2)

