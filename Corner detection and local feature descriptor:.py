"""
@author: Yun Chen
"""
import numpy as np
from skimage import io
import matplotlib.pyplot as plt 
import cv2
from scipy import signal, ndimage
import math
from PIL import Image
import os

def CornerDetection(sub, size, sigma, threshold, wind_size, img1):
    img = io.imread(img1)
    (M,N) = img.shape
    outborder = int(np.floor(size/2))
    g = np.zeros(size, dtype=float)
    summ=0 
    for k in range (size):
        g[k] =  np.exp(-((k-outborder)**2) / (2*sigma*sigma) )
        summ += g[k]   
    for k in range (size):
        g[k] =  g[k]/summ
        
    # Smooth between 5th row, 5th column, (N-5)th row and (N-5)th column
    h_1 = np.zeros((M,N), float)
    for i in range (M):
        for j in range(outborder,(N-outborder)):
            summ = 0.0
            for k in range(size):
                summ += g[k]* img[i][j+(k-outborder)]
            h_1[i][j] = summ

    h_2 = np.copy(h_1) 
    for j in range (N):
        for i in range(outborder, M-outborder):
            summ = 0.0
            for k in range(size):
                summ += g[k]* h_1[i+(k-outborder),j]
            h_2[i][j] = summ

    Iy, Ix= np.copy(h_2), np.copy(h_2)
    A, B, C = np.zeros((M,N), float),np.zeros((M,N), float), np.zeros((M,N), float)

    for i in range(1, M-1):
        for j in range(1, N-1):
            Iy[i][j] = (h_2[i+1][j]-h_2[i][j])/10.0
            Ix[i][j] = (h_2[i][j+1]-h_2[i][j])/10.0
   
    A = Ix**2
    B = Iy**2
    C = Ix*Iy

    outborder_2 = int(np.ceil(wind_size/2))
    g = np.zeros(wind_size, dtype=float)
    sigma_2 = float(wind_size)/2.0
    summ=0
    #1-D Gaussian
    for k in range (wind_size):
        g[k] =  np.exp( -((k-outborder_2)**2) / (2*sigma_2*sigma_2) )
        summ += g[k]
    # normalization (sum as 1.0)
    for k in range (wind_size):
        g[k] =  g[k]/summ

    # for A
    h_1 = np.copy(A)
    for i in range (M):
        for j in range(outborder_2,(N-outborder_2)):
            summ = 0.0
            for k in range(wind_size):
                summ += g[k]* A[i,j+(k-outborder_2)]
            h_1[i][j] = summ

    h_2 = np.copy(h_1) 
    for j in range (N):
        for i in range(outborder_2, M-outborder_2):
            summ = 0.0
            for k in range(wind_size):
                summ += g[k]* h_1[i+(k-outborder_2),j]
            h_2[i][j] = summ

    Agm = np.copy(h_2)

    # for B
    h_1 = np.copy(B)
    for i in range (M):
        for j in range(outborder_2,(N-outborder_2)):
            summ = 0.0
            for k in range(wind_size):
                summ += g[k]* B[i,j+(k-outborder_2)]
            h_1[i][j] = summ

    h_2 = np.copy(h_1) 
    for j in range (N):
        for i in range(outborder_2, M-outborder_2):
            summ = 0.0
            for k in range(wind_size):
                summ += g[k]* h_1[i+(k-outborder_2),j]
            h_2[i][j] = summ

    Bgm = np.copy(h_2)

    # for C
    h_1 = np.copy(C)
    for i in range (M):
        for j in range(outborder_2,(N-outborder_2)):
            summ = 0.0
            for k in range(wind_size):
                summ += g[k]* C[i,j+(k-outborder_2)]
            h_1[i][j] = summ

    h_2 = np.copy(h_1)
    for j in range (N):
        for i in range(outborder_2, M-outborder_2):
            summ = 0.0
            for k in range(wind_size):
                summ += g[k]* h_1[i+(k-outborder_2),j]
            h_2[i][j] = summ

    Cgm = np.copy(h_2)
    # 8. Non-maxima suppression
    R = np.zeros((M,N), float)
    for i in range (M):
        for j in range (N):
            new_M = np.matrix([[Agm[i][j], Cgm[i][j]], [Cgm[i][j], Bgm[i][j]]])
            det = np.linalg.det(new_M)
            tra = np.trace(new_M)
            R[i][j] = det-0.04* tra**2
    
    cornerPointMark = np.zeros((M,N), dtype=np.uint8)
    for i in range(5,M-5):
        for j in range(5,N-5):
            if (R[i,j] > threshold):
                cornerPointMark[i,j] = 255

    for i in range(5,M-5):
        for j in range(5,N-5):
            if(cornerPointMark[i,j] == 255):
                if (R[i,j] > R[i-1,j-1] and R[i,j] > R[i-1,j] and R[i,j] > R[i,j+1] and 
                R[i,j] > R[i,j-1] and R[i,j] > R[i,j+1] and 
                R[i,j] > R[i+1,j-1] and R[i,j] > R[i+1,j] and R[i,j] > R[i+1,j+1]):
                    cornerPointMark[i,j] = 255
                else:
                    cornerPointMark[i,j] = 0

    for i in range(len(cornerPointMark)):
        for j in range(len(cornerPointMark[0])):
            if cornerPointMark[i][j] == 255:
                cv2.circle(img, (j,i), 2, (255,0,0), -1)

    plt.title("Corner detection")
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()
    io.imsave('./output/Q4_Corner_detection.jpg', img) 

    """
    part B
    """
    his = {}
    for m in range(5,M-5):
        for n in range(5,N-5):
            if(cornerPointMark[m,n] == 255):
                his[(m,n)] = 0

                gradientDir = np.zeros((sub,sub), float)
                for i in range(sub):
                    for j in range(sub):
                        gradientDir[i,j] = np.arctan2(Iy[m+i-4,n+j-4],Ix[m+i-4,n+j-4])*180.0/np.pi 
                        if(gradientDir[i,j] < 0.0):
                            gradientDir[i,j] = 360.0 + gradientDir[i,j]

                gradientDirQuantized = np.zeros((sub,sub), dtype=np.int8)
                for i in range(sub):
                    for j in range(sub):
                        gradientDirQuantized[i,j] = int(gradientDir[i,j]/45.0)
                his[(m,n)] = gradientDirQuantized

    new_hist = {}
    for k, v in his.items(): 
        h = {} # for oringal historgam
        hn = {}   # for normailied historgam
        for i in range(0,8):
            h[i] = 0
            hn[i] = 0
        
        for i in range(sub):
            for j in range(sub):
                
                if v[i][j] in h:
                    h[v[i][j]] += 1
        his_list = [0]*sub
        for k1, v in h.items():
            his_list[k1] = v

        max_value = max(his_list)
        max_index = his_list.index(max_value)
        hn_list = [0]*sub
        for i in range(sub):
            hn_list[i] = his_list[(max_index-4+i)%8]
        
        for i, number in enumerate(hn_list):
            if i in hn:
                hn[i] = number

        new_hist[k] = hn_list
    
        plt.subplot(1,2,1)
        plt.title('original histogram for {}'.format(k))
        plt.bar(list(h.keys()), h.values(), color='r')
        plt.subplot(1,2,2)
        plt.title('normalized histogram for {}'.format(k))
        plt.bar(list(hn.keys()), hn.values(), color='b')
        plt.show()
    print('newdic:', new_hist)
if __name__ == "__main__":
    img1 = 'pic1grey300.jpg'
    img2 = 'pic2grey300.jpg'

    CornerDetection(9, 9, 2.0, 0.09, 11, img2)
    

 
