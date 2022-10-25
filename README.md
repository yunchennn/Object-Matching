# Object-Matching
use the corner detection to make the feature descriptor and make the feature matching
## Histogram Equalization

```python
  pdf = hist/img.size
  cdf = pdf.cumsum()
```
 
 <img src="https://user-images.githubusercontent.com/53567979/197894693-462ffc5a-27a7-495e-9f49-ca2fbc5ac163.png" width=20% height=20%>
 
## Image Flitering/Convolution

use two 5x5 filter to convolve the figure

<img src="https://user-images.githubusercontent.com/53567979/197897154-dff9c29b-2699-4da0-9e78-a3de24fb85fd.jpg" width=20% height=20%> <img src="https://user-images.githubusercontent.com/53567979/197897202-9c18c268-43b3-4998-987e-d8a11ea1da4e.jpg" width=20% height=20%>

## Edge detection

smooth the input image with a Gaussian filter. Then compute the gradient magnitude and threshold it to mark edge pixels.
sigma: 1.0, 2.0, 3.0
threshold: 20, 30, 40, 50

<img src="https://user-images.githubusercontent.com/53567979/197897792-b89434bd-18f2-4512-aa7e-b3a0124b1cba.png" width=50% height=50%> 


## Corner detection and local feature descriptor
#### Corner detection
```python
cornerPointMark = np.zeros((M,N), dtype=np.uint8)
for i in range(5,M-5):
  for j in range(5,N-5):
    if (R[i,j] > threshold):
      cornerPointMark[i,j] = 255
for i in range(5,M-5):
  for j in range(5,N-5):
    if(cornerPointMark[i,j] == 255):
      if (R[i,j] > R[i-1,j-1] and R[i,j] > R[i-1,j] and R[i,j] > R[i,j+1]
          and R[i,j] > R[i,j-1] and R[i,j] > R[i,j+1] and
          R[i,j] > R[i+1,j-1] and R[i,j] > R[i+1,j] and R[i,j] > R[i+1,j+1]):
                cornerPointMark[i,j] = 255
      else:
        cornerPointMark[i,j] = 0
for i in range(len(cornerPointMark)):
  for j in range(len(cornerPointMark[0])):
    if cornerPointMark[i][j] == 255:
      cv2.circle(img, (j,i), 2, (255,0,0), -1)
```

<img src="https://user-images.githubusercontent.com/53567979/197898029-64030f97-17cb-4a86-90cb-bede376f3775.png" width=20% height=20%> 

#### local feature descriptor
```python
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
```
<img src="https://user-images.githubusercontent.com/53567979/197898076-7979938e-ed2c-425c-9ce7-2233f4dc9962.png" width=70% height=70%>
<img src="https://user-images.githubusercontent.com/53567979/197898934-e19233b7-d226-4591-8f00-a9957a6dc872.png" width=70% height=70%>



## Image matching
In order to achieve image matching, we use ”CornerDetection” to compare the coordinates of the corners of the two
images and the corresponding 9x9 historgam. At first, I used while Dic1[1:8] == Dic2[1:8] then add link as the matching
condition. But found it too difficult to grab matching coordinates. So we narrow the range to 180 degrees (index = 5)
Dic14:6 == Dic24:6 as the matching condition. The result is shown in Figure 16. Through the second matching method,
we can see that there are several lines between the two images. Figure 17 is the coordinates where the two graphs
match.
```python
def match(newdic, newdic2, cornerimage1, cornerimage2):
images = [Image.open(x) for x in [cornerimage1, cornerimage2]]
widths, heights = zip(*(i.size for i in images))
total_width = sum(widths)
max_height = max(heights)
new_im = Image.new('RGB', (total_width, max_height))
x_offset = 0
for im in images:
  new_im.paste(im, (x_offset,0))
  x_offset += im.size[0]
new_im.save('./output/test_merge1.jpg')
lines = []
for k, v in newdic.items():
  for k1, v1 in newdic2.items():
    if v == v1:
      x1, y1 = k
      x2, y2 = k1
      lines.append([[x1,y1],[x2,y2]])
img3 = io.imread('./output/test_merge1.jpg')
for line in lines:
  (x1, y1), (x2, y2) = line
  cv2.line(img3, (y1,x1), (y2,x2), (255,0,0), 1)
```

<img src="https://user-images.githubusercontent.com/53567979/197899007-85c182df-7ece-4ceb-8275-d11b97d155ae.png" width=50% height=50%> <img src="https://user-images.githubusercontent.com/53567979/197899059-5d9e5403-c81b-44e2-ad0a-33857668a9de.png" width=10% height=10%>
