import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
def x_cord_contour(contour):
    area = cv2.contourArea(contour)

    if area != None:

        if (int(area)> 10.0):
          
            M=cv2.moments(contour)
            return (int(M['m10']/M['m00']))

def makeSquare(not_square):
    BLACK = [0,0,0]
    img_dim = not_square.shape
    height = img_dim[0]
    width = img_dim[1]
    if(height == width):
        square = not_square
        return square
    else:
        doublesize = cv2.resize(not_square, (2*width,2*height), interpolation = cv2.INTER_CUBIC)
        height = height*2
        width = width*2
        if(height>width):
            pad = (height-width)/2
            doublesize_square = cv2.copyMakeBorder(doublesize,0,0,int(pad),int(pad),cv2.BORDER_CONSTANT, value=BLACK)
        else:
            pad = (width-height)/2
            doublesize_square=cv2.copyMakeBorder(doublesize,int(pad),int(pad),0,0,cv2.BORDER_CONSTANT, value=BLACK)
    return doublesize_square

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
path = 'pup0.jpg'
img1 = cv2.imread(path ,0)

blurred = cv2.GaussianBlur(img1, (5,5),0)

edged = cv2.Canny(blurred,30,150)
contours,heirarchy = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[4]

(x,y,w,h)=cv2.boundingRect(cnt)
roi = blurred[y:y+h,x:x+w]
ret,roi=cv2.threshold(roi,127,255,cv2.THRESH_BINARY_INV)
squared = makeSquare(roi)
squared = cv2.resize(squared,(28,28))

output = cv2.resize(squared,(512,512))
cv2.imshow('blurred',output)
cv2.waitKey(0)
cv2.destroyAllWindows()
img1 = test_images[0]
resized = cv2.resize(img1, (512,512), interpolation = cv2.INTER_AREA)
cv2.imshow('images',resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
