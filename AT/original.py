import cv2
import numpy as np

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

path = 'pup0.jpg'
img1 = cv2.imread(path ,0)

blurred = cv2.GaussianBlur(img1, (5,5),0)

edged = cv2.Canny(blurred,30,150)
contours,heirarchy = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[4]
 
contours = sorted(contours, key = x_cord_contour, reverse = False)

full_number = []

for c in contours:
    (x,y,w,h)=cv2.boundingRect(c)
    roi = blurred[y:y+h,x:x+w]
    ret,roi=cv2.threshold(roi,127,255,cv2.THRESH_BINARY_INV)
    squared = makeSquare(roi)


    roi = cv2.resize(roi,(512,512))
    cv2.imshow('blurred',roi)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
