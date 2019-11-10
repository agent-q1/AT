import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
import cv2
import numpy as np
import cv2
import numpy as np



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

(x,y,w,h)=cv2.boundingRect(cnt)
roi = blurred[y:y+h,x:x+w]
ret,roi=cv2.threshold(roi,127,255,cv2.THRESH_BINARY_INV)
squared = makeSquare(roi)
squared = cv2.resize(squared,(20,20))
squared = cv2.copyMakeBorder(squared,4,4,4,4,cv2.BORDER_CONSTANT)




(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

class_names = ['zero', 'one', 'two', 'three', 'four',
               'five', 'six', 'seven', 'eight', 'nine']

train_images = train_images / 255.0

test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=2)
image = test_images[0]*255
image = cv2.resize(squared,(512,512))

squared = squared/255
list = []

list.append(squared)
a = np.array(list)
predictions = model.predict(a)
for i in range(1):
    pr = np.argmax(predictions[i])    
    print(pr)
cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#for i in range(0,10):
#    img1 = test_images[i]
#    img1 = cv2.resize(img1,(512,512))
#    cv2.imshow('images',img1)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

