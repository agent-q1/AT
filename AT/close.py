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
train_labels = []
train_images=[]
test_labels = []
test_images=[]
for i in range(10,55):
    y = str(i)
    path = 'Sample030/img030-0'+y+'.png'
    img1 = cv2.imread(path ,0)
    blurred = cv2.GaussianBlur(img1, (5,5),0)
    edged = cv2.Canny(blurred,30,150)
    contours,heirarchy = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]

    (x,y,w,h)=cv2.boundingRect(cnt)
    roi = blurred[y:y+h,x:x+w]
    ret,roi=cv2.threshold(roi,127,255,cv2.THRESH_BINARY_INV)
    squared = makeSquare(roi)
    squared = cv2.resize(squared,(20,20))
    squared = cv2.copyMakeBorder(squared,4,4,4,4,cv2.BORDER_CONSTANT)
    squared = squared/255
    train_images.append(squared)
    train_labels.append(0)
for i in range(1,10):
    y = str(i)
    path = 'Sample030/img030-00'+y+'.png'
    img1 = cv2.imread(path ,0)
    blurred = cv2.GaussianBlur(img1, (5,5),0)
    edged = cv2.Canny(blurred,30,150)
    contours,heirarchy = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]

    (x,y,w,h)=cv2.boundingRect(cnt)
    roi = blurred[y:y+h,x:x+w]
    ret,roi=cv2.threshold(roi,127,255,cv2.THRESH_BINARY_INV)
    squared = makeSquare(roi)
    squared = cv2.resize(squared,(20,20))
    squared = cv2.copyMakeBorder(squared,4,4,4,4,cv2.BORDER_CONSTANT)
    squared = squared/255
    test_images.append(squared)
    test_labels.append(0)


for i in range(1,10):
    y = str(i)
    path = 'Sample016/img016-00'+y+'.png'
    img1 = cv2.imread(path ,0)
    blurred = cv2.GaussianBlur(img1, (5,5),0)
    edged = cv2.Canny(blurred,30,150)
    contours,heirarchy = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]

    (x,y,w,h)=cv2.boundingRect(cnt)
    roi = blurred[y:y+h,x:x+w]
    ret,roi=cv2.threshold(roi,127,255,cv2.THRESH_BINARY_INV)
    squared = makeSquare(roi)
    squared = cv2.resize(squared,(20,20))
    squared = cv2.copyMakeBorder(squared,4,4,4,4,cv2.BORDER_CONSTANT)
    squared = squared/255
    test_images.append(squared)
    test_labels.append(1)


for i in range(10,55):
    y = str(i)
    path = 'Sample016/img016-0'+y+'.png'
    img1 = cv2.imread(path ,0)
    blurred = cv2.GaussianBlur(img1, (5,5),0)
    edged = cv2.Canny(blurred,30,150)
    contours,heirarchy = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]

    (x,y,w,h)=cv2.boundingRect(cnt)
    roi = blurred[y:y+h,x:x+w]
    ret,roi=cv2.threshold(roi,127,255,cv2.THRESH_BINARY_INV)
    squared = makeSquare(roi)
    squared = cv2.resize(squared,(20,20))
    squared = cv2.copyMakeBorder(squared,4,4,4,4,cv2.BORDER_CONSTANT)
    squared = squared/255
    train_images.append(squared)
    train_labels.append(1)



a = np.array(train_images)




#(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = np.array(train_images)
train_labels = np.array(train_labels)

test_images = np.array(test_images)
test_labels = np.array(test_labels)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)

path = 'sampleT5.jpg'
img1 = cv2.imread(path ,0)

blurred = cv2.GaussianBlur(img1, (5,5),0)

edged = cv2.Canny(blurred,30,150)
contours,heirarchy = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if(len(contours)!=0):
    cnt = contours[len(contours)-1]

    (x,y,w,h)=cv2.boundingRect(cnt)
    roi = blurred[y:y+h,x:x+w]
    ret,roi=cv2.threshold(roi,127,255,cv2.THRESH_BINARY_INV)
else:
    roi = blurred
    ret,roi=cv2.threshold(roi,127,255,cv2.THRESH_BINARY_INV)
print('lenght is ',len(contours))
squared = makeSquare(roi)
squared = cv2.resize(squared,(20,20))
squared = cv2.copyMakeBorder(squared,4,4,4,4,cv2.BORDER_CONSTANT)
img1 = cv2.resize(squared,(512,512))
cv2.imshow('images',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

list = []
squared=squared/255
list.append(squared)
a = np.array(list)
predictions = model.predict(a)
for i in range(1):
    pr = np.argmax(predictions[i])    
    print(pr)

#for i in range(0,10):
#    img1 = test_images[i]
#    img1 = cv2.resize(img1,(512,512))
#    cv2.imshow('images',img1)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

