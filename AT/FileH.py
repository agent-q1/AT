import cv2
f=open("c.txt", "r")
contents = f.read()
lista = []
for x in contents:
    if(x!=' '):
        lista.append(x)
print(lista)

img = cv2.imread("pup0.jpg")
print(img.shape)
crop_img = img[100:600, 0:400]
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)
