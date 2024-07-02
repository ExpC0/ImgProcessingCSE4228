import matplotlib.pyplot as plt
import numpy as np
import cv2

filename = "Lab1/Lena.jpg"
img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)


# adding some pixel values and normalizing
print(img)
cv2.imshow('loaded img',img)
cv2.waitKey(0)
out = np.zeros((img.shape[0],img.shape[1]))


for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        out[i][j]=img[i][j]+100

cv2.imshow('added 100 to pixel value',out)

cv2.normalize(out,out,0,255,cv2.NORM_MINMAX)
out = np.round(out).astype(np.uint8)

cv2.imshow('normalized',out)
cv2.waitKey(0)

cv2.imshow("input img",img)
cv2.waitKey(0)


cv2.destroyAllWindows()

