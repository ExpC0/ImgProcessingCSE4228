import cv2
import numpy as np



img = cv2.imread('Lab1/l',cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (512, 512))


kernel = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])
kernel_size = 3
padding_h = (kernel_size-1)//2
padding_v = (kernel_size-1)//2

bordered_img = np.zeros((img.shape[0]+padding_v*2,img.shape[1]+padding_h*2))
bordered_img = cv2.copyMakeBorder(img, top=padding_h, bottom=padding_h, left=padding_v, right=padding_v,borderType=cv2.BORDER_CONSTANT)



output_img = np.zeros((bordered_img.shape[0],bordered_img.shape[1]))

for i in range(padding_h,output_img.shape[0]-padding_h):
    for j in range(padding_v,output_img.shape[1]-padding_v):
        sum = 0
        for m in range(-padding_h,padding_h+1):
            for n in range(-padding_v,padding_v+1):
                sum = sum+bordered_img[i-m][j-n]*kernel[m+padding_h][n+padding_v]
        output_img[i][j]=sum

cv2.normalize(output_img,output_img,0,1,cv2.NORM_MINMAX)

cv2.imshow("bordered img",bordered_img)

cv2.imshow("convolved img",output_img)

cv2.waitKey(0)
# result=cv2.normalize(result,None,0,1,cv2.NORM_MINMAX)
# cv2.imshow("img after convolution",result)
# cv2.waitKey(0)
cv2.destroyAllWindows()
