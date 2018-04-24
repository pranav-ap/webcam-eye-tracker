import cv2
import numpy as np

img1 = cv2.imread('color palette.png')
print('img 1 shape : ', img1.shape)

img2 = cv2.imread('wall.jpg')
print('img 2 shape : ', img2.shape)

while True:
    cv2.imshow('original 1', img1)
    cv2.imshow('original 2', img2)

    cv2.imshow('addition', cv2.add(img1, img2))

    cv2.imshow('bitwise and', cv2.bitwise_and(img1, img2))
    cv2.imshow('bitwise or', cv2.bitwise_or(img1, img2))
    cv2.imshow('bitwise xor', cv2.bitwise_xor(img1, img2))
    cv2.imshow('bitwise not', cv2.bitwise_not(img1, img2))

    # check for quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()