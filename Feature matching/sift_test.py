import cv2
import numpy as np

sift = cv2.SIFT_create()

img1 = cv2.imread("book1.jpg")
img2 = cv2.imread("book2.jpg")

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])


img3 = cv2.drawMatchesKnn(
    img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imshow("test",img3)
cv2.waitKey(0)
cv2.destroyAllWindows()