import cv2
import numpy as np



img1 = cv2.imread("book1.jpg")
img2 = cv2.imread("book2.jpg")


orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x: x.distance)


img3 = cv2.drawMatches(
    img1,
    kp1,
    img2,
    kp2,
    matches[:],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

cv2.imshow("test",img3)
cv2.waitKey(0)
cv2.destroyAllWindows()