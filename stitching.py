from sys import flags
import cv2
from cv2 import INTER_LINEAR
from cv2 import INTER_NEAREST
import numpy as np

FPS_REDUCTION_FACTOR = 5
MIN_MATCHES = 8
ITER_FOR_ADDING_CONTRAST = 60


def get_data(img1, img2, feature_ext_method):
    if feature_ext_method == "sift":
        feature_ext = cv2.SIFT_create()
    if feature_ext_method == "surf":
        feature_ext = cv2.SURF_create()
    if feature_ext_method == "orb":
        feature_ext = cv2.ORB_create()

    k1, d1 = feature_ext.detectAndCompute(img1, None)
    k2, d2 = feature_ext.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(d1, d2, k=2)

    verify_ratio = 0.8
    verified_matches = []
    for m1, m2 in matches:
        if m1.distance < 0.8 * m2.distance:
            verified_matches.append(m1)

    if len(verified_matches) > MIN_MATCHES:
        img1_pts = []
        img2_pts = []

        for match in verified_matches:
            img1_pts.append(k1[match.queryIdx].pt)
            img2_pts.append(k2[match.trainIdx].pt)
        img1_pts = np.float32(img1_pts).reshape(-1, 1, 2)
        img2_pts = np.float32(img2_pts).reshape(-1, 1, 2)

        M, _ = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)
        return M, k1, k2, verified_matches
    else:
        print("Error: Not enough matches")
        exit()


def show_data(img1, img2, feature_ext_method="surf"):
    _, k1, k2, matches = get_data(img1, img2, feature_ext_method)
    img_3 = cv2.drawMatches(img1, k1, img2, k2, matches[:200], img2, flags=2)
    return img_3


def stitch(img1, img2, feature_ext_method="sift"):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    M, _, _, _ = get_data(gray2, gray1, feature_ext_method)

    w1, h1 = img1.shape[:2]
    w2, h2 = img2.shape[:2]

    img1_dims = np.float32([[0, 0], [0, w1], [h1, w1], [h1, 0]]).reshape(-1, 1, 2)
    img2_dims_temp = np.float32([[0, 0], [0, w2], [h2, w2], [h2, 0]]).reshape(-1, 1, 2)

    img2_dims = cv2.perspectiveTransform(img2_dims_temp, M)

    result_dims = np.concatenate((img1_dims, img2_dims), axis=0)

    [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)

    transform_dist = [-x_min, -y_min]
    transform_array = np.array(
        [[1, 0, transform_dist[0]], [0, 1, transform_dist[1]], [0, 0, 1]]
    )

    result_img = cv2.warpPerspective(
        img2, transform_array.dot(M), (x_max - x_min, y_max - y_min)
    )
    result_img[
        transform_dist[1] : w1 + transform_dist[1],
        transform_dist[0] : h1 + transform_dist[0],
    ] = img1
    return result_img


def stitch_vid(cap):
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if ret == False:
            break
        if i == 0:
            res = frame
        if i % FPS_REDUCTION_FACTOR == 0:
            img = frame
            res = stitch(img, res)
        if i == 101:
            break
        # print(i)
        # if i%ITER_FOR_ADDING_CONTRAST == 0:
        # 	sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])*(1.1)
        # res = cv2.filter2D(res, -1, sharpen_kernel)

        i += 1
    cap.release()
    return res


# img2 = cv2.imread("frames\img0.jpg")
# img1 = cv2.imread("frames\img20.jpg")
# cv2.imwrite("result.jpg", stitch(img1, img2))
# cv2.imwrite("data.jpg", show_data(img1, img2))
# for i in range(10,121,5):
# 	img2 = cv2.imread('result.jpg')
# 	img1 = cv2.imread('frames\img' + str(i) + '.jpg')
# 	cv2.imwrite('result.jpg',stitch(img1,img2))
# 	print(i)

cap = cv2.VideoCapture("test.mp4")
res = stitch_vid(cap)
cv2.imwrite("test.jpg", res)
