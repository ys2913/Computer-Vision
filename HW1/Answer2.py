import numpy as np
import cv2
import random
from PIL import Image
import scipy.spatial.distance as dist
import itertools

ITERATIONS = 100
MAXDIST = 10
MIN_MATCH_COUNT = 10
IMAGE1 = 'assignment1/scene.pgm'
IMAGE2 = 'assignment1/book.pgm'
MERGED_IMAGE = 'ass1/merged.jpg'
MATCHES_IMAGE = 'ass1/matches.jpg'
SIFT_IMG1 = 'ass1/sift_scene.pgm'
SIFT_IMG2 = 'ass1/sift_book.pgm'
SIFT_IMGM = 'ass1/matches.jpg'
IMG_AFFINE = "ass1/affine/img"
REFIT_FILE = 'ass1/finalrefit.txt'


PRINT_KEYPOINTS = True
FINAL_REFIT = True
WARP_AFFINE = True
DRAW_MATCHES = True


def finalrefit2(indexes, src_pts, dst_pts):
    mA = []
    mB = []
    for x in indexes:
        mA.append([src_pts[x][0], src_pts[x][1], 0, 0, 1, 0])
        mA.append([0, 0, src_pts[x][0], src_pts[x][1], 0, 1])
        mB.append(dst_pts[x][0])
        mB.append(dst_pts[x][1])
    mA = np.array(mA)
    mB = np.array(mB)
    m = np.linalg.lstsq(mA,mB)
    return m[0],indexes;

def finalrefit(indexes, src_pts, dst_pts, length, maxdist):
    A = 1
    maxcount = 0
    matchindex = []
    comb = itertools.combinations(indexes, 3)
    iteration = 1;
    for x in comb:
        print iteration
        iteration = iteration + 1

        count = 0
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]

        ma = np.matrix([
            [src_pts[x1][0], src_pts[x1][1], 0, 0, 1, 0], [0, 0, src_pts[x1][0], src_pts[x1][1], 0, 1],
            [src_pts[x2][0], src_pts[x2][1], 0, 0, 1, 0], [0, 0, src_pts[x2][0], src_pts[x2][1], 0, 1],
            [src_pts[x3][0], src_pts[x3][1], 0, 0, 1, 0], [0, 0, src_pts[x3][0], src_pts[x3][1], 0, 1]
        ])
        mb = np.matrix([
            [dst_pts[x1][0]], [dst_pts[x1][1]],
            [dst_pts[x2][0]], [dst_pts[x2][1]],
            [dst_pts[x3][0]], [dst_pts[x3][1]]
        ])
        det_ma = np.linalg.det(ma)
        if det_ma == 0:
            continue
        ma = np.float32(ma)
        mb = np.float32(mb)
        mx = np.linalg.solve(ma, mb)
        matches = []
        for i in range(0, length):
            ma = np.matrix([
                [src_pts[i][0], src_pts[i][1], 0, 0, 1, 0], [0, 0, src_pts[i][0], src_pts[i][1], 0, 1]])
            mp2 = np.dot(ma, mx)
            mp1 = np.matrix([[dst_pts[i][0]], [dst_pts[i][1]]])
            dist1 = dist.euclidean(mp1, mp2)

            if dist1 <= maxdist:
                count += 1
                matches.append(i)

        if count > maxcount:
            A = mx
            matchindex = matches
            maxcount = count
    return A, matchindex;


def ransac(src_pts, dst_pts, length, iterations, maxdist):
    A = 1
    matchindex = []
    maxcount = 0
    for x in range(0, iterations):
        count = 0
        rand = random.sample(range(length), 3)
        x1 = rand[0]
        x2 = rand[1]
        x3 = rand[2]

        ma = np.matrix([
            [src_pts[x1][0], src_pts[x1][1], 0, 0, 1, 0], [0, 0, src_pts[x1][0], src_pts[x1][1], 0, 1],
            [src_pts[x2][0], src_pts[x2][1], 0, 0, 1, 0], [0, 0, src_pts[x2][0], src_pts[x2][1], 0, 1],
            [src_pts[x3][0], src_pts[x3][1], 0, 0, 1, 0], [0, 0, src_pts[x3][0], src_pts[x3][1], 0, 1]
        ])
        mb = np.matrix([
            [dst_pts[x1][0]], [dst_pts[x1][1]],
            [dst_pts[x2][0]], [dst_pts[x2][1]],
            [dst_pts[x3][0]], [dst_pts[x3][1]]
        ])
        det_ma = np.linalg.det(ma)
        if det_ma == 0:
            continue
        ma = np.float32(ma)
        mb = np.float32(mb)
        mx = np.linalg.solve(ma, mb)
        matches = []
        for i in range(0, length):
            ma = np.matrix([
                [src_pts[i][0], src_pts[i][1], 0, 0, 1, 0], [0, 0, src_pts[i][0], src_pts[i][1], 0, 1]])
            mp2 = np.dot(ma, mx)
            mp1 = np.matrix([[dst_pts[i][0]], [dst_pts[i][1]]])
            dist1 = dist.euclidean(mp1, mp2)

            if dist1 <= maxdist:
                count += 1
                matches.append(i)

        if count > maxcount:
            A = mx
            matchindex = matches
            maxcount = count

    return A, matchindex;


def mergeimages(img_1, img_2):
    img1 = Image.open(img_1)
    img2 = Image.open(img_2)
    (width1, height1) = img1.size
    (width2, height2) = img2.size
    result_width = width1 + width2
    result_height = max(height1, height2)
    result = Image.new('RGB', (result_width, result_height))
    result.paste(img1, box=(0, 0))
    result.paste(img2, box=(width1, 0))
    # result.save(MERGED_IMAGE, "jpeg")
    return np.array(result), width1;


def drawMatches(SIFT_IMG1, src_pts, SIFT_IMG2, dst_pts, indexes):
    merged, width1 = mergeimages(SIFT_IMG1, SIFT_IMG2)

    for i in indexes:
        source = (src_pts[i][0], src_pts[i][1])
        dest = dst_pts[i]
        dest[0] = dest[0] + width1  # updating the width pixel of the second feature point in the merged image
        destination = (dest[0], dest[1])
        cv2.line(merged, source, destination, (255, 0, 0))  # drawing blue line of width 4 pixels
    cv2.imwrite(MATCHES_IMAGE, merged)
    return;


def findmatches(image1, image2, iterations, maxdist, iterval):
    img1 = cv2.imread(image1, 0)  # scene
    img2 = cv2.imread(image2, 0)  # book

    # SIFT detector
    sift = cv2.SIFT()
    # extracting keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if PRINT_KEYPOINTS:
        img1_f = cv2.drawKeypoints(img1, kp1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img2_f = cv2.drawKeypoints(img2, kp2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite(SIFT_IMG1, img1_f)
        cv2.imwrite(SIFT_IMG2, img2_f)

    # BFMatcher object
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # storing all the good matches
    good = []
    for m, n in matches:
        if m.distance < 0.9 * n.distance:
            good.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])

    transformation_mat, result_indexes = ransac(src_pts, dst_pts, src_pts.shape[0], iterations, maxdist)

    transformation_mat_reshape = np.array([
        transformation_mat[0], transformation_mat[1], transformation_mat[4],
        transformation_mat[2], transformation_mat[3], transformation_mat[5]
    ])
    transformation_mat_reshape = transformation_mat_reshape.reshape(2, 3)
    print transformation_mat_reshape

    if FINAL_REFIT:
        fwrite = open(REFIT_FILE, 'w')
        transformation_mat, result_indexes = finalrefit2(result_indexes, src_pts, dst_pts, src_pts.shape[0], maxdist)
        # reshaping transformation_mat into (2,3) matrix
        transformation_mat_reshape = np.array([
            transformation_mat[0], transformation_mat[1], transformation_mat[4],
            transformation_mat[2], transformation_mat[3], transformation_mat[5]
        ])

        transformation_mat_reshape = transformation_mat_reshape.reshape(2, 3)

        np.savetxt(fwrite, transformation_mat_reshape)
        fwrite.close()

    if DRAW_MATCHES:
        drawMatches(SIFT_IMG1, src_pts, SIFT_IMG2, dst_pts, result_indexes)

    #reshaping transformation_mat into (2,3) matrix
    transformation_mat_reshape = np.array([
        transformation_mat[0], transformation_mat[1], transformation_mat[4],
        transformation_mat[2], transformation_mat[3], transformation_mat[5]
    ])

    transformation_mat_reshape = transformation_mat_reshape.reshape(2, 3)

    print transformation_mat_reshape
    rows, cols = img1.shape


    if WARP_AFFINE:
        dst = cv2.warpAffine(img1, transformation_mat_reshape, (cols, rows))
        IMG_AFFINE2 = IMG_AFFINE + str(iterval) + str(WARP_AFFINE) + ".jpg"
        cv2.imwrite(IMG_AFFINE2, dst)

    return;


for iterval in range(1):
    findmatches(IMAGE1, IMAGE2, ITERATIONS, MAXDIST, iterval)

