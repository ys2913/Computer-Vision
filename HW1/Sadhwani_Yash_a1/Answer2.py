import numpy as np
import cv2
import random
from PIL import Image
import scipy.spatial.distance as dist


def finalrefit(indexes, src_pts, dst_pts):
    mA = []
    mB = []
    for x in indexes:
        mA.append([src_pts[x][0], src_pts[x][1], 0, 0, 1, 0])
        mA.append([0, 0, src_pts[x][0], src_pts[x][1], 0, 1])
        mB.append(dst_pts[x][0])
        mB.append(dst_pts[x][1])
    mA = np.array(mA)
    mB = np.array(mB)
    # for taking into consideration all the inliers we take the leastsquare error method
    m = np.linalg.lstsq(mA, mB)
    return m[0];


# implements RANSAC and returns transformation matrix and the inlier indexes
def ransac(src_pts, dst_pts, length, iterations, maxdist):
    # initializing transformation matrix(A)
    A = 1
    # for storing inliers indexes
    matchindex = []
    maxcount = 0

    for x in range(0, iterations):
        count = 0
        # randomly taking 3 indexes from good matches
        rand = random.sample(range(length), 3)
        x1 = rand[0]
        x2 = rand[1]
        x3 = rand[2]

        # constructing matrix A
        ma = np.matrix([
            [src_pts[x1][0], src_pts[x1][1], 0, 0, 1, 0], [0, 0, src_pts[x1][0], src_pts[x1][1], 0, 1],
            [src_pts[x2][0], src_pts[x2][1], 0, 0, 1, 0], [0, 0, src_pts[x2][0], src_pts[x2][1], 0, 1],
            [src_pts[x3][0], src_pts[x3][1], 0, 0, 1, 0], [0, 0, src_pts[x3][0], src_pts[x3][1], 0, 1]
        ])
        # constructing matrix b
        mb = np.matrix([
            [dst_pts[x1][0]], [dst_pts[x1][1]],
            [dst_pts[x2][0]], [dst_pts[x2][1]],
            [dst_pts[x3][0]], [dst_pts[x3][1]]
        ])
        # calculating determinant of A for singular matrix calculation
        det_ma = np.linalg.det(ma)
        # error handling for singular matrix
        if det_ma == 0:
            continue

        ma = np.float32(ma)
        mb = np.float32(mb)

        mx = np.linalg.solve(ma, mb)
        matches = []
        for i in range(0, length):
            ma = np.matrix([
                [src_pts[i][0], src_pts[i][1], 0, 0, 1, 0], [0, 0, src_pts[i][0], src_pts[i][1], 0, 1]])
            # transforming features of image 1 into image 2
            mp2 = np.dot(ma, mx)
            # actual corresponding features of the image2
            mp1 = np.matrix([[dst_pts[i][0]], [dst_pts[i][1]]])
            # calculating the error of the
            # calculated transformed feature of image1 into image2 from the good matches
            dist1 = dist.euclidean(mp1, mp2)
            # condition for inliers
            if dist1 <= maxdist:
                count += 1
                matches.append(i)

        if count > maxcount:
            A = mx  # updating the transformation matrix
            matchindex = matches  # updating the indexes of the inliers
            maxcount = count  # updating the maximum count of inliers
    return A, matchindex;


# merges the two input images horizontally and returns the merged image and the width offset for the second image
def mergeimages(img_1, img_2):
    # reading the images
    img1 = Image.open(img_1)
    img2 = Image.open(img_2)
    # extracting the dimensions of the images
    (width1, height1) = img1.size
    (width2, height2) = img2.size
    # calculating the width of the merged images
    result_width = width1 + width2
    # calculating the height of the merged images
    result_height = max(height1, height2)
    # merged image
    result = Image.new('RGB', (result_width, result_height))
    # pasting image 1 on the left side of the merged image
    result.paste(img1, box=(0, 0))
    # pasting image 2 on the right side of the merged image
    result.paste(img2, box=(width1, 0))
    return np.array(result), width1;


def drawMatches(sift_img1, src_pts, sift_img2, dst_pts, indexes, img_matches):
    merged, width1 = mergeimages(sift_img1, sift_img2)  # merges the two images horizontally
    # for each good match
    for i in indexes:
        source = (src_pts[i][0], src_pts[i][1])
        dest = dst_pts[i]
        # updating the width pixel of the matching feature point of the image 2 in the merged image
        temp = int(dest[0] + width1)
        destination = (temp, dest[1])
        # drawing blue line from feature points in the image1 to the matching feature points in the image2
        cv2.line(merged, source, destination, (255, 0, 0))
    # storing the new image at MATCHES_IMAGE
    cv2.imwrite(img_matches, merged)
    return;


def findMatchesAllign(image1, image2, sift_img1, sift_img2, iterations, maxdist, lowe_ratio, img_affine,
                      matches_b_ransac, match_a_ransac):
    img1 = cv2.imread(image1, 0)  # scene
    img2 = cv2.imread(image2, 0)  # book

    # initializing SIFT detector
    sift = cv2.SIFT()
    # extracting keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # drawing keypoints for the input images
    img1_f = cv2.drawKeypoints(img1, kp1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_f = cv2.drawKeypoints(img2, kp2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # storing keypoints images at SIFT_IMG1 and SIFT_IMG2
    cv2.imwrite(sift_img1, img1_f)
    cv2.imwrite(sift_img2, img2_f)

    # Brute Force Matcher object
    bf = cv2.BFMatcher()
    # putative matches between region descriptors in each image
    # stores matching descriptor for image1 with 2 closest descriptors for image2
    matches = bf.knnMatch(des1, des2, k=2)
    # storing all the good matches
    good_matches = []
    for m, n in matches:
        if m.distance < lowe_ratio * n.distance:
            good_matches.append(m)

    # extracting matching features of the pair of image1 in src_pts and matching features of image2 in dst_pts
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # drawing matches from inlier features in image1 to image2
    matching_indexes = range(len(src_pts))
    drawMatches(sift_img1, src_pts, sift_img2, dst_pts, matching_indexes, matches_b_ransac)

    # calling RANSAC
    transformation_mat, result_indexes = ransac(src_pts, dst_pts, src_pts.shape[0], iterations, maxdist)

    # drawing good matches after RANSAC
    drawMatches(sift_img1, src_pts, sift_img2, dst_pts, result_indexes, match_a_ransac)

    # calling final refit
    transformation_mat = finalrefit(result_indexes, src_pts, dst_pts)  # H Matrix

    # reshaping the tranformation matrix
    # from [m1 m2 m3 m4 t1 t2] to
    # [[m1 m2 t1], [m3 m4 t2]]
    transformation_mat_reshape = np.array([
        transformation_mat[0], transformation_mat[1], transformation_mat[4],
        transformation_mat[2], transformation_mat[3], transformation_mat[5]
    ])
    transformation_mat_reshape = transformation_mat_reshape.reshape(2, 3)
    print "Answer 2"
    print "H Matrix: \n"
    print transformation_mat_reshape
    print "\n"
    # tranforming image 1 to image 2
    rows, cols = img1.shape
    dst = cv2.warpAffine(img1, transformation_mat_reshape, (cols, rows))
    # storing transformed image at IMG_AFFINE
    cv2.imwrite(img_affine, dst)
    return;
