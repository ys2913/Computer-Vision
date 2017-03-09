import scipy.io as sio
import numpy as np
import pylab
from mpl_toolkits.mplot3d import Axes3D


def calculatecentroid(mat, r, c, h):
    cen = []
    # for each frame 0 to h calculate centroid (Cx,Cy)
    for i in range(h):
        xsum = 0
        ysum = 0
        tcent = []
        for j in range(c):
            xsum = xsum + mat[0][j][i]
            ysum = ysum + mat[1][j][i]

        tcent.append(xsum / c)
        tcent.append(ysum / c)
        cen.append(tcent)
    cen = np.array(cen).reshape(h, 2)
    return np.transpose(cen);


def removemeans(imgpoints, centroid, c, h):
    for i in range(h):
        for j in range(c):
            imgpoints[0][j][i] -= centroid[0][i]
            imgpoints[1][j][i] -= centroid[1][i]
    return imgpoints;


def measurementmat(img, c, h):
    W = []
    for i in range(h):
        x = []
        y = []
        for j in range(c):
            x.append(img[0][j][i])
            y.append(img[1][j][i])
        W.append(x)
        W.append(y)
    # print np.array(W)
    return np.array(W);


def struMotion(sfm_points, plot3D):
    # reading .mat file
    mat_contents = sio.loadmat(sfm_points)
    # extracting image points
    imgpoints = mat_contents['image_points']
    shape = imgpoints.shape
    # computing centroids in each image frame ti is each row[i] of centroid
    # centroid[0][j] represents the x coordindates  for the jth frame
    # centroid[1][j] represents the y coordindates  for the jth frame
    centroid = calculatecentroid(imgpoints, shape[0], shape[1], shape[2])
    # centering the points in each image
    imgpoints = removemeans(imgpoints, centroid, shape[1], shape[2])
    # constructing the measurement matrix from the centred image points
    W = measurementmat(imgpoints, shape[1], shape[2])

    u, d, v = np.linalg.svd(W)
    # extracting first 3 columns of u
    M = u[:, :3]
    D = np.transpose(d[:3])
    # multiplying each column with the respective eigenvalue
    for i in range(3):
        for j in range(20):
            M[j][i] *= D[i]

    # first 3 rows of V corresponds to the 3D world points
    vx = v[0]  # x coordinate
    vy = v[1]  # y coordinate
    vz = v[2]  # z coordinate

    # plotting 3D points
    fig = pylab.figure()
    ax = Axes3D(fig)
    ax.scatter(vx, vy, vz)
    fig.savefig(plot3D)

    print "Answer 4"
    print "M1="
    print M[:2, :]
    print "\nt1="
    print centroid[0][0], centroid[1][0]
    print "\n"
    # printing top 10 3D world points
    print "World Coordinates :"
    World = []
    for i in range(10):
        temp = []
        temp.append(vx[i])
        temp.append(vy[i])
        temp.append(vz[i])
        World.append(temp)
    print np.array(World).reshape(10,3)
    return;
