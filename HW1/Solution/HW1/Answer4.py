import scipy.io as sio
import numpy as np
from matplotlib import pyplot
import pylab
from mpl_toolkits.mplot3d import Axes3D
import random

SFM_POINTS = 'ass1/sfm_points.mat'


def calculatecentroid(mat, r, c, h):
    cen = []
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
    #print np.array(W)
    return np.array(W);


def struMotion():
    mat_contents = sio.loadmat(SFM_POINTS)
    imgpoints = mat_contents['image_points']
    shape = imgpoints.shape
    centroid = calculatecentroid(imgpoints, shape[0], shape[1], shape[2])
    imgpoints = removemeans(imgpoints, centroid, shape[1], shape[2])
    W = measurementmat(imgpoints, shape[1], shape[2])
    u, w, v = np.linalg.svd(W)

    U = u[:, :3]
    W = np.transpose(w[:3])

    for i in range(3):
        for j in range(20):
            U[j][i] *= W[i]

    vx = v[0]
    vy = v[1]
    vz = v[2]
    # plotting 3D points
    fig = pylab.figure()
    ax = Axes3D(fig)


    ax.scatter(vx,vy,vz)
    pyplot.show()
    print "end"
    return;

struMotion()
