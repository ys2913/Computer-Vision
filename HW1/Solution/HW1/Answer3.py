from io import StringIO
import numpy as np
from scipy import linalg

WORLD_FILE = 'ass1/world.txt'
IMAGE_FILE = 'ass1/image.txt'


def checkTransmat(A, world, image):
    image1 = np.dot(world, np.transpose(A))

    for j in range(image1.shape[0]):
        for i in range(3):
            image1[j][i] /= image1[j][2]

    error = image1 - image
    return error;


def convtoReal(v):
    for i in range(len(v)):
        v[i] /= v[len(v) - 1]
    return v[:3];


def camparam():
    world = np.loadtxt(WORLD_FILE)
    image = np.loadtxt(IMAGE_FILE)

    worldlen = len(world[0])
    imagelen = len(image[0])

    world_h = []
    image_h = []

    for i in range(worldlen):
        temp = []
        temp.append(world[0][i])
        temp.append(world[1][i])
        temp.append(world[2][i])
        temp.append(1)
        world_h.append(temp)

    for i in range(imagelen):
        temp = []
        temp.append(image[0][i])
        temp.append(image[1][i])
        temp.append(1)
        image_h.append(temp)

    A = []
    for i in range(worldlen):
        temp = []
        for j in range(4):
            temp.append(0)
        for j in range(4):
            temp.append(world_h[i][j])
        for j in range(4):
            temp.append(-image_h[i][1] * world_h[i][j])
        A.append(temp)

        temp = []
        for j in range(4):
            temp.append(world_h[i][j])
        for j in range(4):
            temp.append(0)
        for j in range(4):
            temp.append(-image_h[i][0] * world_h[i][j])
        A.append(temp)

    u, s, v = np.linalg.svd(A)

    P = []
    i = 0
    while i < 12:
        temp = []
        for j in range(4):
            temp.append(v[len(s) - 1][i])
            i += 1
        P.append(temp)

    error = checkTransmat(P, world_h, image_h)  # verifying by checking the error

    u, s, v = np.linalg.svd(P)
    C = v[len(v) - 1]
    C_real = convtoReal(C)
    C_2 = QRmethod(P)
    error = C_real-C_2
    print error
    return;


def QRmethod(P):
    C=0
    P = np.array(P).reshape(3,4)
    r, q = linalg.rq(P,mode='economic')
    R = np.array(q[:,:3])
    t = q[:,3:4]
    Cnh = -1*np.linalg.solve(R,t)
    return np.transpose(Cnh);

camparam()
