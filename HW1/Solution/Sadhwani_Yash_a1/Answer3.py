import numpy as np
from scipy import linalg


# returns the error values for the reprojected world points from the actual world point
def checkTransmat(A, world, image):
    # Reprojecting to the world point
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


def camparam(world, image):
    world = np.loadtxt(world)  # reading world.txt
    image = np.loadtxt(image)  # reading image.txt

    worldlen = len(world[0])
    imagelen = len(image[0])

    world_h = []  # homogeneous coordinates of world
    image_h = []  # homogeneous coordinates of image

    # appending 1 to the non homogeneous point to convert them into homogeneous coordinates
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

    # converting x x PX=0 into matrix A, such that Ap=0
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

    # Solving for p in Ap=0
    u, s, v = np.linalg.svd(A)

    P = []
    i = 0

    # Extracting the last row for v, as it corresponds to the minimum eigenvalue
    while i < 12:
        temp = []
        for j in range(4):
            temp.append(v[len(s) - 1][i])
            i += 1
        P.append(temp)

    P = np.array(P).reshape(3, 4)

    print "Answer 3"
    print "P="
    print P

    # verifying by checking the error
    error = checkTransmat(P, world_h, image_h)
    # solving for C in PC=0
    u, s, v = np.linalg.svd(P)
    # picking vector corresponding to the null space
    C = v[len(v) - 1]
    # converting to real coordinate system
    C_real = convtoReal(C)
    print "C="
    print C_real
    # solving for C by decomposing P into its constituent matrices
    C_2 = QRmethod(P)
    # calculating the error in calculations from the alternative way
    error = C_real - C_2

    return;


def QRmethod(P):
    C = 0
    P = np.array(P).reshape(3, 4)
    # extracting K and [R|t] into r and q respectively
    r, q = linalg.rq(P, mode='economic')
    # rotation matrix R as q=[R|t]
    R = np.array(q[:, :3])
    t = q[:, 3:4]
    # calculating C non homogenoeous from R and t as t=-RC
    Cnh = -1 * np.linalg.solve(R, t)
    return np.transpose(Cnh);
