import numpy as np
from scipy.misc import imread
from scipy.misc import toimage


def buildkernel(size):
    mid = size // 2
    gauslist = range(size)
    listsum = 0

    for i in range(0, mid + 1):
        value = 2*(mid + 1 - i)  # For exponential value we can use math.exp(-.5 * (i ** 2)), taking sigma as 1
        gauslist[mid - i] = value
        gauslist[mid + i] = value
        listsum += value
        if i > 0:
            listsum += value

    for i in range(0, mid + 1):
        value = (1.0 * gauslist[mid - i]) / listsum
        gauslist[mid - i] = value
        gauslist[mid + i] = value
    return np.array(gauslist)


def convolve1D(image, kernel):
    rows = image.shape[0]
    xblur = []

    for i in range(0, rows):
        temp = image[i]
        tblur = np.convolve(temp, kernel, 'valid')
        xblur.append(tblur)
    return np.array(xblur)


def gaussianblur(image, size):
    image = np.array(imread(image))  # reads the given image
    #toimage(image).show()
    kernel = buildkernel(size)  # returns the 1D gaussian kernal of the given size
    xblurimage = convolve1D(image, kernel)
    xyblurimage = np.transpose(convolve1D(np.transpose(xblurimage), kernel))
    toimage(image).save('blurred.png')
    return xyblurimage


image = 'scene.pgm'
size = 3
im = gaussianblur(image, size)
print "Ended"
#toimage(im).show()
