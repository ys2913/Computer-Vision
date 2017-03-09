import numpy as np
from scipy.misc import imread
from scipy.misc import toimage

# returns approximate 1-D gaussian kernel of the size kernel_size
def buildkernel(kernel_size):
    mid = kernel_size // 2 + 1  # gives floor integer value
    # initializing the 1D gaussain kernel with dummy values
    gauslist = range(kernel_size)
    # listsum stores the total sum of gaussian kernel index values
    listsum = 0

    # starting from the middle index as array[mid-i]=array[mid+i]
    for i in range(0, mid):
        # updating the value to be stored in the index [mid-i]
        value = mid - i
        # For exponential value we can use
        # value = np.math.exp(-.5 * (i ** 2))
        # for sigma as 1
        # updating the value for the index [mid-i]
        gauslist[mid - 1 - i] = value
        gauslist[mid - 1 + i] = value
        # updating the total sum
        listsum += value
        # middle index element is added once
        # other indexes are added to the filter twice
        if i > 0:
            listsum += value

    for i in range(0, mid):
        value = (1.0 * gauslist[mid - 1 - i]) / listsum
        gauslist[mid - 1 - i] = value
        gauslist[mid - 1 + i] = value
    return np.array(gauslist)


# returns the row convolved matrix(image) with the kernel
def convolve1D(image, kernel):
    rows = image.shape[0]
    rowblur = []
    # applying guassian kernel for each row in the image
    for i in range(0, rows):
        row = image[i]
        row_blur = np.convolve(row, kernel, 'valid')    # valid boundary convolution
        rowblur.append(row_blur)
    return np.array(rowblur)


# given the size of the gaussian kernel, returns the gaussian blurred image
def gaussianblur(IMAGE_1, kernel_size, BLURRED_IMAGE_1):
    image = np.array(imread(IMAGE_1))  # reads the given image
    gaus_kernel = buildkernel(kernel_size)  # returns the 1D gaussian kernel of the given size
    row_blurimage = convolve1D(image, gaus_kernel)  # gaussian kernel applied to the row
    # for applying gau_kernel to columns, we take the transpose of the input image (converting columns to rows)
    # and then take the transpose of the output image (converting back to the original image)
    blurred_image = np.transpose(convolve1D(np.transpose(row_blurimage), gaus_kernel))
    toimage(blurred_image).save(BLURRED_IMAGE_1)
    return

