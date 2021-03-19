import cv2
from matplotlib import pyplot
from scipy.ndimage import convolve, maximum_filter
import numpy as np
import skimage.measure

img = cv2.imread('Monkey.jpg', cv2.IMREAD_GRAYSCALE)
# For some reason using cv2.randu the img variable will be the random image one,
# even when not assigned so I have to make an extra image for that purpose only
randomColor = cv2.imread('Monkey.jpg', cv2.IMREAD_GRAYSCALE)

img = img.astype(np.int16)


def myimage(image):
    # image view
    pyplot.imshow(image, cmap='gray', vmin=0, vmax=255)
    pyplot.show()
    # pixel view
    print('image size: ', image.shape)
    ('pixel matrix:\n', image)

SIZE = 320
img = cv2.resize(img, (SIZE, SIZE))
myimage(img)

# A:
randomColor = cv2.randu(randomColor, 0, 255);
myimage(randomColor)

# B: Using kernel that detectes edges
convertedImg = convolve(img, np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]))
myimage(convertedImg)

#C & D: maxpool with 2x2
maxPool = skimage.measure.block_reduce(img, (2,2), np.max)
myimage(maxPool)
