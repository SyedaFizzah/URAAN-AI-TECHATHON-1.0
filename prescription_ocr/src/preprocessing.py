

## src/preprocessing.py

"""Preprocessing utilities: denoise, contrast, resize, transpose similar to helper.preprocess
"""
import cv2
import numpy as np


def preprocess(img, imgSize=(128,32)):
    # img: grayscale uint8
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # increase contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img = clahe.apply(img)

    # denoise
    img = cv2.bilateralFilter(img, 9, 75, 75)

    # adaptive threshold then invert -> maintain 0=black stroke
    th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 15, 8)
    img = 255 - th

    widthTarget, heightTarget = imgSize
    height, width = img.shape

    factor_x = width / widthTarget
    factor_y = height / heightTarget
    factor = max(factor_x, factor_y)
    newSize = (max(1, min(widthTarget, int(width / factor))), max(1, min(heightTarget, int(height / factor))))
    img = cv2.resize(img, newSize)
    target = np.ones(shape=(heightTarget, widthTarget), dtype='uint8') * 255
    target[0:newSize[1], 0:newSize[0]] = img

    img = cv2.transpose(target)
    mean, stddev = cv2.meanStdDev(img)
    mean = mean[0][0]
    stddev = stddev[0][0]
    img = img - mean
    if stddev > 0:
        img = img // stddev
    return img.astype('uint8')


if __name__ == '__main__':
    import sys
    import cv2
    img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    out = preprocess(img)
    cv2.imwrite('debug_preprocessed.png', out)
