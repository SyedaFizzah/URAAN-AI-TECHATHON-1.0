"""Simple connected-component based word/line segmentation for prescriptions.
Returns list of word images in reading order (top-to-bottom then left-to-right).
"""
import cv2
import numpy as np


def segment_lines_and_words(img_gray):
    # img_gray: grayscale
    # compute binary
    _, th = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = 255 - th

    # dilate to connect characters in a word
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    connected = cv2.dilate(th, kernel, iterations=1)

    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 10 or h < 8:
            continue
        boxes.append((x, y, w, h))

    # sort by y then x
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

    # extract images (words/line chunks)
    words = []
    for (x, y, w, h) in boxes:
        crop = img_gray[max(0, y-3): y+h+3, max(0, x-3): x+w+3]
        words.append(((x,y,w,h), crop))
    return words


if __name__ == '__main__':
    import sys
    img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    words = segment_lines_and_words(img)
    print('found', len(words))