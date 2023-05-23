import cv2
import numpy as np
import imageio.v3 as imageio
import matplotlib.pyplot as plt

def get_area(segm):
    plt.imshow(segm)
    plt.savefig('tmp_imgs/segm.png')
    return np.count_nonzero(segm)

def get_bbox(segm, img_id):
    ret, thresh = cv2.threshold(segm, 0, 255, 0)
    print(ret)
    print(thresh.shape)
    
    imageio.imwrite('tmp_imgs/thresh.png', thresh)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(cnts)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    rect = cv2.minAreaRect(cnts[0])
    print('rect ', rect)
    dims = rect[1]
    size = max(dims)
    print('dims ', dims, size)
    box = np.int0(cv2.boxPoints(rect))
    cv2.drawContours(thresh, [box], 0, (128,0,0), 1)
    
    cv2.imwrite('tmp_imgs/{}.png'.format(img_id), thresh)
    
    return size
    
def cv_test():
    # Load image, convert to grayscale, Otsu's threshold for binary image
    image = cv2.imread('1.jpg')
    print(image.shape)
    #print(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Find contours, find rotated rectangle, obtain four verticies, and draw 
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    rect = cv2.minAreaRect(cnts[0])
    box = np.int0(cv2.boxPoints(rect))
    cv2.drawContours(image, [box], 0, (36,255,12), 3) # OR
    # cv2.polylines(image, [box], True, (36,255,12), 3)

    #cv2.imshow('image', image)
    #cv2.waitKey()

def get_size(segm, img_id):
    size = get_area(segm)
    #size = get_bbox(segm, img_id)
    return size
