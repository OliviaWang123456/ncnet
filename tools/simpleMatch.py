import math
import numpy as np
import skimage.io as sio

def matchScore(realMask, synMask):
    realMask = realMask > 1
    realMask = realMask.astype(int)

    synMask = synMask > 1
    synMask = synMask.astype(int)

    intersection = realMask * synMask
    union = realMask + synMask
    union = union > 1
    union = union.astype(int)

    iou = intersection.sum() / union.sum()

    return iou

