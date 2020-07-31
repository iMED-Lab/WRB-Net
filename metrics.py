"""
===============================
Evaluating segmentation metrics
===============================

When trying out different segmentation methods, how do you know which one is
best? If you have a *ground truth* or *gold standard* segmentation, you can use
various metrics to check how close each automated method comes to the truth.
In this example we use an easy-to-segment image as an example of how to
interpret various segmentation metrics. We will use the the adapted Rand error
and the variation of information as example metrics, and see how
*oversegmentation* (splitting of true segments into too many sub-segments) and
*undersegmentation* (merging of different true segments into a single segment)
affect the different scores.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage import data
from skiMetrics import adapted_rand_error, variation_of_information
from skiMetrics import mean_squared_error, normalized_root_mse, mean_absoluted_error
from skimage.filters import sobel
from skimage.measure import label
from skimage.util import img_as_float
from skimage.feature import canny
from skimage.morphology import remove_small_objects
from skimage.segmentation import ( watershed, mark_boundaries)
import  cv2
#image = data.coins()
def getPoint_one(image):
    '''计算单张图片的边缘坐标'''
    # 计算GT的边缘坐标
    binary_image = np.zeros(shape=np.shape(image))
    binary_image[image == 255] = 1
    edge_pred = np.zeros(shape=[1, np.shape(binary_image)[1]])

    for i in range(np.shape(binary_image)[1]):
        location_point_1 = np.where(binary_image[:, i] > 0)[0]
        if len(location_point_1) == 1:
            edge_pred[0, i] = location_point_1
        elif len(location_point_1) == 0:
            edge_pred[0, i] = edge_pred[0, i - 1]
        else:
            edge_pred[0, i] = location_point_1[0]

    return edge_pred

def getPosition(pred,GT):
    '''计算边缘的坐标点,用来计算MSE,MAE等指标'''
    #由于边缘长短可能不一致,所以取最短的,同时把长短差值也作为误差返回
    #先计算两张图片分别的起始点
    m = np.shape(GT)
    startPoint_1 = 0
    endPoint_1 = 0
    startPoint_2 = 0
    endPoint_2 = 0
    for i in range(m[1]):
        sumNow = np.sum(pred[:, i])
        if sumNow > 200:
            startPoint_1 = i
            for i in range(m[1]-1,0,-1):
                sumNow = np.sum(pred[:, i])
                if sumNow > 200:
                    endPoint_1 = i
                    break
            break
    for i in range(m[1]):
        sumNow = np.sum(GT[:, i])
        if sumNow > 200:
            startPoint_2 = i
            for i in range(m[1]-1,0,-1):
                sumNow = np.sum(GT[:, i])
                if sumNow > 200:
                    endPoint_2 = i
                    break
            break

    start = max(startPoint_1,startPoint_2)
    end = min(endPoint_1,endPoint_2)
    error_sub = np.abs((endPoint_2 - startPoint_2) - (endPoint_1 - startPoint_1))

    #计算pred的边缘坐标
    edge_pred = getPoint_one(pred)
    edge_GT = getPoint_one(GT)

    r1 = edge_pred[0,start:end]
    r2 = edge_GT[0,start:end]

    if end == 0 or start == 0:
        return np.zeros((1,2)), np.zeros((1,2)), error_sub
    return r1, r2, error_sub

def getTICError(pred, GT, fileName,length=40):
    m = np.shape(GT)
    TICMap = np.zeros(m)
    GTMap = np.zeros(m)
    #print(fileName)
    if 'right' in fileName:
        GT = cv2.flip(GT,1)
        pred = cv2.flip(pred,1)
    _, GT = cv2.threshold(GT, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, pred = cv2.threshold(pred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    endPoint = 0
    for i in range(m[1]):
        sumNow = np.sum(GT[:, i])
        if sumNow > 200:
            startPoint = i
            endPoint = i + length
            GTMap[:, startPoint:endPoint] = GT[:, startPoint:endPoint]
            break
    for i in range(m[1]):
        sumNow = np.sum(pred[:, i])
        if sumNow > 200:
            startPoint = i
            TICMap[:, startPoint:endPoint] = pred[:, startPoint:endPoint]
            break
    #将边缘转换为坐标
    pred_p, GT_p, error = getPosition(TICMap,GTMap)
    result = mean_absoluted_error(pred_p, GT_p) + error
    return result

def get_MSE(pred,GT,isEdge = False):
    if isEdge:
        pred = cv2.Canny(pred, 50, 150)
        GT = cv2.Canny(GT, 50, 150)
    pred_p, GT_p, error = getPosition(pred, GT)
    return mean_squared_error(pred_p, GT_p) + error**2

def get_NRMSE(pred,GT,isEdge = False):
    if isEdge:
        pred = cv2.Canny(pred, 50, 150)
        GT = cv2.Canny(GT, 50, 150)
    return normalized_root_mse(pred, GT)

def get_Rand_error(pred,GT,isEdge = False):
    if isEdge:
        pred = cv2.Canny(pred, 50, 150)
        GT = cv2.Canny(GT, 50, 150)
    error, precision, recall = adapted_rand_error(GT, pred)

    return error

def get_False_splt_merge(pred,GT,isEdge = False):
    if isEdge:
        pred = cv2.Canny(pred, 50, 150)
        GT = cv2.Canny(GT, 50, 150)
    splits, merges = variation_of_information(GT, pred)

    return splits, merges
###############################################################################
# First, we generate the true segmentation. For this simple image, we know
# exact functions and parameters that will produce a perfect segmentation. In
# a real scenario, typically you would generate ground truth by manual
# annotation or "painting" of a segmentation.

# elevation_map = sobel(image)
# markers = np.zeros_like(image)
# markers[image < 30] = 1
# markers[image > 150] = 2
# im_true = watershed(elevation_map, markers)
# im_true = ndi.label(ndi.binary_fill_holes(im_true - 1))[0]


###############################################################################
# Next, we create three different segmentations with different characteristics.
# The first one uses :func:`skimage.segmentation.watershed` with
# *compactness*, which is a useful initial segmentation but too fine as a
# final result. We will see how this causes the oversegmentation metrics to
# shoot up.

# edges = sobel(image)
# im_test1 = watershed(edges, markers=468, compactness=0.001)
#
# ###############################################################################
# # The next approach uses the Canny edge filter, :func:`skimage.filters.canny`.
# # This is a very good edge finder, and gives balanced results.
#

#GT = ndi.label(remove_small_objects(GT, 21))[0]
if __name__ == '__main__':

    GTPath = 'datas/GT.jpg'
    res1Path = 'datas/unet_res.png'
    res2Path = 'datas/wbnet_res.png'
    imagePath = 'datas/image.jpg'

    GT = cv2.imread(GTPath)
    GT = cv2.resize(GT,(536,536))
    GT = GT[:,:,1]

    res1 = cv2.imread(res1Path)
    res1 = res1[:,:,1]
    res2 = cv2.imread(res2Path)
    res2 = res2[:,:,1]
    image = cv2.imread(imagePath)
    image = cv2.resize(image,(536,536))
    image = image[:,:,1]

    GT = cv2.Canny(GT, 50, 150)
    res1 = cv2.Canny(res1, 50, 150)
    res2 = cv2.Canny(res2, 50, 150)
    cv2.imwrite('res1_edge.jpg',res1)
    cv2.imwrite('res2_edge.jpg',res2)
    cv2.imwrite('GT_edge.jpg',GT)

    plt.subplot(221),plt.imshow(mark_boundaries(image, GT))
    plt.subplot(222),plt.imshow(mark_boundaries(image, res1))
    plt.subplot(223),plt.imshow(mark_boundaries(image, res2))
    plt.show()

    error, precision, recall = adapted_rand_error(GT, res1)
    splits, merges = variation_of_information(GT, res1)
    mse = mean_squared_error(GT, res1)
    nrmse = normalized_root_mse(GT, res1)

    print(f"Unet:")
    print(f"Adapted Rand error: {error}")
    print(f"Adapted Rand precision: {precision}")
    print(f"Adapted Rand recall: {recall}")
    print(f"False Splits: {splits}")
    print(f"False Merges: {merges}")
    print(f"MSE: {mse}")
    print(f"NRMSE: {nrmse}")

    error, precision, recall = adapted_rand_error(GT, res2)
    splits, merges = variation_of_information(GT, res2)
    mse = mean_squared_error(res2, GT)
    nrmse = normalized_root_mse(GT, res2)

    print(f"WBUnet:")
    print(f"Adapted Rand error: {error}")
    print(f"Adapted Rand precision: {precision}")
    print(f"Adapted Rand recall: {recall}")
    print(f"False Splits: {splits}")
    print(f"False Merges: {merges}")
    print(f"MSE: {mse}")
    print(f"NRMSE: {nrmse}")

