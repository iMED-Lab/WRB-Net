'''
该函数用来获取虹膜轮廓的下边界
'''

import math
import numpy as np
from PIL import Image
import cv2
import copy
import os

wholeImgPath = '/media/hjk/10E3196B10E3196B/dataSets/iris/gausslabel4/'
upperImgPath = '/media/hjk/10E3196B10E3196B/dataSets/iris/gausslabel4down/'
outPath = '/media/hjk/10E3196B10E3196B/dataSets/iris/finallabel4/'
filenames = os.listdir(wholeImgPath)
l = 0
for filename in filenames:
    l = l + 1
    print(l,'.....')
    if os.path.splitext(filename)[1] == ".jpg":  # 筛选csv文件
        whole_image_path = wholeImgPath + filename
        whole_image = cv2.imread(whole_image_path)
        upper_image_path = upperImgPath + filename
        upper_image = cv2.imread(upper_image_path)
        down_image = cv2.add(upper_image,whole_image)
        cv2.imwrite(outPath + filename, down_image)







