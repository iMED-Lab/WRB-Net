import torch.utils.data as data
import torch
import numpy as np
import os
from PIL import Image
import random
from torchvision import transforms
import torchvision.transforms.functional as TF
import cv2
from scipy import misc

def random_crop(data, label, crop_size):
    # Random crop
    i, j, h, w = transforms.RandomCrop.get_params(data, output_size=(512, 512))
    data = TF.crop(data, i, j, h, w)
    label = TF.crop(label, i, j, h, w)
    return data, label

def img_transforms(img, label, crop_size):
    img, label = random_crop(img, label, crop_size)
    img_tfs = transforms.Compose([
        #transforms.ColorJitter(brightness=(80,150),contrast=(80,150),saturation=(80,150)),
        transforms.ColorJitter(brightness=0.6,contrast=0.8,saturation=0.6),
        transforms.ToTensor(),
    ])
    label_tfs = transforms.ToTensor()

    img = img_tfs(img)
    label = label_tfs(label)

    return img, label

class IrisDataset(data.Dataset):

    def __init__(self, root,isTraining = True):
        self.labelPath = self.get_dataPath(root, isTraining)
        self.imgPath = root
        self.isTraining = isTraining
        self.name = ' '
        self.root = root

    def __getitem__(self, index):
  
        labelPath = self.labelPath[index]

        filename = labelPath.split('/')[-1]
        self.name = filename
        #print(filename)

        imagePath = self.root + '/images/' + filename[5:]
        #打开图像并进行增强
        # img_transform = transforms.Compose([
        #     #transforms.Grayscale(num_output_channels=3),
        #     transforms.ColorJitter(brightness=0.7, contrast=0.6, saturation=0.5),
        #     transforms.ToTensor(),
        # ])
        # simple_transform = transforms.ToTensor()

        img = np.array(Image.open(imagePath))
        mask = np.array(Image.open(labelPath))
        img = cv2.resize(img,(536,536))
        mask = cv2.resize(mask, (536,536))
        _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = Image.fromarray(np.uint8(img))
        mask = Image.fromarray(np.uint8(mask))
        if self.isTraining:
            rotate = 10
            # augumentation
            angel = random.randint(-rotate, rotate)
            img = img.rotate(angel)
            mask = mask.rotate(angel)
            # img = img_transform(img)
            # mask = simple_transform(mask)
            img, mask = img_transforms(img, mask, (512, 512))

        else:
            img_transform = transforms.Compose([
                transforms.FiveCrop((512, 512)),  # this is a list of PIL Images
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                # returns a 4D tensor
            ])
            label_transform = transforms.Compose([
                # transforms.CenterCrop(512),
                transforms.ToTensor(),
            ])
            img = img_transform(img)
            mask = label_transform(mask)

        return img, mask

    def __len__(self):
        '''
        返回总的图像数量
        :return:
        '''
        return len(self.labelPath)

    def get_dataPath(self, root, isTraining):
        '''
        依次读取输入图片和label的文件路径，并放到array中返回
        :param root: 存放的文件夹
        :return:
        '''
        if isTraining:
            root = os.path.join(root + "/labels/train")
            imgPath = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        else:
            root = os.path.join(root + "/labels/test")
            imgPath = list(map(lambda x: os.path.join(root, x), os.listdir(root)))

        #print(len(imgPath))

        return imgPath

    def getFileName(self):
        return self.name

class IrisDataset_all(data.Dataset):

    def __init__(self,root,isTraining = True):
        self.labelPath = self.get_dataPath(root, isTraining)
        self.imgPath = root
        self.isTraining = isTraining
        self.name = ' '

    def __getitem__(self, index):
        '''
        内建函数，当对该类的实例进行类似字典的操作时，就会自动执行该函数，并返会对应的值
        这是必须要重载的函数，就是实现给定索引，返回对应的图像
        给出图像编号，返回变换后的输入图像和对应的label
        :param index: 图像编号
        :return:
        '''
        labelPath = self.labelPath[index]

        filename = labelPath.split('/')[-1]
        self.name = filename
        #print(filename)

        imagePath = self.imgPath + filename
        # img = np.zeros((1072, 1072, 3), dtype=np.float32)
        # image1 = np.array(Image.open(imagePath))
        # img[0:1072, 0:1072, :] = image1[0:1072, 0:1072,:]
        # img = cv2.resize(img,(536,536))
        img = np.array(Image.open(imagePath))
        img = Image.fromarray(np.uint8(img))

        img_transform = transforms.Compose([
            transforms.FiveCrop((512, 512)),  # this is a list of PIL Images
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            # returns a 4D tensor
        ])
        img = img_transform(img)

        return img

    def __len__(self):
        '''
        返回总的图像数量
        :return:
        '''
        return len(self.labelPath)

    def get_dataPath(self, root, isTraining):
        '''
        依次读取输入图片和label的文件路径，并放到array中返回
        :param root: 存放的文件夹
        :return:
        '''

        imgPath = list(map(lambda x: os.path.join(root, x), os.listdir(root)))

        return imgPath

    def getFileName(self):
        return self.name


