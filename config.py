#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :config.py
@Description :
@Time        :2020/07/31 10:29:19
@Author      :Jinkui Hao
@Version     :1.0
'''

class Config():

    irisPath = '/media/hjk/10E3196B10E3196B/dataSets/AS-OCT/iris'
    saveFileName = '/media/hjk/10E3196B10E3196B/dataSets/result/2019121201/A1'

    ENV_NAME = 'Iris_Seg'

    isParalleling = True
    isDiceLoss = True
    #结果保存文件夹

    maxEpoch = 150
    train_batch_size = 8
    base_lr = 0.0001
    weight_decay = 0.0001

