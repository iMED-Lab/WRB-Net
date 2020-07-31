#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :main.py
@Description : train and test
@Time        :2020/07/31 10:31:50
@Author      :Jinkui Hao
@Version     :1.0
'''

from torch.utils.data import DataLoader
from torch import  optim
from dataset import  IrisDataset
from tensorboardX import SummaryWriter
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
from evaluation import *
from Visualizer import Visualizer
from functools import partial
from model import Our_WRB
import matlab.engine
from metrics import get_MSE,get_False_splt_merge,get_NRMSE,get_Rand_error,getTICError
from config import Config


def get_aveOutput(five_output):
    five_output_np = five_output.cpu().numpy()
    output = np.zeros((6,536,536))
    output[0,:512,:512] = five_output_np[0,:,:]
    output[1,:512, -512:] = five_output_np[1, :, :]
    output[2,-512:,:512] = five_output_np[2,:,:]
    output[3,-512:, -512:] = five_output_np[3, :, :]
    output[4,-512:, -512:] = five_output_np[4, :, :]
    for i in range(4):
        output[5,:,:] = np.maximum(output[5,:,:],output[i,:,:])

    return output[5,:,:] 


def findUpBoundary(eng,edgeImg):
    '''
    :param eng: matlab engine
    :param edgeImg: segmentation result
    :return: upper boundary
    '''
    output = edgeImg.tolist()
    rest = eng.iris_seg_up(output)
    numpy_res = np.zeros((536, 536))
    numpy_res[:, :] = rest
    #cv2.imwrite('edges.jpg',numpy_res*255)

    return numpy_res*255

def test(model, dataloader):
    model.eval()

    DC = 0. # Dice 
    AUC = 0.
    newAcc = 0.
    newSen = 0.
    newSpe = 0.
    IOU = 0.
    HausDist = 0.

    MSE = 0.0
    NRMSE = 0.0
    Rand_error = 0.0
    Split_error = 0.0
    merge_error = 0.0
    TICerror = 0.0

    i = 0
    plt.ion()

    with torch.no_grad():
        for img, true_mask in dataloader:
            print('Evaluate %03d...' %i)
            i += 1

            img = img.cuda()
            true_mask = true_mask.cuda()

            image = img[0, :, :, :, :]
            #image = img
            mask_pred_ori = model(image)  # fuse batch size and ncrops
            # five_mask_pred = torch.argmax(five_mask_pred, dim=1)
            mask_pred_ori = get_aveOutput(mask_pred_ori)
            AUC += AUC_score(mask_pred_ori, true_mask)

            th2 = ((mask_pred_ori) * 255).astype('uint8')
            value, threshed_pred = cv2.threshold(th2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            true_mask = torch.squeeze(true_mask).cpu().numpy()

            imgName = dataloader.dataset.getFileName()


            edge_GT = np.uint8(findUpBoundary(eng, true_mask*255))
            edge_pred = np.uint8(findUpBoundary(eng, threshed_pred))
            MSE = get_MSE(edge_pred, edge_GT)
            NRMSE = get_NRMSE(edge_pred, edge_GT)
            Rand_error = get_Rand_error(edge_pred, edge_GT)
            Split_error, merge_error = get_False_splt_merge(edge_pred, edge_GT)
            HausDist += hausdorff_score(edge_pred/255, edge_GT/255)
            tictemp = getTICError(edge_pred, edge_GT,imgName)
            if np.isnan(tictemp):
                tictemp = 0
            print(tictemp)
            TICerror += tictemp

            IOU += intersection_over_union(threshed_pred/255, true_mask)
            mask_pred = torch.from_numpy(threshed_pred / 255.0)
            true_mask = torch.from_numpy(true_mask)
            DC += get_DC(mask_pred, true_mask)
            temp = confusion(mask_pred, true_mask)
            newAcc += temp["Accuracy"]
            newSen += temp["TPR"]
            newSpe += temp["TNR"]

    length = len(dataset.dataset)
    print('length is :', length)

    AUC = AUC / length
    newAcc = newAcc / length
    newSen = newSen / length
    newSpe = newSpe / length
    DC = DC / length
    IOU = IOU / length
    MSE = MSE / length
    NRMSE = NRMSE  / length
    Rand_error = Rand_error / length
    Split_error = Split_error / length
    TICerror = TICerror / length
    HausDist = HausDist / length

    return AUC, newAcc, newSen, newSpe, DC, IOU, MSE,NRMSE,Rand_error,Split_error,TICerror,HausDist


def train(model,dataloader_train, dataloader_test):
 
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.base_lr ,weight_decay = Config.weight_decay)
    schedulers = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=Config.maxEpoch,last_epoch=-1)

    for epoch in range(Config.maxEpoch):
        print('Epoch %d/%d' % (epoch, Config.maxEpoch-1))
        print('-'*10)
        dt_size = len(dataloader_train.dataset)
        epoch_loss = 0
        step = 0
        for x, y in dataloader_train:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            show_image = inputs* 255.
            viz.img(name='images', img_=show_image[0, :, :, :])
            viz.img(name='labels', img_=labels[0, :, :, :])
            viz.img(name='prediction', img_=outputs[0, :, :, :])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
           
            print("%d/%d,train_loss:%0.4f" % (step, (dt_size - 1) // dataloader_train.batch_size + 1, loss.item()))
            viz.plot('loss', loss.item())

        print("epoch %d loss:%0.4f" % (epoch, epoch_loss))

        if bool(epoch%3) is False:

            R = test(model,dataloader_test)
            AUC, Acc, Sen, Spe, DC, IOU, MSE, NRMSE, Rand_error, Split_error, TICerror, HausDist = R

            model.train(mode=True)

            viz.plot('Test AUC', AUC)
            viz.plot('Test ACC', Acc)
            viz.plot('Test SEN', Sen)
            viz.plot('Test SPE', Spe)
            viz.plot('Test DC', DC)
            viz.plot('Test IOU', IOU)

            viz.plot('Test MSE', MSE)
            viz.plot('Test NRMSE', NRMSE)
            viz.plot('Test Rand_error', Rand_error)
            viz.plot('Test Split_error', Split_error)
            viz.plot('Test TICerror', TICerror)
            viz.plot('Test HausDrof', HausDist)

        save_path0 = Config.saveFileName
        isExists = os.path.exists(save_path0)
        if not isExists:
            os.makedirs(save_path0)
        save_path = os.path.join(save_path0, 'state-{}-{}-{}.pth'.format(epoch + 1, i + 1,test_acc))
        
        if DC >= 0.85:
            torch.save(net, save_path)
        schedulers.step()



if __name__ == '__main__':
    eng = matlab.engine.start_matlab()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    viz = Visualizer(env=Config.ENV_NAME)

    train_dataloader = DataLoader(IrisDataset(Config.irisPath, isTraining=True), batch_size=Config.train_batch_size, num_workers=16, shuffle=True)
    test_dataloader = DataLoader(IrisDataset(Config.irisPath, isTraining=False), batch_size=1, num_workers=16, shuffle=True)

    model = Our_WRB(1, 1).to(device)

    if Config.isParalleling:
        model = nn.DataParallel(model, device_ids=[0,1])

    train(model,train_dataloader,test_dataloader)

    eng.quit()



