import numpy as np
import os
import PIL
import PIL.Image
import pathlib
import pandas as pd
import csv
import gc
import torch.nn.functional as F
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as transforms
import torch.utils.model_zoo as model_zoo
import cmath
import math
import torchvision.models as models
from torchvision.utils import save_image, make_grid
import random
import threading
import heapq as hq
import time
import os
from multiprocessing import Pool
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

device1 = torch.device("cuda:0")
devicecpu = torch.device("cpu")

imgsize = 512

batchsize = 1

modelchannel=800

print_mage_interval=200

patchsize=8

class encoder_decoder512(nn.Module):

    def __init__(self, num_classes=200 * 200):
        super(encoder_decoder512, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(1, modelchannel, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=True),
            nn.Conv2d(modelchannel, modelchannel, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=True),
            nn.Conv2d(modelchannel,modelchannel, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=True),
            nn.Conv2d(modelchannel, modelchannel, kernel_size=4, stride=2, padding=1),
            )
        self.decode = nn.Sequential(

            nn.ConvTranspose2d(modelchannel,modelchannel, kernel_size=3,stride=1,padding=1),
            torch.nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(modelchannel, modelchannel, kernel_size=4,stride=2,padding=1),
            torch.nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(modelchannel, modelchannel, kernel_size=4, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(modelchannel, modelchannel, kernel_size=4, stride=2, padding=2),
            torch.nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(modelchannel, 1, kernel_size=4,stride=2,padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        num = x.size(0)
        x = self.encode(x)

        x=x.view(batchsize,modelchannel,1,1)
        x=self.decode(x)
        return x

class encoder_decoder256(nn.Module):

    def __init__(self, num_classes=200 * 200):
        super(encoder_decoder256, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(1, modelchannel, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=True),
            nn.Conv2d(modelchannel, modelchannel, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=True),
            nn.Conv2d(modelchannel,modelchannel, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=True),
            nn.Conv2d(modelchannel, modelchannel, kernel_size=4, stride=2, padding=1),
            )
        self.decode = nn.Sequential(

            nn.ConvTranspose2d(modelchannel,modelchannel, kernel_size=3,stride=1,padding=1),
            torch.nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(modelchannel, modelchannel, kernel_size=4,stride=2,padding=1),
            torch.nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(modelchannel, modelchannel, kernel_size=4, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(modelchannel, modelchannel, kernel_size=4, stride=2, padding=2),
            torch.nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(modelchannel, 1, kernel_size=4,stride=2,padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        num = x.size(0)
        x = self.encode(x)
        x=x.view(batchsize,modelchannel,1,1)
        x=self.decode(x)
        return x

class encoder_decoder128(nn.Module):

    def __init__(self, num_classes=200 * 200):
        super(encoder_decoder128, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(1, modelchannel, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=True),
            nn.Conv2d(modelchannel, modelchannel, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=True),
            nn.Conv2d(modelchannel,modelchannel, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=True),
            nn.Conv2d(modelchannel, modelchannel, kernel_size=4, stride=2, padding=1),
            )
        self.decode = nn.Sequential(

            nn.ConvTranspose2d(modelchannel,modelchannel, kernel_size=3,stride=1,padding=1),
            torch.nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(modelchannel, modelchannel, kernel_size=4,stride=2,padding=1),
            torch.nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(modelchannel, modelchannel, kernel_size=4, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(modelchannel, modelchannel, kernel_size=4, stride=2, padding=2),
            torch.nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(modelchannel, 1, kernel_size=4,stride=2,padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        num = x.size(0)
        x = self.encode(x)
        x=x.view(batchsize,modelchannel,1,1)
        x=self.decode(x)
        return x


trainmode = 5
gamma=2.0

encoder_decoder512_net = torch.load('encoder_decoder512.pkl')
encoder_decoder512_net = encoder_decoder512_net.to(device1)

encoder_decoder256_net = torch.load('encoder_decoder256.pkl')
encoder_decoder256_net = encoder_decoder256_net.to(device1)

encoder_decoder128_net = torch.load('encoder_decoder128.pkl')
encoder_decoder128_net = encoder_decoder128_net.to(device1)


def getresult1(input_image):
    result1 = np.zeros((512, 512))

    size = (512, 512)
    transform = transforms.Compose([
        transforms.Resize(size),
    ])
    input_image=transform(input_image)

    total=[]
    mean=0
    deviation=0
    for ii in range(0, 512 - patchsize, patchsize):
        for jj in range(0, 512 - patchsize, patchsize):
            patchimg = torch.ones((patchsize, patchsize), dtype=torch.float32)
            for i in range(0, patchsize, 1):
                for j in range(0, patchsize, 1):
                    patchimg[i][j] = input_image[0][0][ii + i][jj + j]
            patchimg=patchimg.view(1,1,patchsize,patchsize).to(device1)
            output=encoder_decoder512_net(patchimg)
            res=torch.abs(output-patchimg)
            res=torch.sum(res)
            total.append((res.item(),ii,jj))

            mean+=res.item()

    mean/=len(total)



    for i in range(0,len(total),1):
        deviation+=(total[i][0]-mean)**2
    deviation/=len(total)
    deviation=math.sqrt(deviation)

    for i in range(0,len(total),1):
        if(total[i][0]>mean+gamma*deviation or total[i][0]<mean-gamma*deviation):
            for t1 in range(total[i][1],total[i][1]+patchsize,1):
                for t2 in range(total[i][2],total[i][2]+patchsize,1):
                    result1[t1][t2]=1

    return result1


def getresult2(input_image):
    result2 = np.zeros((256, 256))
    result2_1 = np.zeros((512, 512))
    size = (256, 256)
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.GaussianBlur(3, 3),
    ])
    input_image = transform(input_image)
    total = []
    mean = 0
    deviation = 0

    for ii in range(0, 256 - patchsize, patchsize):
        for jj in range(0, 256 - patchsize, patchsize):
            patchimg = torch.ones((patchsize, patchsize), dtype=torch.float32)
            for i in range(0, patchsize, 1):
                for j in range(0, patchsize, 1):
                    patchimg[i][j] = input_image[0][0][ii + i][jj + j]

            patchimg = patchimg.view(1, 1, patchsize, patchsize).to(device1)
            output = encoder_decoder256_net(patchimg)
            res = torch.abs(output - patchimg)
            res = torch.sum(res)
            total.append((res.item(), ii, jj))
            mean += res.item()
    mean /= len(total)
    for i in range(0, len(total), 1):
        deviation += (total[i][0] - mean) ** 2
    deviation /= len(total)
    deviation = math.sqrt(deviation)

    for i in range(0, len(total), 1):
        if (total[i][0] > mean + gamma * deviation or total[i][0] < mean -gamma * deviation):
            for t1 in range(total[i][1], total[i][1] + patchsize, 1):
                for t2 in range(total[i][2], total[i][2] + patchsize, 1):
                    result2[t1][t2] = 1
    for i in range(0,256,1):
        for j in range(0,256,1):
            if(result2[i][j]==1):
                result2_1[i*2][j*2]=1
                result2_1[i*2][j*2+1]=1
                result2_1[i*2+1][j*2]=1
                result2_1[i*2+1][j*2+1]=1

    return result2_1


def getresult3(input_image):
    result3 = np.zeros((128, 128))
    result3_1 = np.zeros((256, 256))
    result3_2=np.zeros((512, 512))
    size = (128, 128)
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.GaussianBlur(3, 3),
        transforms.GaussianBlur(3, 3)
    ])
    input_image = transform(input_image)
    total = []
    mean = 0
    deviation = 0
    for ii in range(0, 128 - patchsize,patchsize):
        for jj in range(0, 128 - patchsize, patchsize):
            patchimg = torch.ones((patchsize, patchsize), dtype=torch.float32)
            for i in range(0, patchsize, 1):
                for j in range(0, patchsize, 1):
                    patchimg[i][j] = input_image[0][0][ii + i][jj + j]
            patchimg = patchimg.view(1, 1, patchsize, patchsize).to(device1)
            output = encoder_decoder256_net(patchimg)
            res = torch.abs(output - patchimg)
            res = torch.sum(res)
            total.append((res.item(), ii, jj))
            mean += res.item()
    mean /= len(total)
    for i in range(0, len(total), 1):
        deviation += (total[i][0] - mean) ** 2
    deviation /= len(total)
    deviation = math.sqrt(deviation)

    for i in range(0, len(total), 1):
        if (total[i][0] > mean + gamma * deviation or total[i][0] < mean - gamma * deviation):
            for t1 in range(total[i][1], total[i][1] + patchsize, 1):
                for t2 in range(total[i][2], total[i][2] + patchsize, 1):
                    result3[t1][t2] = 1

    for i in range(0,128,1):
        for j in range(0,128,1):
            if(result3[i][j]==1):
                result3_1[i * 2][j * 2] = 1
                result3_1[i*2][j*2+1]=1
                result3_1[i*2+1][j*2]=1
                result3_1[i*2+1][j*2+1]=1

    for i in range(0,256,1):
        for j in range(0,256,1):
            if(result3_1[i][j]==1):
                result3_2[i * 2][j * 2] = 1
                result3_2[i*2][j*2+1]=1
                result3_2[i*2+1][j*2]=1
                result3_2[i*2+1][j*2+1]=1
    return result3_2

if __name__ == '__main__':
    paths = '/home/ipx/data/Class5/Test/'
    pathslabel='/home/ipx/data/Class5/Test/Label/'
    labelpath=os.listdir(pathslabel)


    def testing(path):
        for fname in os.listdir(path):
            print(fname)
            s = fname
            s='0148.PNG'
            img = cv2.imread(paths + s, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, dsize=(imgsize, imgsize))
            #cv2.imshow(fname, img)
            img = img.astype(np.float32)

            tensorimg = torch.from_numpy(img)
            tensorimg /= 255.0
            tensorimg = tensorimg.view(1, 1, imgsize, imgsize)
            tensorimg=tensorimg.to(device1)

            result1=getresult1(tensorimg.clone())
            result2=getresult2(tensorimg.clone())
            result3=getresult3(tensorimg.clone())

            final_result=np.zeros((512, 512) )

            for i in range(0,512,1):
                for j in range(0,512,1):
                    if( (result1[i][j]==1 and result2[i][j]==1) or (result2[i][j]==1 and result3[i][j]==1) ):
                        final_result[i][j]=1
                        img[i][j]=1


            cv2.imshow('result', img)
            cv2.waitKey(0)
            del img

    testing(paths)
    print(1/0)


