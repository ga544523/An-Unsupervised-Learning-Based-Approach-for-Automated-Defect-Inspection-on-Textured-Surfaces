import numpy as np
import os
import PIL
import PIL.Image
import pathlib
import pandas as pd
import csv
import gc

'''
paths='/home/ipx/data/append'
cons=4600
def batch_rename(path):
    for fname in os.listdir(path):
        new_fname=fname.split('.')
        print(new_fname)
        os.rename(os.path.join(path, fname), os.path.join(path, str( (int)(new_fname[0])-1151+cons) )+'.png' )

def batch_renamelabel(path):
    for fname in os.listdir(path):
        new_fname=fname.split('_')
        print(new_fname)
        os.rename(os.path.join(path, fname), os.path.join(path, str( (int)(new_fname[0])-1151+cons) )+'_'+new_fname[1] )


#batch_renamelabel(paths)
#print(1/0)


f = open('/home/ipx/data/totallabel/Labels.txt', 'r')
lines=f.readlines() # 讀取檔案內容的每一行文字為陣列
for line in lines:
    x = line.split()
    print(x)
    with open('output.csv', 'a+', newline='') as csvfile:
        # 建立 CSV 檔寫入器
        writer = csv.writer(csvfile)
        # 寫入一列資料
        writer.writerow([ str(int(x[0])-1151+cons )+'.png', x[1] ])
f.close() # 關閉檔案
'''
def show_img(img):
    global showcnt
    cv2.imshow('My Image'+str(showcnt), img)
    showcnt=showcnt+1


showcnt=0
def showtensor(img,size):
    global showcnt
    showblur = img.to(devicecpu)
    showblur = showblur.view(size, size)
    showblur = showblur.detach().numpy()
    cv2.imshow('My Image'+str(showcnt), showblur)
    showcnt=showcnt+1


import torch.nn.functional as F
import pandas as pd
from random import shuffle
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
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
from multiprocessing import Pool
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F

device1 = torch.device("cuda:0")
devicecpu = torch.device("cpu")
directory = '/home/ipx/venv/'
df = pd.read_csv(directory + "output.csv")
data = []

print(torch.cuda.is_available())
print(torch.__version__)
'''
with open('/home/ipx/venv/output.csv', newline='') as csvfile:

  # 讀取 CSV 檔案內容
  rows = csv.reader(csvfile)

  # 寫入一列資料
  with open('/home/ipx/venv/output1.csv', 'a+', newline='') as csvfile1:
      writer1 = csv.writer(csvfile1)
      # 以迴圈輸出每一列
      cnt=0
      cla=0
      cona=575
      conb=1150
      for row in rows:
        data.append(row)
        print(row)
        writer1.writerow([row[0], row[1],cla])
        cnt=cnt+1
        cla+=int(cnt/cona)
        cnt%=cona

        if(cla==6):
            cona=1150
'''
with open('/home/ipx/venv/output.csv', newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        data.append(row)

shuffle(data)
imgsize = 512

totalimagedir = '/home/ipx/data/totalimage/'
totallabeldir = '/home/ipx/data/totallabel/'

batchsize = 128

modelchannel=800

print_image_interval=6400
patchsize=8
P=0.8


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


trainmode = 4

encoder_decoder512_net= encoder_decoder512()
encoder_decoder512_net= encoder_decoder512_net.to(device1)

encoder_decoder256_net= encoder_decoder256()
encoder_decoder256_net= encoder_decoder256_net.to(device1)

encoder_decoder128_net= encoder_decoder128()
encoder_decoder128_net= encoder_decoder128_net.to(device1)


encoder_decoder512_optimizer = torch.optim.Adam(encoder_decoder512_net.parameters(), lr=0.0001)
encoder_decoder256_optimizer = torch.optim.Adam(encoder_decoder256_net.parameters(), lr=0.0001)
encoder_decoder128_optimizer = torch.optim.Adam(encoder_decoder128_net.parameters(), lr=0.0001)

#encoder_optimizer = torch.optim.SGD(encoder_net.parameters(), lr=0.0001, momentum=0.9)
#decoder_optimizer = torch.optim.SGD(decoder_net.parameters(), lr=0.0001, momentum=0.9)

drawx = []
drawy = []

count_update = 0

drawx1 = []
drawy1 = []
drawx2 = []
drawy2 = []
drawx3 = []
drawy3 = []

count_update1 = 0
count_update2 = 0
count_update3 = 0

def custom_decoder_loss(output, totalimage,currentstep):
    global count_update1
    global count_update2
    global count_update3
    global drawy1
    global drawx1

    sum = 0
    if (count_update1 % print_image_interval == 0):
        print(output.size())

    for b in range(0, batchsize, 1):
        dif = (output[b][0] - totalimage[b][0])**2
        dif=torch.sum(dif)
        #dif=torch.abs(dif)

        sum += torch.sqrt(dif)
    #sum /= imgsize * imgsize
    sum /= batchsize
    plt.clf()
    if(currentstep==0):
        if(count_update1%print_image_interval==0):
            print('decoder_loss_512=', sum)
        drawx1.append(count_update1)
        drawy1.append(sum.item())
        count_update1 += 1

        if (count_update1 % print_image_interval == 0):
            plt.plot(drawx1, drawy1)
            plt.savefig('decoder_loss_512 ' + str(count_update1) + '.png')

    if (currentstep == 1):
        if (count_update2 % print_image_interval == 0):
            print('decoder_loss_256=', sum)
        drawx2.append(count_update2)
        drawy2.append(sum.item())
        count_update2 += 1

        if (count_update2 % print_image_interval == 0):
            plt.plot(drawx2, drawy2)
            plt.savefig('decoder_loss_256 ' + str(count_update2) + '.png')

    if (currentstep == 2):

        if (count_update3 % print_image_interval == 0):
            print('decoder_loss_128=', sum)

        drawx3.append(count_update3)
        drawy3.append(sum.item())
        count_update3 += 1

        if (count_update3 % print_image_interval == 0):
            plt.plot(drawx3, drawy3)
            plt.savefig('decoder_loss_128 ' + str(count_update3) + '.png')


    return sum




def train_decoder(traindata_non_defect,traindata_original,currentstep):

    if(currentstep==0):
        encoder_decoder512_optimizer.zero_grad()
        size = (512, 512)
        transform = transforms.Compose([
            transforms.Resize(size),
        ])
        traindata_non_defect1 = transform(traindata_non_defect)
        traindata_original1=transform(traindata_original)

    if (currentstep == 1):
        encoder_decoder256_optimizer.zero_grad()
        size = (256, 256)
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.GaussianBlur(3, 3)
        ])
        traindata_non_defect1 = transform(traindata_non_defect)
        for i in range(0,256,1):
            for j in range(0,256,1):
                noise = random.random()
                if(noise>P):
                    traindata_non_defect1[0][0][i][j]=1

        traindata_original1=transform(traindata_original)

    if (currentstep == 2):
        encoder_decoder128_optimizer.zero_grad()
        size = (128, 128)
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.GaussianBlur(3, 3),
            transforms.GaussianBlur(3, 3)
        ])
        traindata_non_defect1 = transform(traindata_non_defect)
        for i in range(0,128,1):
            for j in range(0,128,1):
                noise = random.random()
                if(noise>P):
                    traindata_non_defect1[0][0][i][j]=1
        traindata_original1=transform(traindata_original)

    size=traindata_non_defect1.size(2)

    total_patchimg=0
    total_compareimg=0
    flag=0
    for b in range(0,batchsize,1):
        patchimg = torch.ones((patchsize, patchsize), dtype=torch.float32)
        compareimg = torch.ones((patchsize, patchsize), dtype=torch.float32)
        ii=random.randint(0, size-patchsize)
        jj=random.randint(0, size-patchsize)
        for i in range(0,patchsize,1):
            for j in range(0,patchsize,1):

                patchimg[i][j]=traindata_non_defect1[0][0][ii+i][jj+j]
                compareimg[i][j]=traindata_original1[0][0][ii+i][jj+j]

        patchimg=patchimg.view(1,1,patchsize,patchsize)
        compareimg = compareimg.view(1, 1, patchsize, patchsize)

        if(flag==0):
            total_patchimg=patchimg
            total_compareimg = compareimg
            flag=1
        else:
            total_patchimg=torch.cat((total_patchimg, patchimg), 0)
            total_compareimg = torch.cat((total_compareimg, compareimg), 0)

    if (currentstep == 0):
        output = encoder_decoder512_net(total_patchimg.to(device1))
        output = output.to(device1)
    if (currentstep == 1):
        output = encoder_decoder256_net(total_patchimg.to(device1))
        output = output.to(device1)
    if (currentstep == 2):
        output = encoder_decoder128_net(total_patchimg.to(device1))
        output = output.to(device1)

    total_compareimg=total_compareimg.to(device1)

    if(count_update1%print_image_interval==0):
        torchvision.utils.save_image(output, '/home/ipx/result/test'+'type1  '+ str(count_update1) + '.png',normalize=False)
    elif (count_update2%print_image_interval==0):
        torchvision.utils.save_image(output, '/home/ipx/result/test' + 'type2  ' + str(count_update2) + '.png',normalize=False)
    elif (count_update3 % print_image_interval == 0):
        torchvision.utils.save_image(output, '/home/ipx/result/test' + 'type3  ' + str(count_update3) + '.png',normalize=False)

    #saveimage = output[0][0].clone().detach()
    #saveimage *= 255
    #print(saveimage)
    #saveimage = saveimage.to(devicecpu)
    #na = saveimage.detach().numpy()
    #cv2.imwrite('/home/ipx/result/output' + str(count_update1) + '.png', na)

    decoder_loss = custom_decoder_loss(output, total_compareimg,currentstep)

    return decoder_loss


def show_grad(grad):
    k=0
    # print('1111111111111111')
    grad = torch.clamp(grad, max=3, min=-3)
    #print(torch.max(grad))
    #print(torch.min(grad))
    # print('22222222222222222')



for p in encoder_decoder512_net.parameters():
    # print(p.size())
    p.register_hook(show_grad)

for p in encoder_decoder256_net.parameters():
    # print(p.size())
    p.register_hook(show_grad)

for p in encoder_decoder128_net.parameters():
    # print(p.size())
    p.register_hook(show_grad)



for t in range(0, 1000000, 1):
    shuffle(data)

    print('epoch ', t)
    positive = []
    negative = []

    for i in range(0, len(data), 1):
        if (data[i][1] == '0' and int(data[i][2]) - int('0') == int(trainmode)):
            positive.append(data[i][0])
        else:
            1

    print(len(positive))

    # get defect
    for j in range(0, len(positive) , 1):
        traindata_non_defect = 0

        traindata_original =0
        flag = 0
        flag1=0

        s = positive[j]
        img = cv2.imread(totalimagedir + s, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, dsize=(imgsize, imgsize))
        img = img.astype(np.float32)
        imgdefect=img.copy()

        tensorimg = torch.from_numpy(img)
        tensorimg/=255.0
        tensorimg = tensorimg.view(1, 1, imgsize, imgsize)
        if (flag1 == 0):
            traindata_original = tensorimg
            flag1 = 1
        else:
            traindata_original = torch.cat((traindata_original, tensorimg), 0)


        for ni in range(0, imgsize, 1):
            for nj in range(0, imgsize, 1):
                noise=random.random()
                if(noise>P):
                    imgdefect[ni][nj]=255.0
                imgdefect[ni][nj] /= 255.0


        tensorimg = torch.from_numpy(imgdefect)

        tensorimg = tensorimg.view(1, 1, imgsize, imgsize)

        if (flag == 0):
            traindata_non_defect = tensorimg
            flag = 1
        else:
            traindata_non_defect = torch.cat((traindata_non_defect, tensorimg), 0)


        del img
        del imgdefect

        for step in range(0,3,1):
            decoder_loss=train_decoder(traindata_non_defect,traindata_original,step)
            decoder_loss.backward()
            if(step==0):
                #torch.nn.utils.clip_grad_norm_(encoder_decoder512_net.parameters(), max_norm=3,)
                encoder_decoder512_optimizer.step()
            if(step==1):
                #torch.nn.utils.clip_grad_norm_(encoder_decoder256_net.parameters(), max_norm=3)
                encoder_decoder256_optimizer.step()
            if(step==2):
                #torch.nn.utils.clip_grad_norm_(encoder_decoder128_net.parameters(), max_norm=3)
                encoder_decoder128_optimizer.step()


        traindata_non_defect = 0
        traindata_original = 0
    if(t%30==10):
        torch.save(encoder_decoder512_net, 'encoder_decoder512.pkl')
        torch.save(encoder_decoder256_net, 'encoder_decoder256.pkl')
        torch.save(encoder_decoder128_net, 'encoder_decoder128.pkl')

    positive.clear()
    gc.collect()