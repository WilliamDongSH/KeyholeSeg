import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from visual_loss import Visualizer
from torchnet import meter
import csv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import CenterCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import BASNet

import pytorch_ssim
import pytorch_iou

# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)
vis = Visualizer(env='my_wind')
loss_meter = meter.AverageValueMeter()
loss_tar_meter = meter.AverageValueMeter()
loss_meter_t1 = meter.AverageValueMeter()
loss_tar_meter_t1 = meter.AverageValueMeter()
loss_meter_t2 = meter.AverageValueMeter()
loss_tar_meter_t2 = meter.AverageValueMeter()


def bce_ssim_loss(pred,target):

	bce_out = bce_loss(pred,target)
	ssim_out = 1 - ssim_loss(pred,target)
	iou_out = iou_loss(pred,target)
	loss = bce_out + ssim_out + iou_out

	return loss

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, d7, labels_v):

	loss0 = bce_ssim_loss(d0,labels_v)
	loss1 = bce_ssim_loss(d1,labels_v)
	loss2 = bce_ssim_loss(d2,labels_v)
	loss3 = bce_ssim_loss(d3,labels_v)
	loss4 = bce_ssim_loss(d4,labels_v)
	loss5 = bce_ssim_loss(d5,labels_v)
	loss6 = bce_ssim_loss(d6,labels_v)
	loss7 = bce_ssim_loss(d7,labels_v)
	#ssim0 = 1 - ssim_loss(d0,labels_v)

	# iou0 = iou_loss(d0,labels_v)
	#loss = torch.pow(torch.mean(torch.abs(labels_v-d0)),2)*(5.0*loss0 + loss1 + loss2 + loss3 + loss4 + loss5) #+ 5.0*lossa
	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7#+ 5.0*lossa
	print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data[0],loss1.data[0],loss2.data[0],loss3.data[0],loss4.data[0],loss5.data[0],loss6.data[0]))
	# print("BCE: l1:%3f, l2:%3f, l3:%3f, l4:%3f, l5:%3f, la:%3f, all:%3f\n"%(loss1.data[0],loss2.data[0],loss3.data[0],loss4.data[0],loss5.data[0],lossa.data[0],loss.data[0]))

	return loss0, loss

def fill_label_image_list(data_dir, label_dir, label_ext, img_name_list):
    lbl_name_list = []
    for img_path in img_name_list:
	    img_name = img_path.split("/")[-1]
        
	    aaa = img_name.split(".")
	    bbb = aaa[0:-1]
	    imidx = bbb[0]
	    for i in range(1,len(bbb)):
		    imidx = imidx + "." + bbb[i]
	    lbl_name_list.append(data_dir + label_dir + imidx + label_ext)

    return lbl_name_list

# ------- 2. set the directory of training dataset --------

data_dir = './train_data/'
tra_image_dir = 'images/BASNet-Train-Input-PairC/'
tra_label_dir = 'labels/BASNet-Train-Output-PairC/'

image_ext = '.png'
label_ext = '.png'

tes_data_dir = './test_data/'
tes_image_dir_1 = 'test_images/BASNet-Test-Input-PairC-113/'
tes_label_dir_1 = 'test_results/BASNet-Test-gTru-PairC-113/'
tes_image_dir_2 = 'test_images/BASNet-Test-Input-PairC-118/'
tes_label_dir_2 = 'test_results/BASNet-Test-gTru-PairC-118/'

model_dir = "./saved_models/basnet_bsi/"


epoch_num = 100000
batch_size_train = 1 
batch_size_val = 1
train_num = 0
val_num = 0

tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)
tes_img_name_list_1 = glob.glob(tes_data_dir + tes_image_dir_1 + '*' + image_ext)
tes_img_name_list_2 = glob.glob(tes_data_dir + tes_image_dir_2 + '*' + image_ext)

tra_lbl_name_list = fill_label_image_list(data_dir, tra_label_dir, label_ext, tra_img_name_list)
tes_lbl_name_list_1 = fill_label_image_list(tes_data_dir, tes_label_dir_1, label_ext, tes_img_name_list_1)
tes_lbl_name_list_2 = fill_label_image_list(tes_data_dir, tes_label_dir_2, label_ext, tes_img_name_list_2)

print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("test images 1: ", len(tes_img_name_list_1))
print("test labels 1: ", len(tes_lbl_name_list_1))
print("test images 2: ", len(tes_img_name_list_2))
print("test labels 2: ", len(tes_lbl_name_list_2))
print("---")

train_num = len(tra_img_name_list)
test_num_1 = len(tes_img_name_list_1)
test_num_2 = len(tes_img_name_list_2)

salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(256),
        RandomCrop(224),
        ToTensorLab(flag=0)]))
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)
salobj_dataset_1 = SalObjDataset(
    img_name_list=tes_img_name_list_1,
    lbl_name_list=tes_lbl_name_list_1,
    transform=transforms.Compose([
        RescaleT(256),
        ToTensorLab(flag=0)]))
salobj_dataloader_1 = DataLoader(salobj_dataset_1, batch_size=batch_size_val, shuffle=False, num_workers=1)
salobj_dataset_2 = SalObjDataset(
    img_name_list=tes_img_name_list_2,
    lbl_name_list=tes_lbl_name_list_2,
    transform=transforms.Compose([
        RescaleT(256),
        ToTensorLab(flag=0)]))
salobj_dataloader_2 = DataLoader(salobj_dataset_2, batch_size=batch_size_val, shuffle=False, num_workers=1)

#logging
csvtitle = ['Iterations', 'Epoches', 'Train Fusion Loss', 'Train Target Loss', 'Test 1 Fusion Loss', 'Test 1 Target Loss', 'Test 2 Fusion Loss', 'Test 2 Target Loss']
with open("basnet_train_log.csv", 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(csvtitle)
#csvtitle_tes = ['Iterations', 'Test 1 Fusion Loss', 'Test 1 Target Loss', 'Test 2 Fusion Loss', 'Test 2 Target Loss']
#with open("basnet_val_log.csv", 'w') as csvfile:
#    csvwriter = csv.writer(csvfile)
#    csvwriter.writerow(csvtitle_tes)

# ------- 3. define model --------
# define the net
net = BASNet(3, 1)
if torch.cuda.is_available():
    net.cuda()

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# ------- 5. training process --------
print("---start training...")
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
testing_loss = 0.0
ite_num4val = 0
test_loss_1 = 0.0
test_loss_2 = 0.0
test_tar_loss_1 = 0.0
test_tar_loss_2 = 0.0
ite_num4tes1 = 0
ite_num4tes2 = 0

for epoch in range(0, epoch_num):
    net.train()

    for i, data in enumerate(salobj_dataloader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs, labels = data['image'], data['label']

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                        requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        # y zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6, d7 = net(inputs_v)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, d7, labels_v)

        loss.backward()
        optimizer.step()

        # # print statistics
        running_loss += loss.data[0]
        running_tar_loss += loss2.data[0]

        # # visualization
        loss_meter.add((running_loss/ite_num4val).item())
        loss_tar_meter.add((running_tar_loss / ite_num4val).item())
        vis.plot_many_stack({'train_loss': loss_meter.value()[0]})
        vis.plot_many_stack({'train_tar_loss': loss_tar_meter.value()[0]})

        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, d7, loss2, loss

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
        epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

    for i, data in enumerate(salobj_dataloader_1):
        
        ite_num4tes1 = ite_num4tes1 + 1
        
        inputs, labels = data['image'], data['label']
        
        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                        requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        d0, d1, d2, d3, d4, d5, d6, d7 = net(inputs_v)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, d7, labels_v)

        # # print statistics
        test_loss_1 += loss.data[0]
        test_tar_loss_1 += loss2.data[0]

        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, d7, loss2, loss

        print("!getting test loss")

    for i, data in enumerate(salobj_dataloader_2):
        
        ite_num4tes2 = ite_num4tes2 + 1
        
        inputs, labels = data['image'], data['label']
        
        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                        requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        d0, d1, d2, d3, d4, d5, d6, d7 = net(inputs_v)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, d7, labels_v)

        # # print statistics
        test_loss_2 += loss.data[0]
        test_tar_loss_2 += loss2.data[0]

        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, d7, loss2, loss

        print("!getting test loss")

    # # logging
    csvlog = [str(ite_num), str(epoch), (running_loss/ite_num4val).item(), (running_tar_loss / ite_num4val).item(), (test_loss_1/ite_num4tes1).item(), (test_tar_loss_1/ite_num4tes1).item(), (test_loss_2/ite_num4tes2).item(), (test_tar_loss_2/ite_num4tes2).item()]
    with open("basnet_train_log.csv", 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(csvlog)

    # # visualization
    loss_meter_t1.add((test_loss_1/ite_num4tes1).item())
    loss_tar_meter_t1.add((test_tar_loss_1/ite_num4tes1).item())
    vis.plot_many_stack({'test_loss_1': loss_meter_t1.value()[0]})
    vis.plot_many_stack({'test_tar_loss_1': loss_tar_meter_t1.value()[0]})
    loss_meter_t2.add((test_loss_2/ite_num4tes2).item())
    loss_tar_meter_t2.add((test_tar_loss_2/ite_num4tes2).item())
    vis.plot_many_stack({'test_loss_2': loss_meter_t2.value()[0]})
    vis.plot_many_stack({'test_tar_loss_2': loss_tar_meter_t2.value()[0]})

    if ite_num % 6000 == 0:  # save model every 2000 iterations

        torch.save(net.state_dict(), model_dir + "basnet_bsi_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
        running_loss = 0.0
        running_tar_loss = 0.0
        test_loss_1 = 0.0
        test_tar_loss_1 = 0.0
        test_loss_2 = 0.0
        test_tar_loss_2 = 0.0
        net.train()  # resume train
        ite_num4val = 0
        ite_num4tes1 = 0
        ite_num4tes2 = 0



print('-------------Congratulations! Training Done!!!-------------')
