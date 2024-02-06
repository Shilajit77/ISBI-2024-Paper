
import os
import glob
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
import os
import PIL
#import tensorflow as tf
import torch
import warnings
warnings.filterwarnings("ignore")
import torch
import torchvision
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
# %matplotlib inline
import os
import torch
import torchvision
#import tarfile
#from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torchvision.models as models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

class DataGenerator(Dataset):

	def __init__(self, img_dir, split_file, transform):

		self.img_name_list = []
		self.img_label_list = []
		self.transform = transform
		# self.img_directory = img_dir

		with open(split_file, 'r') as split_name:
			img_and_label_list = split_name.readlines()[1:]

		for index in img_and_label_list:
			img_path = glob.glob(os.path.join(img_dir,'**/',index.split(",")[0]), recursive=True)[0]
			arr = index.split(",")
			length = len(arr)-2
			img_label = np.zeros(length)
			for i in range(length):
			    img_label[i] = int(arr[i+2])
			# img_label = (index.split(",")[1:])
			self.img_name_list.append(img_path)
			self.img_label_list.append(img_label)

	def __getitem__(self, index):

		img_path = self.img_name_list[index]
		# img_path = os.path.join(img_dir, img_path)
		img_name = os.path.basename(img_path)
		image_data = Image.open(img_path).convert('RGB')
		image_data = self.transform(image_data)
		# image_label= torch.FloatTensor(self.img_label_list[index])
		image_label= self.img_label_list[index]

		return (image_data, image_label, img_name)

	def __len__(self):

		return len(self.img_name_list)


import torchvision.transforms as transforms

composed_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])



train_loader1 = torch.load('/home/shilajit/newnih/train_loader.pt')
val_loader = torch.load('/home/shilajit/newnih/val_loader.pt')

batch_size = 8
train_loader =  DataLoader(train_loader1, batch_size=batch_size, shuffle=True,drop_last=True)
train_loader2 =  DataLoader(train_loader1, batch_size=16, shuffle=True,drop_last=True)
#test_loader = DataLoader(test_loader, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_loader, batch_size=batch_size, shuffle=False)

import torch
import torchvision.models as models

tmodel = models.densenet121(pretrained=True)
num_ftrs = tmodel.classifier.in_features
print(num_ftrs)

tmodel.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(inplace=1),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=1),
            nn.Linear(1024,512),
            nn.ReLU(inplace=1),
            nn.Linear(512,15),
            nn.Sigmoid()
        )

optimizer2 = torch.optim.Adam(tmodel.parameters(), lr=2e-5,betas=(0.5,0.99))
import time
from timeit import default_timer as timer
tmodel = tmodel.cuda()
print('train started')

tmodel.load_state_dict(torch.load('best_teacher.pth'))

import torch
from torch.utils.data import DataLoader, Subset



train_dataset = train_loader.dataset

# Calculate the total number of images in the dataset
total_images = len(train_dataset)

# Shuffle the indices to create a random order
shuffled_indices = np.random.permutation(total_images)

# Calculate the sizes of the four subsets
subset_size = total_images // 4

# Use the Subset class to create four subsets of the dataset based on the shuffled indices
part1_indices = shuffled_indices[:subset_size]
part2_indices = shuffled_indices[subset_size:2 * subset_size]
part3_indices = shuffled_indices[2 * subset_size:3 * subset_size]
part4_indices = shuffled_indices[3 * subset_size:]

part1_dataset = Subset(train_dataset, part1_indices)
part2_dataset = Subset(train_dataset, part2_indices)
part3_dataset = Subset(train_dataset, part3_indices)
part4_dataset = Subset(train_dataset, part4_indices)

# Create four DataLoader objects for each subset
batch_size = train_loader.batch_size
sloader1 = DataLoader(part1_dataset, batch_size=batch_size, shuffle=True)
sloader2 = DataLoader(part2_dataset, batch_size=batch_size, shuffle=True)
sloader3 = DataLoader(part3_dataset, batch_size=batch_size, shuffle=True)
sloader4 = DataLoader(part4_dataset, batch_size=batch_size, shuffle=True)



# Pre-trained MobileNet model
smodel1 = models.mobilenet_v2(pretrained=True)
# Pre-trained ShuffleNet model
smodel2 = models.shufflenet_v2_x1_0(pretrained=True)
smodel3 = models.mobilenet_v2(pretrained=True)
# Pre-trained ShuffleNet model
smodel4 = models.shufflenet_v2_x1_0(pretrained=True)

num_ftrs = smodel1.classifier[1].in_features
print(num_ftrs)
print('mnet')
smodel1.classifier =  nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(inplace=1),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=1),
            nn.Linear(1024,15)
            #nn.Sigmoid()
        )


optimizer1 = torch.optim.Adam( smodel1.parameters(),lr=2e-4,betas=(0.5,0.99))
smodel1 = smodel1.cuda()

num_ftrs = smodel3.classifier[1].in_features
print(num_ftrs)
smodel3.classifier =  nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(inplace=1),
            nn.Linear(1024,512),
            nn.ReLU(inplace=1),
            nn.Linear(512,15)
            #nn.Sigmoid()
        )


optimizer2 = torch.optim.Adam(smodel3.parameters(),lr=2e-4,betas=(0.5,0.99))
smodel3 = smodel3.cuda()

in_ftrs = smodel2.fc.in_features
smodel2.fc = nn.Sequential(
            nn.Linear(in_ftrs, 1024),
            nn.ReLU(inplace=1),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=1),
            nn.Linear(1024,15)
            #nn.Sigmoid()
)
in_ftrs = smodel4.fc.in_features
print('snet: ',in_ftrs)
smodel4.fc = nn.Sequential(
            nn.Linear(in_ftrs, 1024),
            nn.ReLU(inplace=1),
            nn.Linear(1024,512),
            nn.ReLU(inplace=1),
            nn.Linear(512,15)
            #nn.Sigmoid()
)

optimizer3 = torch.optim.Adam(smodel2.parameters(),lr=2e-5,betas=(0.5,0.99))
optimizer4 = torch.optim.Adam(smodel4.parameters(),lr=2e-5,betas=(0.5,0.99))
smodel2 = smodel2.cuda()
smodel4 = smodel4.cuda()

import torch
import torch.nn as nn

def mnettap(model,x):
    x = model.features(x)
    x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
    x = torch.flatten(x, 1)
    #print(model.classifier[0])
    x = model.classifier[0](x)
    x = model.classifier[1](x)
    x = model.classifier[2](x)
    return x



def snettap(model,x):
    x = model.conv1(x)
    x = model.maxpool(x)
    #print(model.classifier[0])
    x = model.stage2(x)
    x = model.stage3(x)
    x = model.stage4(x)
    x = model.conv5(x)
    x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
    x = torch.flatten(x,1)
    x = model.fc[0](x)
    x = model.fc[1](x)
    x = model.fc[2](x)
    return x

import torch
import torch.nn as nn

def model_tap1_r152(tmodel,x):
    x = tmodel.conv1(x)
    x = tmodel.bn1(x)
    x = tmodel.relu(x)
    x = tmodel.maxpool(x)
    x = tmodel.layer1(x)
    x = tmodel.layer2(x)
    x = tmodel.layer3(x)
    x = tmodel.layer4(x)
    x = tmodel.avgpool(x)
    x = torch.flatten(x, 1)
    x = tmodel.classifier[0](x)
    x = tmodel.classifier[1](x)
    x = tmodel.classifier[2](x)
    return x
def model_tap2_r152(tmodel,x):
    x = tmodel.conv1(x)
    x = tmodel.bn1(x)
    x = tmodel.relu(x)
    x = tmodel.maxpool(x)
    x = tmodel.layer1(x)
    x = tmodel.layer2(x)
    x = tmodel.layer3(x)
    x = tmodel.layer4(x)
    x = tmodel.avgpool(x)
    x = torch.flatten(x, 1)
    x = tmodel.classifier[0](x)
    x = tmodel.classifier[1](x)
    x = tmodel.classifier[2](x)
    x = tmodel.classifier[3](x)
    x = tmodel.classifier[4](x)
    return x

for name, child in tmodel.named_children():
    print(name)

def model_tap1_d121(tmodel,x):
    x = tmodel.features(x)
    x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
    x = torch.flatten(x, 1)
    x = tmodel.classifier[0](x)
    x = tmodel.classifier[1](x)
    x = tmodel.classifier[2](x)
    return x
def model_tap2_d121(tmodel,x):
    x = tmodel.features(x)
    x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
    x = torch.flatten(x, 1)
    x = tmodel.classifier[0](x)
    x = tmodel.classifier[1](x)
    x = tmodel.classifier[2](x)
    x = tmodel.classifier[3](x)
    x = tmodel.classifier[4](x)
    return x
def tmodel_output(tmodel,x):
    x = tmodel.features(x)
    x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
    x = torch.flatten(x, 1)
    x = tmodel.classifier[0](x)
    x = tmodel.classifier[1](x)
    x = tmodel.classifier[2](x)
    x1 = tmodel.classifier[3](x)
    x1 = tmodel.classifier[4](x1)
    x1 = tmodel.classifier[5](x1)
    x1 = tmodel.classifier[6](x1)
    return x1


def m1losss(s1output,labels,img1,tmodel,smodel1,toutput):
    ss1 = F.log_softmax(s1output,dim=1) #S1 output
    tt = F.softmax(toutput,dim=1)
    s1_norm_loss = nn.BCELoss()(torch.sigmoid(s1output),labels.float())
    compare = model_tap1_d121(tmodel,img1)
       
    comp_stud = mnettap(smodel1,img1)
    soft_loss_teacher1 = nn.KLDivLoss(reduction='batchmean')(ss1, tt)
    soft_loss_teacher = nn.MSELoss()(comp_stud,compare)

    lloss = (s1_norm_loss +soft_loss_teacher  +soft_loss_teacher1)
    #print(lloss)
    return lloss

def m2losss(s1output,labels,img1,tmodel,smodel1,toutput):
    ss1 = F.log_softmax(s1output,dim=1) #S1 output
    tt = F.softmax(toutput,dim=1)
    s1_norm_loss = nn.BCELoss()(torch.sigmoid(s1output),labels.float())
    compare = model_tap2_d121(tmodel,img1)
            #print(compare.shape)
    comp_stud = mnettap(smodel1,img1)
    #logg = torch.log(s1output)
    soft_loss_teacher1 = nn.KLDivLoss(reduction='batchmean')(ss1, tt)
    soft_loss_teacher = nn.MSELoss()(comp_stud,compare)

    lloss = (s1_norm_loss +soft_loss_teacher + soft_loss_teacher1)
    return lloss
def sh1losss(s1output,labels,img1,tmodel,smodel1,toutput):
    ss1 = F.log_softmax(s1output,dim=1) #S1 output
    tt = F.softmax(toutput,dim=1)
    fact = 1.0
    s1_norm_loss = fact*nn.BCELoss()(torch.sigmoid(s1output),labels.float())
    compare = model_tap1_d121(tmodel,img1)
            #print(compare.shape)
    comp_stud = snettap(smodel1,img1)
    #logg = torch.log(s1output)
    soft_loss_teacher1 = fact*nn.KLDivLoss(reduction='batchmean')(ss1, tt)
    soft_loss_teacher = fact*nn.MSELoss()(comp_stud,compare)
    lloss = (s1_norm_loss +soft_loss_teacher + soft_loss_teacher1)

    return lloss
def sh2losss(s1output,labels,img1,tmodel,smodel1,toutput):
    ss1 = F.log_softmax(s1output,dim=1) #S1 output  F.log_softmax
    tt = F.softmax(toutput,dim=1)
    fact = 1.0
    s1_norm_loss = fact*nn.BCELoss()(torch.sigmoid(s1output),labels.float())
    compare = model_tap2_d121(tmodel,img1)
            #print(compare.shape)
    comp_stud = snettap(smodel1,img1)
    #logg = torch.log(s1output)
    soft_loss_teacher1 = fact*nn.KLDivLoss(reduction='batchmean')(ss1, tt)
    soft_loss_teacher = fact*nn.MSELoss()(comp_stud,compare)

    lloss = (s1_norm_loss +soft_loss_teacher + soft_loss_teacher1)
    #print(lloss)
    return lloss




print('ok!!!!!')
m1tl = []
m1vl = []
m1tac = []
m1valac = []
m2tl = []
m2vl = []
m2tac = []
m2valac = []
m3tl = []
m3vl = []
m3tac = []
m3valac = []
m4tl = []
m4vl = []
m4tac = []
m4valac = []
best1 = 0.0
best2 = 0.0
best3 = 0.0
best4 = 0.0
for i in range(200):
        tl1 = []
        vl1 = []
        tacc1 = []
        val_acc1 = []
        tl2 = []
        vl2 = []
        tacc2 = []
        val_acc2 = []
        tl3 = []
        vl3 = []
        tacc3 = []
        val_acc3 = []
        tl4 = []
        vl4 = []
        tacc4 = []
        val_acc4 = []
        predicted_probs_list1 = []
        true_labels_list1 = []
        predicted_probs_list2 = []
        true_labels_list2 = []
        predicted_probs_list3 = []
        true_labels_list3 = []
        predicted_probs_list4 = []
        true_labels_list4 = []
        for index,data in enumerate(zip(sloader1,sloader2,sloader3,sloader4,train_loader2)):
            img1,labels1,img2,labels2,img3,labels3,img4,labels4,img,lbl = data[0][0],data[0][1],data[1][0],data[1][1],data[2][0],data[2][1],data[3][0],data[3][1],data[4][0],data[4][1]
            img1 = img1.to(device)
            labels1 = labels1.to(device)
            img2 = img2.to(device)
            labels2 = labels2.to(device)
            img3 = img3.to(device)
            labels3 = labels3.to(device)
            img4 = img4.to(device)
            labels4 = labels4.to(device)
            img = img.to(device)
            lbl = lbl.to(device)
            #print(img[0])

            #S1:

            s1output = smodel1(img1)
            #s2output = smodel2(img1)
            #s3output = smodel3(img1)
            #s4output = smodel4(img1)
            toutput = tmodel_output(tmodel,img1)
            s1_loss = m1losss(s1output,labels1,img1,tmodel,smodel1,toutput)


            #S2:


            s2output = smodel2(img2)
            #s1output = smodel1(img2)
            #s3output = smodel3(img2)
            #s4output = smodel4(img2)
            toutput =  tmodel_output(tmodel,img2)
            s2_loss = sh1losss(s2output,labels2,img2,tmodel,smodel2,toutput)

            


            #S3:

            #s1output = smodel1(img3)
            #s2output = smodel2(img3)
            s3output = smodel3(img3)
            #s4output = smodel4(img3)
            #toutput =  tmodel_output(tmodel,img3)

            s3_loss = m2losss(s3output,labels3,img3,tmodel,smodel3,toutput)


            

            #S4:
            #s1output = smodel1(img4)
            #s2output = smodel2(img4)
            #s3output = smodel3(img4)
            s4output = smodel4(img4)
            toutput =  tmodel_output(tmodel,img4)

            s4_loss = sh2losss(s4output,labels4,img4,tmodel,smodel4,toutput)


            






            tl1.append(s1_loss.item())
            tl2.append(s2_loss.item())
            tl3.append(s3_loss.item())
            tl4.append(s4_loss.item())


            #print(s1_loss,s2_loss,s3_loss,s4_loss)
            s1_loss.backward(retain_graph=True)
            s2_loss.backward(retain_graph=True)
            s3_loss.backward(retain_graph=True)
            s4_loss.backward(retain_graph=True)

            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            optimizer4.zero_grad()



        for p in val_loader:
                a = p[0]
                b = p[1]
                a = a.to(device)
                b = b.to(device)
                rr = smodel1(a)
                predicted_probs = torch.sigmoid(rr).detach().cpu().numpy()
                predicted_probs_list1.append(predicted_probs)
                true_labels_list1.append(b.cpu().numpy())
                sloss1 = nn.BCELoss()(torch.sigmoid(rr),b.float())
                vl1.append(sloss1.item())


                ypred = smodel2(a)
                #acc2 = accuracy(ypred,b)
                predicted_probs = torch.sigmoid(ypred).detach().cpu().numpy()
                predicted_probs_list2.append(predicted_probs)
                true_labels_list2.append(b.cpu().numpy())
                #val_acc2.append(acc2)
                sloss2 = nn.BCELoss()(torch.sigmoid(ypred),b.float())
                vl2.append(sloss2.item())


                rr = smodel3(a)
                predicted_probs = torch.sigmoid(rr).detach().cpu().numpy()
                predicted_probs_list3.append(predicted_probs)
                true_labels_list3.append(b.cpu().numpy())
                sloss3 = nn.BCELoss()(torch.sigmoid(rr),b.float())
                vl3.append(sloss3.item())

                rr = smodel4(a)
                predicted_probs = torch.sigmoid(rr).detach().cpu().numpy()
                predicted_probs_list4.append(predicted_probs)
                true_labels_list4.append(b.cpu().numpy())
                sloss4 = nn.BCELoss()(torch.sigmoid(rr),b.float())
                vl4.append(sloss4.item())







        








        predicted_probs_array1 = np.concatenate(predicted_probs_list1, axis=0)
        true_labels_array1 = np.concatenate(true_labels_list1, axis=0)
        predicted_probs_array2 = np.concatenate(predicted_probs_list2, axis=0)
        true_labels_array2 = np.concatenate(true_labels_list2, axis=0)
        predicted_probs_array3 = np.concatenate(predicted_probs_list3, axis=0)
        true_labels_array3 = np.concatenate(true_labels_list3, axis=0)
        predicted_probs_array4 = np.concatenate(predicted_probs_list4, axis=0)
        true_labels_array4 = np.concatenate(true_labels_list4, axis=0)
        roc_auc_scores1 = []
        roc_auc_scores2 = []
        roc_auc_scores3 = []
        roc_auc_scores4 = []
        for class_idx in range(15):
            roc_auc = roc_auc_score(true_labels_array1[:, class_idx], predicted_probs_array1[:, class_idx])
            roc_auc_scores1.append(roc_auc)
        for class_idx in range(15):
            roc_auc = roc_auc_score(true_labels_array2[:, class_idx], predicted_probs_array2[:, class_idx])
            roc_auc_scores2.append(roc_auc)
        for class_idx in range(15):
            roc_auc = roc_auc_score(true_labels_array3[:, class_idx], predicted_probs_array3[:, class_idx])
            roc_auc_scores3.append(roc_auc)
        for class_idx in range(15):
            roc_auc = roc_auc_score(true_labels_array4[:, class_idx], predicted_probs_array4[:, class_idx])
            roc_auc_scores4.append(roc_auc)
        #m1tac.append(np.mean(tacc1))
        m1tl.append(np.mean(tl1))
        m1valac.append(np.mean(roc_auc_scores1))
        m1vl.append(np.mean(vl1))
        #m2tac.append(np.mean(tacc2))
        m2tl.append(np.mean(tl2))
        m2valac.append(np.mean(roc_auc_scores2))
        m2vl.append(np.mean(vl2))

        #m3tac.append(np.mean(tacc3))
        m3tl.append(np.mean(tl3))
        m3valac.append(np.mean(roc_auc_scores3))
        m3vl.append(np.mean(vl3))

        #m4tac.append(np.mean(tacc4))
        m4tl.append(np.mean(tl4))
        m4valac.append(np.mean(roc_auc_scores4))
        m4vl.append(np.mean(vl4))


        if(np.mean(roc_auc_scores1)>best1):
                torch.save(smodel1.state_dict(),'models/modi_s1.pth')
                best1 = np.mean(roc_auc_scores1)

        if(np.mean(roc_auc_scores2)>best2):
                torch.save(smodel2.state_dict(),'models/modi_s2.pth')
                best2 = np.mean(roc_auc_scores2)
        if(np.mean(roc_auc_scores3)>best3):
                torch.save(smodel3.state_dict(),'models/modi_s3.pth')
                best3 = np.mean(roc_auc_scores3)
        if(np.mean(roc_auc_scores4)>best4):
                torch.save(smodel4.state_dict(),'models/modi_s4.pth')
                best4 = np.mean(roc_auc_scores4)




        
        print(f'{i+1}:S1L:{np.mean(tl1):.4f} S2L: {np.mean(tl2):.4f} S3L: {np.mean(tl3):.4f} S4L: {np.mean(tl4):.4f} S1VL:{np.mean(vl1):.4f} S2VL: {np.mean(vl2):.4f} S3VL: {np.mean(vl3):.4f} S4VL: {np.mean(vl4):.4f}  S1Vc: {np.mean(roc_auc_scores1):.4f} S2Vc: {np.mean(roc_auc_scores2):.4f} S3Vc: {np.mean(roc_auc_scores3):.4f} S4Vc: {np.mean(roc_auc_scores4):.4f}')
