from datetime import datetime
import scipy.misc as sm
from collections import OrderedDict
import glob
import numpy as np
import socket

# PyTorch includes
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

# Custom includes
from dataloaders.combine_dbs import CombineDBs as combine_dbs
import dataloaders.pascal as pascal
import dataloaders.sbd as sbd
from dataloaders import custom_transforms as tr
from networks.loss import class_cross_entropy_loss 
from dataloaders.helpers import *
from networks.mainnetwork import *
import sys

import imageio

import numpy as np
import matplotlib.pyplot as plt

##########################################################################################
image_name = sys.argv[1]
out_mask = sys.argv[2]

threshold_seg = 0.1
nInputChannels = 5  # Number of input channels (RGB + heatmap of IOG points)
pad = 10 # padding cropped image
crop_img_size = (512, 512)
sigma = 20
model_path = 'models/IOG_pascal_epoch-99.pth'

img = cv2.imread(image_name)

#######################################################################################################
# Top-Left, Bottom Right and Inside point 

coordinates_bbox = []
def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN: 
        coordinates_bbox.append((x,y))
        cv2.circle(img, (x,y), 2, (255, 0, 0), 2)
        cv2.imshow('image', img) 

cv2.imshow('image', img) 
cv2.setMouseCallback('image', click_event) 
cv2.waitKey(0) 
cv2.destroyAllWindows() 

INSIDE_GUIDE = coordinates_bbox[2:]
OUTSIDE_GUIDE = coordinates_bbox[:2] + [(coordinates_bbox[0][0], coordinates_bbox[1][1]), (coordinates_bbox[1][0], coordinates_bbox[0][1])]
#######################################################################################################

img = cv2.imread(image_name) # reading again beacuse image was painted earlier

x_min = min(OUTSIDE_GUIDE[0][0],OUTSIDE_GUIDE[1][0])
y_min = min(OUTSIDE_GUIDE[0][1],OUTSIDE_GUIDE[1][1])
x_max = max(OUTSIDE_GUIDE[0][0],OUTSIDE_GUIDE[1][0])
y_max = max(OUTSIDE_GUIDE[0][1],OUTSIDE_GUIDE[1][1])

# padding and cropping
x_min = max(x_min - pad, 0)
y_min = max(y_min - pad, 0)
x_max = min(x_max + pad, img.shape[1]-1)
y_max = min(y_max + pad, img.shape[0]-1)

bbox = (x_min, y_min, x_max, y_max)
crop_img = crop_from_bbox(img, bbox)

crop_img = cv2.resize(crop_img, crop_img_size)
crop_h, crop_w = crop_img.shape[:2]

plt.imshow(crop_img)
plt.show()

# To tensor
img_tensor_channel_1 = torch.from_numpy(np.array([crop_img[:,:,0]]))
img_tensor_channel_2 = torch.from_numpy(np.array([crop_img[:,:,1]]))
img_tensor_channel_3 = torch.from_numpy(np.array([crop_img[:,:,2]]))

###########################################################################################################

# # Resize adjust
bbox_h, bbox_w = x_max-x_min, y_max-y_min
ratio_h, ratio_w = crop_h/float(bbox_h), crop_w/float(bbox_w)

# Single Inside point 
# # Crop and Pad adjust
# INSIDE_GUIDE = (INSIDE_GUIDE[0]-x_min, INSIDE_GUIDE[1]-y_min)

# INSIDE_GUIDE = (INSIDE_GUIDE[0]*ratio_h, INSIDE_GUIDE[1]*ratio_w)
# inside = make_gaussian((crop_h, crop_w), center=INSIDE_GUIDE, sigma=20)
# inside_tensor = torch.from_numpy(np.array([inside*100]))

# Multiple Inside point 
inside = np.zeros((crop_h, crop_w))
for coordinate in INSIDE_GUIDE:
    # Crop and Pad adjust
    coordinate = (coordinate[0]-x_min, coordinate[1]-y_min)
    # Resize adjust
    coordinate = (coordinate[0]*ratio_h, coordinate[1]*ratio_w)
    inside += make_gaussian((crop_h, crop_w), center=coordinate, sigma=sigma)

inside_tensor = torch.from_numpy(np.array([inside*100]))

# plt.imshow(inside_tensor.numpy()[0, :, :])
# plt.show()
###########################################################################################################

outside = np.zeros((crop_h, crop_w))
for coordinate in OUTSIDE_GUIDE:
    # Crop and Pad adjust
    coordinate = (coordinate[0]-x_min, coordinate[1]-y_min)
    # Resize adjust
    coordinate = (coordinate[0]*ratio_h, coordinate[1]*ratio_w)
    outside += make_gaussian((crop_h, crop_w), center=coordinate, sigma=sigma)

outside_tensor = torch.from_numpy(np.array([outside*100]))

# plt.imshow(outside_tensor.numpy()[0, :, :])
# plt.show()
###########################################################################################################

inputs = torch.cat((img_tensor_channel_1, img_tensor_channel_2, img_tensor_channel_3)).double()
inputs = torch.cat((inputs, inside_tensor, outside_tensor))
inputs = torch.from_numpy(np.array([inputs.numpy()]))

train_loader = torch.utils.data.DataLoader(inputs)

##########################################################################################
# Set gpu_id to -1 to run in CPU mode, otherwise set the id of the corresponding gpu
gpu_id = -1
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('Using GPU: {} '.format(gpu_id))

# Network definition
modelName = 'IOG_pascal'
net = Network(nInputChannels=nInputChannels,num_classes=1,
                backbone='resnet101',
                output_stride=16,
                sync_bn=None,
                freeze_bn=False)

# load pretrain_dict
pretrain_dict = torch.load(model_path)
net.load_state_dict(pretrain_dict)
net.to(device)
net.eval()

print('Testing Network')
with torch.no_grad():
    for input_ in train_loader:
        input_ = input_.to(device)
   
        coarse_outs1,coarse_outs2,coarse_outs3,coarse_outs4,fine_out = net.forward(input_.float())
        outputs = fine_out.to(torch.device('cpu'))

        pred = np.transpose(outputs.data.numpy()[0, :, :, :], (1, 2, 0))
        pred = 1 / (1 + np.exp(-pred))
        pred = np.squeeze(pred)
        
        # plt.imshow(pred)
        # plt.show()

        result = crop2fullmask(pred, bbox, im_size=img.shape[:2], zero_pad=True, relax=0, mask_relax=False)

        # plt.imshow(result)
        # plt.show()

        result[result > threshold_seg] = 255
        result[result <= threshold_seg] = 0
        imageio.imwrite(out_mask, result)

        # from PIL import Image

        # #image1 Original image 
        # #image2 Segmentation image
        # image1 = Image.open(image_name)
        # image2 = Image.open(out_mask)
        
        # image1 = image1.convert('RGBA')
        # image2 = image2.convert('RGBA')

        # image = Image.blend(image1,image2,0.3)
        # image.show()

        plt.imshow(img)
        plt.imshow(result, alpha=0.5)
        plt.show()

        # imageio.imwrite(out_mask, result)


