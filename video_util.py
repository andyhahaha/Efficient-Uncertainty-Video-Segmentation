import numpy as np
import cv2
import scipy.misc as misc
import pdb

import torch
import torch.nn as nn
from torch.autograd import Variable


def visualize_flow(flow):

    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * 4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return bgr


def cal_flow(DF, image1, image2):

    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
    image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    flow = DF.calc(image1_gray, image2_gray, None)

    return -flow


def warp_array(array, flow):

    height = flow.shape[0]
    width = flow.shape[1]
    grid = np.dstack(np.meshgrid(np.arange(width),
                                 np.arange(height))).astype('float32')
    grid = flow + grid
    new_array = cv2.remap(array, grid, None, cv2.INTER_LINEAR)

    return new_array


def generate_meshgrid(flow):
    h = flow.size(2)
    w = flow.size(3)
    y = torch.arange(0, h).unsqueeze(1).repeat(1, w) / (h - 1) * 2 - 1
    x = torch.arange(0, w).unsqueeze(0).repeat(h, 1) / (w - 1) * 2 - 1
    mesh_grid = Variable(torch.stack([x,y], 0).unsqueeze(0).repeat(flow.size(0), 1, 1, 1).cuda(), volatile=True)
    return mesh_grid


def warp_tensor(tensor, flow):
    mesh_grid = generate_meshgrid(flow)
    flow = flow.clone()
    flow[:,0,:,:] = flow[:,0,:,:] / (tensor.size()[3] / 2)
    flow[:,1,:,:] = flow[:,1,:,:] / (tensor.size()[2] / 2)
    grid = flow + mesh_grid
    grid = torch.transpose(torch.transpose(grid, 1, 2), 2, 3)
    warped_tensor = nn.functional.grid_sample(tensor, grid)

    return warped_tensor


def image_process(img, normalize=True):

    mean = [0.41189489566336, 0.4251328133025, 0.4326707089857]
    std = [0.27413549931506, 0.28506257482912, 0.28284674400252]


    img = img.astype(np.float64)
    if normalize:
        img = img.astype(float) / 255.0
        img -= mean
        img /= std
    # NHWC -> NCHW
    img_torch = img.transpose(2, 0, 1)
    img_torch = np.expand_dims(img_torch, 0)
    img_torch = torch.from_numpy(img_torch).float()

    return img_torch

