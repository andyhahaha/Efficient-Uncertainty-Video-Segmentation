import numpy as np
import sys
import visdom
import argparse
from tqdm import tqdm
import time
import os
import glob
import cv2
import scipy.misc as misc
import matplotlib.pyplot as plt
from PIL import Image
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils import data

from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.metrics import scores
from ptsemseg.loader.camvid_dataset import CamVid, LabelToLongTensor
from video_util import *
from pytorch_flownet2.FlowNet2_src import FlowNet2, flow_to_image
from ptsemseg.loss import cross_entropy2d

import pdb

def set_dropout(m):
    if type(m) == nn.Dropout2d:
        m.train()


def eval_metrics(args, gts, preds, verbose=True):

    classes = ['Sky', 'Building', 'Column-Pole', 'Road',
            'Sidewalk', 'Tree', 'Sign-Symbol', 'Fence', 'Car', 'Pedestrain',
            'Bicyclist']
    class_order = [1, 5, 0, 8, 6, 3, 9, 7, 2, 4, 10]  # class order on the paper

    score, class_iou, class_acc = scores(gts, preds, args.n_classes)
    results_str = ''

    for k, v in score.items():
        if verbose:
            print(k, v)
        results_str += str(k) + ': ' + str(v) + '\n'

    if verbose:
        print 'class iou:'
    results_str += '\nclass iou:\n'
    for i in range(len(classes)):
        if verbose:
            print(classes[class_order[i]], class_iou[class_order[i]])
        results_str += str(class_iou[class_order[i]]) + ' '
    if verbose:
        print 'class acc:'
    results_str += '\n\nclass acc:\n'
    for i in range(len(classes)):
        if verbose:
            print(classes[class_order[i]], class_acc[class_order[i]])
        results_str += str(class_acc[class_order[i]]) + ' '
    results_str += '\n'

    if args.save_output:
        print 'Save results to ', args.out_dir
        f = open(os.path.join(args.out_dir, 'results.txt'), 'w')
        f.write(results_str)
        f.close()

    return results_str, score


def acquisition_func(acqu, output_mean, square_mean=None, entropy_mean=None):
    if acqu == 'e':  # max entropy
        return -(output_mean * torch.log(output_mean)).mean(1)
    elif acqu == 'b':  
        return acquisition_func('e', output_mean) - entropy_mean
    elif acqu == 'r':  # variation ratios
        return 1 - output_mean.max(1)[0]
    elif acqu == 'v':  # mean STD
        return (square_mean - output_mean.pow(2)).mean(1)



def validate_bayesian(args, model, split, labeled_index=None, verbose=False):
    
    # Setup Data
    data_loader = get_loader('camvid')
    data_path = get_data_path('camvid')
    dataset = data_loader(data_path, split, is_transform=True, labeled_index=labeled_index)
    valloader = data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)

    # Setup Model
    model.eval()
    model.apply(set_dropout)

    # Uncertainty Hyperparameter
    T = args.sample_num
    inference_time = 0
    gts, preds, uncts = [], [], []
    uncts_r, uncts_e, uncts_v, uncts_b = [], [], [], []
    image_names_all = []
    for i, (images, labels, image_names) in enumerate(valloader):
        torch.cuda.synchronize()
        t1 = time.time()
        if torch.cuda.is_available():
            model.cuda()
            images = Variable(images.cuda(), volatile=True)
            labels_var = Variable(labels.cuda(async=True), volatile=True)
        else:
            images = Variable(images, volatile=True)
            labels_var = Variable(labels, volatile=True)

        output_list = []

        # MC dropout
        for t in range(T):
            output = F.softmax(model(images))
            if t == 0:
                output_mean = output * 0
                output_square = output * 0
                entropy_mean = output.mean(1) * 0
            output_mean += output
            output_square += output.pow(2)
            entropy_mean += acquisition_func('e', output)
        output_mean = output_mean / T
        output_square = output_square / T
        entropy_mean = entropy_mean / T
        # Uncertainty estimation
        if args.acqu_func != 'all':
            unc_map = acquisition_func(args.acqu_func, output_mean,\
                                    square_mean=output_square, entropy_mean=entropy_mean)
        else:
            unc_map_r = acquisition_func('r', output_mean,\
                                    square_mean=output_square, entropy_mean=entropy_mean)
            unc_map_e = acquisition_func('e', output_mean,\
                                    square_mean=output_square, entropy_mean=entropy_mean)
            unc_map_b = acquisition_func('b', output_mean,\
                                    square_mean=output_square, entropy_mean=entropy_mean)
            unc_map_v = acquisition_func('v', output_mean,\
                                    square_mean=output_square, entropy_mean=entropy_mean)
        pred = torch.max(output_mean, 1)[1]
        torch.cuda.synchronize()
        t2 = time.time()    
        gts += list(labels.numpy())
        preds += list(pred.data.cpu().numpy())
        if args.acqu_func != 'all':
            uncts += list(unc_map.data.cpu().numpy())
        else:
            uncts_r += list(unc_map_r.data.cpu().numpy())
            uncts_e += list(unc_map_e.data.cpu().numpy())
            uncts_b += list(unc_map_b.data.cpu().numpy())
            uncts_v += list(unc_map_v.data.cpu().numpy())
        image_names_all += list(image_names)
        inference_time += t2 - t1
        if verbose:
            print '[Info] evaluate ', image_names
    print '[Time] Average Inference Time = ', inference_time / (i+1)
        
    # Save unct_map and pred_map
    if args.save_output:
        for index in range(len(preds)):
            out_name = os.path.basename(image_names_all[index]).replace('.png', '')
            np.save(os.path.join(args.out_pred_dir, out_name), preds[index])
            if args.acqu_func != 'all':
                np.save(os.path.join(args.out_unct_dir, out_name), uncts[index])
            else:
                np.save(os.path.join(args.out_unct_dir_r, out_name), uncts_r[index])
                np.save(os.path.join(args.out_unct_dir_e, out_name), uncts_e[index])
                np.save(os.path.join(args.out_unct_dir_b, out_name), uncts_b[index])
                np.save(os.path.join(args.out_unct_dir_v, out_name), uncts_v[index])
    return gts, preds, uncts


def validate_video(args, model, split, labeled_index=None, verbose=False):
    # Setup Data
    n_classes = 11
    frame_names = json.load(open(args.root + 'data_split.json', 'r'))[split]['frames']
    frame_names = list(np.concatenate(frame_names))
    labeled_image_names = json.load(open(args.root + 'data_split.json', 'r'))[split]['labeled']

    # Setup Model
    model.eval()
    model.apply(set_dropout)

    # Optical Flow
    if args.flow == 'DF':
        DF = cv2.optflow.createOptFlow_DeepFlow()
    elif args.flow == 'flownet2':
        flownet2 = FlowNet2()
        path = 'pytorch_flownet2/FlowNet2_src/pretrained/FlowNet2_checkpoint.pth.tar'
        pretrained_dict = torch.load(path)['state_dict']
        model_dict = flownet2.state_dict()                                                                   
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        flownet2.load_state_dict(model_dict)
        flownet2.cuda()

    # Uncertainty Hyperparameter
    threshold = args.error_thres
    alpha_normal = args.alpha_normal
    alpha_error = args.alpha_error

    prev_frame = None
    gts, preds, uncts = [], [], []
    uncts_r, uncts_e, uncts_v, uncts_b = [], [], [], []
    video_name = ''
    inference_time = 0
    target_transform = LabelToLongTensor()


    for i, frame_name in enumerate(frame_names):
        torch.cuda.synchronize()
        t1 = time.time()

        img = misc.imread(args.root + frame_name)
        img_tensor = image_process(img)
        img_tensor_or = image_process(img, normalize=False)
        if torch.cuda.is_available():
            model.cuda()
            images = Variable(img_tensor.cuda(async=True), volatile=True)
            images_or = Variable(img_tensor_or.cuda(async=True), volatile=True)
        else:
            images = Variable(img_torch, volatile=True)
            images_or = Variable(img_tensor_or, volatile=True)
        
        output = F.softmax(model(images))

        # Temporal Aggregation
        if video_name != frame_name.split('/')[1]:   # first frame of video
            output_mean = output
            square_mean = output.pow(2)
            entropy_mean = acquisition_func('e', output)
            video_name = frame_name.split('/')[1]
            reconstruction_loss = None
        else:
            if args.flow == 'DF':
                flow = cal_flow(DF, prev_frame, img)
            elif args.flow == 'flownet2':
                input = torch.cat((prev_frame, images_or), 0).unsqueeze(0).transpose(1,2)
                flow = -flownet2(input)

            # generate spatial alpha
            warp_frame = warp_tensor(prev_frame, flow)
            reconstruction_loss = (warp_frame - images_or).abs().mean(1)
            mask = reconstruction_loss < threshold
            alpha = mask.float() * alpha_normal + (1 - mask.float()) * alpha_error

            # warp and running mean
            output_mean = warp_tensor(output_mean, flow)
            square_mean = warp_tensor(square_mean, flow)
            entropy_mean = warp_tensor(entropy_mean.unsqueeze(1), flow).squeeze(1)
            output_mean = output_mean * (1 - alpha) + output * alpha
            square_mean = square_mean * (1 - alpha) + output.pow(2) * alpha
            entropy_mean = entropy_mean * (1 - alpha) + acquisition_func('e', output) * alpha

        # prediction and uncertainty
        pred = output_mean.max(1)[1].squeeze()
        if args.acqu_func != 'all':
            unc_map = acquisition_func(args.acqu_func, output_mean,\
                                   square_mean=square_mean, entropy_mean=entropy_mean) 
            unc_map = unc_map.squeeze()
        else:
            unc_map_r = acquisition_func('r', output_mean,\
                                    square_mean=square_mean, entropy_mean=entropy_mean)
            unc_map_e = acquisition_func('e', output_mean,\
                                    square_mean=square_mean, entropy_mean=entropy_mean)
            unc_map_b = acquisition_func('b', output_mean,\
                                    square_mean=square_mean, entropy_mean=entropy_mean)
            unc_map_v = acquisition_func('v', output_mean,\
                                    square_mean=square_mean, entropy_mean=entropy_mean)
            unc_map_r = unc_map_r.squeeze()
            unc_map_e = unc_map_e.squeeze()
            unc_map_b = unc_map_b.squeeze()
            unc_map_v = unc_map_v.squeeze()
        torch.cuda.synchronize()
        t2 = time.time()

        # If the frame has label, evaluate it
        if frame_name in labeled_image_names:
            if verbose:
                print '[Info] evaluate', frame_name
            first_frame = int(os.path.basename(frame_name).split('_')[1].split('.')[0]) == 0
            gt = cv2.imread(args.root + frame_name.replace(split, split+'annot'))[..., 0]
            gt_var = Variable(target_transform(gt).cuda(), volatile=True)
            gts.append(gt)
            pred = pred.data.cpu().numpy()
            preds.append(pred)
            if args.acqu_func != 'all':
                unc_map = unc_map.data.cpu().numpy()
                uncts.append(unc_map)
            else:
                unc_map_r = unc_map_r.data.cpu().numpy()
                uncts_r.append(unc_map_r)
                unc_map_e = unc_map_e.data.cpu().numpy()
                uncts_e.append(unc_map_e)
                unc_map_b = unc_map_b.data.cpu().numpy()
                uncts_b.append(unc_map_b)
                unc_map_v = unc_map_v.data.cpu().numpy()
                uncts_v.append(unc_map_v)
            if not first_frame:
                reconstruction_loss = reconstruction_loss.data.cpu().numpy()
            # Save unct_map and pred_map
            if args.save_output:
                out_name = os.path.basename(frame_name).replace('.png', '')
                np.save(os.path.join(args.out_pred_dir, out_name), pred)

                # save warp frame and alpha if you want
                #np.save(os.path.join(args.out_warp_dir, out_name), warp_frame.data.cpu().numpy().squeeze().transpose(1,2,0))
                #np.save(os.path.join(args.out_alpha_dir, out_name), alpha.squeeze().data.cpu().numpy())
                if args.acqu_func != 'all':
                    np.save(os.path.join(args.out_unct_dir, out_name), unc_map)
                else:
                    np.save(os.path.join(args.out_unct_dir_r, out_name), unc_map_r)
                    np.save(os.path.join(args.out_unct_dir_e, out_name), unc_map_e)
                    np.save(os.path.join(args.out_unct_dir_b, out_name), unc_map_b)
                    np.save(os.path.join(args.out_unct_dir_v, out_name), unc_map_v)
                if not first_frame:
                    haha = 1
                    #np.save(os.path.join(args.out_error_dir, out_name), reconstruction_loss)
        prev_frame = images_or
        inference_time += t2 - t1       
    print '[Profile] Average Inference Time = ', inference_time / (i+1)


    return gts, preds, uncts


def setup_output(args):
    # Setup output directories
    if not args.video_unct:
        args.out_dir = os.path.join('checkpoint', args.exp_name, \
                                    'output_{}_{}_s{}_{}'.format(args.ckpt_episode, args.ckpt_epoch, args.sample_num, args.acqu_func))
    else:
        args.out_dir = os.path.join('checkpoint', args.exp_name, \
                                    'output_{}_{}_{}_th{}_an{}_ae{}_{}'.format(args.ckpt_episode, args.ckpt_epoch, args.flow, args.error_thres, args.alpha_normal, args.alpha_error, args.acqu_func))

    args.out_pred_dir = os.path.join(args.out_dir, 'pred')
    args.out_error_dir = os.path.join(args.out_dir, 'flow_error')
    args.out_warp_dir = os.path.join(args.out_dir, 'warped_frame')
    args.out_alpha_dir = os.path.join(args.out_dir, 'alpha')
    if not os.path.exists(args.out_pred_dir):
        os.makedirs(args.out_pred_dir)
        print 'mkdir', args.out_pred_dir
    if not os.path.exists(args.out_error_dir):
        os.makedirs(args.out_error_dir)
        print 'mkdir', args.out_error_dir
    if not os.path.exists(args.out_warp_dir):
        os.makedirs(args.out_warp_dir)
        print 'mkdir', args.out_warp_dir
    if not os.path.exists(args.out_alpha_dir):
        os.makedirs(args.out_alpha_dir)
        print 'mkdir', args.out_alpha_dir

    if args.acqu_func != 'all':

        args.out_unct_dir = os.path.join(args.out_dir, 'unct_'+args.acqu_func)
        if not os.path.exists(args.out_unct_dir):
            os.makedirs(args.out_unct_dir)
            print 'mkdir', args.out_unct_dir
    else:

        args.out_unct_dir_r = os.path.join(args.out_dir, 'unct_r')
        if not os.path.exists(args.out_unct_dir_r):
            os.makedirs(args.out_unct_dir_r)
            print 'mkdir', args.out_unct_dir_r

        args.out_unct_dir_e = os.path.join(args.out_dir, 'unct_e')
        if not os.path.exists(args.out_unct_dir_e):
            os.makedirs(args.out_unct_dir_e)
            print 'mkdir', args.out_unct_dir_e

        args.out_unct_dir_b = os.path.join(args.out_dir, 'unct_b')
        if not os.path.exists(args.out_unct_dir_b):
            os.makedirs(args.out_unct_dir_b)
            print 'mkdir', args.out_unct_dir_b

        args.out_unct_dir_v = os.path.join(args.out_dir, 'unct_v')
        if not os.path.exists(args.out_unct_dir_v):
            os.makedirs(args.out_unct_dir_v)
            print 'mkdir', args.out_unct_dir_v

def test(args, model, split, labeled_index=None, verbose=False):
    if 'bayesian' not in args.arch:
        print '[Info] use validate_no_bayesian function'
        gts, preds = validate_no_bayesian(args, model)
        return gts, preds, None, None
    else:
        if args.video_unct:
            print '[Info] use validate_video function'
            gts, preds, uncts = validate_video(args, model, split, labeled_index, verbose)
        else:
            print '[Info] use validate_bayesian function'
            gts, preds, uncts = validate_bayesian(args, model, split, labeled_index, verbose)
        return gts, preds, uncts


