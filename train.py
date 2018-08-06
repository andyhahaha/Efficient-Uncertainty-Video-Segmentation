import numpy as np
import os
import pdb
import visdom
import argparse
from math import ceil, floor
import glob
import time

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils import data

from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.loss import cross_entropy2d
from ptsemseg.loader.camvid_dataset import CamVid
from ptsemseg import utils
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.augmentations import *
from validate import test, eval_metrics
import json
from ptsemseg.models import get_model


def train(args, model, optimizer, dataset, episode=0):
    
    trainloader = data.DataLoader(dataset, batch_size=args.batch_size, num_workers=8, shuffle=True, drop_last=True)

    class_weight = Variable(dataset.class_weight.cuda())

    lr = args.l_rate
    n_epoch = args.n_epoch
    optimizer.param_groups[0]['lr'] = args.l_rate
    model.train()

    # Setup visdom for visualization
    if args.visdom:
        vis = visdom.Visdom(port=args.visdom)
        loss_window = vis.line(X=np.column_stack((np.zeros((1,)))),
                               Y=np.column_stack((np.zeros((1)))),
                               opts=dict(xlabel='epoch',
                                         ylabel='Loss',
                                         title=args.mode + '_' + args.exp_name + '_Episode_' + str(episode),
                                         legend=['Train Loss']))

    t1 = time.time()
    start_epoch = args.start_epoch if episode == args.start_episode else 0
    best_iou = -100.0
    save_interval = int(floor(n_epoch*args.save_percent))
    for epoch in range(1 + start_epoch, n_epoch + 1):
        utils.adjust_learning_rate(optimizer, args.l_rate, args.lr_decay, 
                                     epoch - 1, 1)
        for i, (images, labels, image_name) in enumerate(trainloader):
            
            images = Variable(images.cuda())
            labels = Variable(labels.cuda(async=True))
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = cross_entropy2d(outputs, labels, class_weight)
            loss.backward()
            optimizer.step()
        if epoch % (save_interval*args.eval_interval) == 0:
            gts, preds, uncts = test(args, model=model, split='val')
            model.train()
            _, score = eval_metrics(args, gts, preds, verbose=False)
            print 'val Mean IoU: ', score['Mean IoU : \t']
            if score['Mean IoU : \t'] >= best_iou:
                best_iou = score['Mean IoU : \t']
                state = {'episode': episode, 
                         'epoch': epoch,
                         'model_state': model.state_dict(),
                        'optimizer_state' : optimizer.state_dict(),}
                print "update best model {}".format(best_iou)
                torch.save(state, "checkpoint/{}/{}_{}_{}_best_model.pkl".format(\
                                        args.exp_name, args.arch, 'camvid', episode))           
        
        utils.adjust_learning_rate(optimizer, args.l_rate, args.lr_decay, 
                                     epoch - 1, 1)
        
        if epoch % save_interval == 0:
            print 'data_size : ', len(dataset)
            state = {
                'episode' : episode,
                'epoch': epoch,
                'arch': args.arch,
                'loss': loss.data[0],
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, 'checkpoint/{}/{}_{}_{}.pth.tar'.format(\
                                   args.exp_name, args.arch, episode, epoch))
            print("Epoch [%d/%d] Loss: %.4f  lr:%.4f" %
                    (epoch, n_epoch, loss.data[0], optimizer.param_groups[0]['lr'] )) 
            t2 = time.time()
            print save_interval, 'epoch time :', t2 - t1
            t1 = time.time()

        if args.visdom:
            vis.line(
                X=np.column_stack((np.ones((1,)) * epoch)),
                Y=np.column_stack((np.array([loss.data[0]]))),
                win=loss_window,
                update='append')
    return model, optimizer

