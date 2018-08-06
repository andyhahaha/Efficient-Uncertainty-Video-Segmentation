from math import ceil
import glob
import argparse
import pdb
import os
import sys
import json

import torch
from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from train import  train
from validate import test, setup_output, eval_metrics
from analysis import analysis
from ptsemseg import utils
from ptsemseg.augmentations import *

def setup_model(args):

    # Define dataset 
    args.img_rows = 360
    args.img_cols = 480
    data_loader = get_loader('camvid')
    data_path = get_data_path('camvid')

    if args.arch == 'bayesian_segnet':
        dataset = data_loader(data_path, 'train', is_transform=True, img_size=(args.img_rows, args.img_cols))
    elif args.arch == 'bayesian_tiramisu':
        dataset = data_loader(data_path, 'train', is_transform=True, img_size=(args.img_rows, args.img_cols), crop = True)

    # setup model
    args.n_classes = dataset.n_classes
    model = get_model(args.arch, args.n_classes)
    model = torch.nn.DataParallel(model,
                                  device_ids=range(torch.cuda.device_count()))

    if args.arch == 'bayesian_tiramisu':
        model.apply(utils.weights_init)

    # setup optimizer for training
    if args.mode == 'train':
        args.start_episode = 0
        args.start_epoch = 0

        lr = args.l_rate
        if args.opt == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=lr,
                                        momentum=0.9,
                                        weight_decay=args.weight_decay) #5e-4 for segnet
        elif args.opt == 'rms':
            optimizer = torch.optim.RMSprop(model.parameters(),
                                            lr=lr,
                                            weight_decay=args.weight_decay)

    # load model if needed
    if (args.mode == 'train' and args.resume) or args.mode == 'test':
        print '[Info] Checkpoint path:', args.model_path
        if not os.path.exists(args.model_path):
            sys.exit('[Error] Checkpoint path not exists!')
        ckpt = torch.load(args.model_path)
        model.load_state_dict(ckpt['state_dict'])
        if args.mode == 'train':
            optimizer.load_state_dict(ckpt['optimizer'])
            args.start_episode = ckpt['episode']
            args.start_epoch = ckpt['epoch']
   
    if args.mode == 'train':
        return model.cuda(), optimizer, dataset
    else:
        return model.cuda()


def show_args(args):
    args_dict = vars(args)
    for key in args_dict:
        print(key, args_dict[key])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--mode', nargs='?', type=str, default='',
                            help='Mode to run [train, test]')
    parser.add_argument('--arch', nargs='?', type=str, default='bayesian_segnet',
                        help='Architecture to use [\'segnet, bayesian segnet etc\']')
    parser.add_argument('--description', nargs='?', type=str, default='',
                            help='Description of this model')

    # resume model for training
    parser.add_argument('--resume', dest='resume', action='store_true',
                            help='resume model or not')
    parser.add_argument('--resume_episode', nargs='?', type=int, default=None,
                        help='Episode of the checkpoint')
    parser.add_argument('--resume_epoch', nargs='?', type=int, default=None,
                        help='Epoch of the checkpoint')

    # Regular learning parameter
    parser.add_argument('--n_epoch', nargs='?', type=int, default=100,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=4,
                        help='Batch Size')
    parser.add_argument('--opt', nargs='?', type=str, default='sgd',
                        help='Leaning method [sgd, rms]')
    parser.add_argument('--l_rate', nargs='?', type=float, default=0.001,
                        help='Learning Rate')
    parser.add_argument('--lr_decay', nargs='?', type=float, default=1,
                        help='Learning rate decay factor')
    parser.add_argument('--weight_decay', nargs='?', type=float, default=1e-4,
                        help='Weight decay')
    # Active learning
    parser.add_argument('--init_epoch', nargs='?', type=int, default=None,
                        help='# of the epochs in initial training phase')
    parser.add_argument('--num_select', nargs='?', type=int, default=37,
                        help='# of data selected to label')
    parser.add_argument('--acqu_func', nargs='?', type=str, default='',
                            help='Which acquisition function to use [entropy, variation ratio, variance etc]')
    parser.add_argument('--random_select', dest='random_select', action='store_true',
                            help='random select new data')
    parser.add_argument('--init_lr', nargs='?', type=float, default=0.001,
                        help='Learning Rate')
    parser.add_argument('--strat', nargs='?', type=str, default='fine',
                        help='Learning stratege [fine, scratch]')
    parser.add_argument('--sim', nargs='?', type=float, default=1.0,
                        help='similarity threshold')
    # Uncertainty
    parser.add_argument('--sample_num', nargs='?', type=int, default=None,
                        help='Number of sample')
    parser.add_argument('--error_thres', nargs='?', type=float, default=None,
                        help='threshold of reconstruction error')
    parser.add_argument('--alpha_error', nargs='?', type=float, default=None,
                        help='alpha when flow error')
    parser.add_argument('--alpha_normal', nargs='?', type=float, default=None,
                        help='alpha when flow correct')
    parser.add_argument('--video_unct', dest='video_unct', action='store_true',
                        help='Whether to use video uncertainty or not')
    parser.add_argument('--flow', nargs='?', type=str, default='flownet2',
                        help='Which flow to use [DF, flownet2]')
    


    # validate
    parser.add_argument('--ckpt_episode', nargs='?', type=int, default=None,
                        help='Episode of the checkpoint')
    parser.add_argument('--ckpt_epoch', nargs='?', type=int, default=None,                            
                        help='Epoch of the checkpoint')
    parser.add_argument('--split', nargs='?', type=str, default='val',
                        help='Split of dataset to test on')
    parser.add_argument('--save_output', dest='save_output', action='store_true',
                            help='Whether to save model output or not')

    # Log
    parser.add_argument('--save_percent', nargs='?', type=float, default=0.1,
                            help='Percent of saving model checkpoint')
    parser.add_argument('--eval_interval', nargs='?', type=int, default=1,
                            help='Interval of eval model')
    parser.add_argument('--visdom', nargs='?', type=int, default=None,
                        help='Show visualization(s) on visdom on port | None by  default')
    args = parser.parse_args()
    show_args(args)

    args.exp_name = args.arch
    if args.description:
        args.exp_name += '_' + args.description
    args.root = json.load(open('config.json'))['camvid']['data_path']

    if args.mode == 'train':
        if not os.path.exists('checkpoint/' + args.exp_name):
            os.makedirs('checkpoint/' + args.exp_name)

        if args.resume:
            args.model_path = os.path.join('checkpoint', args.exp_name, '{}_{}_{}.pth.tar'.format(\
                                           args.arch, args.resume_episode, args.resume_epoch))

        model, optimizer, dataset = setup_model(args)
        train(args, model, optimizer, dataset = dataset)
    elif args.mode == 'test':
        args.model_path = os.path.join('checkpoint', args.exp_name, '{}_{}_{}.pth.tar'.format(\
                                       args.arch, args.ckpt_episode, args.ckpt_epoch))

        model = setup_model(args)

        # Setup output directories
        if args.save_output:
            setup_output(args)

        # run validate
        gts, preds, uncts = test(args, model, split=args.split, verbose=True)
        # eval metrics
        eval_metrics(args, gts, preds)
        if args.save_output:
            analysis(args)

    else:
        print 'please choose mode [train, test]'



