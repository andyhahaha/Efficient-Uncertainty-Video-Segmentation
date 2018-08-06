import subprocess, sys
import os
import pdb

mode = 'test'
description = 'lr0.001_b4_trained'
arch = 'bayesian_tiramisu'

# training
n_epoch = 900
l_rate = 0.001
lr_decay = 0.995
weight_decay = 1e-4
batch_size = 4
opt = 'rms'
visdom = 0
resume = False
resume_episode = 0
resume_epoch = 900
save_percent = 0.1
eval_interval = 1

# active
init_epoch = 10000
num_select = 37
acqu_func = 'all'
init_lr = 0.001
random_select = False
strat = 'scratch'
sim = 0.95

# testing
video_unct = False
flow = 'flownet2'
data_split = 'test'
ckpt_episode = 0
ckpt_epoch = 900
sample_num = 5
error_thres = 40
alpha_normal = 0.2
alpha_error = 0.5
save_output = True

if 'train' in mode:

    exp_name = arch + '_' + description if description else arch
    if not os.path.exists('checkpoint/' + exp_name):
        os.makedirs('checkpoint/' + exp_name)

    cmd = 'CUDA_VISIBLE_DEVICES=0,1 python -u main.py --mode '+mode+\
                                                    ' --arch '+arch+\
                                                    ' --n_epoch '+str(n_epoch)+\
                                                    ' --l_rate '+str(l_rate)+\
                                                    ' --lr_decay '+str(lr_decay)+\
                                                    ' --weight_decay '+str(weight_decay)+\
                                                    ' --batch_size '+str(batch_size)+\
                                                    ' --opt '+opt+\
                                                    ' --sample_num '+str(sample_num)+\
                                                    ' --init_epoch '+str(init_epoch)+\
                                                    ' --num_select '+str(num_select)+\
                                                    ' --acqu_func '+acqu_func+\
                                                    ' --init_lr '+str(init_lr)+\
                                                    ' --error_thres '+str(error_thres)+\
                                                    ' --alpha_normal '+str(alpha_normal)+\
                                                    ' --alpha_error '+str(alpha_error)+\
                                                    ' --sim '+str(sim)+\
                                                    ' --strat '+str(strat)+\
                                                    ' --description '+str(description)+\
                                                    ' --visdom '+str(visdom)+\
                                                    ' --save_percent '+str(save_percent)+\
                                                    ' --eval_interval '+str(eval_interval)

    if resume:
        cmd += ' --resume --resume_episode '+str(resume_episode)+' --resume_epoch '+str(resume_epoch)
    if video_unct:
        cmd += ' --video_unct '
    if random_select:
        cmd += ' --random_select '
    cmd += ' 2>&1 | tee checkpoint/'+exp_name+'/log'

elif mode == 'test':
    cmd = 'CUDA_VISIBLE_DEVICES=0,1 python -u main.py --mode '+mode+\
                                                    ' --arch '+arch+\
                                                    ' --flow '+flow+\
                                                    ' --ckpt_episode '+str(ckpt_episode)+\
                                                    ' --ckpt_epoch '+str(ckpt_epoch)+\
                                                    ' --batch_size '+str(batch_size)+\
                                                    ' --split '+data_split+\
                                                    ' --sample_num '+str(sample_num)+\
                                                    ' --acqu_func '+acqu_func+\
                                                    ' --error_thres '+str(error_thres)+\
                                                    ' --alpha_normal '+str(alpha_normal)+\
                                                    ' --alpha_error '+str(alpha_error)+\
                                                    ' --description '+str(description)

    if video_unct:
        cmd += ' --video_unct'
    if save_output:
        cmd += ' --save_output'

print 'cmd: ', cmd
out = subprocess.call(cmd, shell=True)

