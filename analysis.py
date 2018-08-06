import numpy as np
import json
import glob
import cv2
import os
import pdb
import scipy.misc as misc
import argparse
from ptsemseg.metrics import _fast_hist
import matplotlib.pyplot as plt


def Kendall(error_sort, unct_sort):

    frame_num = len(error_sort)
    rank_by_error = range(frame_num)
    rank_by_unct = []
    for index, ele in enumerate(error_sort):
        rank_by_unct.append(unct_sort.index(ele))

    concordant = 0.0
    for index, ele in enumerate(unct_sort):
        set_error = set(rank_by_error[index + 1:])
        set_unct = set(rank_by_unct[index + 1:])
        set_intersec = set_error & set_unct
        concordant += len(set_intersec)

    tau = 4 * concordant / frame_num / (frame_num - 1) - 1
    return tau


def frame_ranking(args, img_names, pred_names, unct_names, label_names):
    unct_max = 0
    error_rate = []
    unct_mean = []

    # Ranking
    for index, pred_name in enumerate(pred_names):
        pred = np.load(pred_name)
        unct = np.load(unct_names[index])
        label = cv2.imread(label_names[index])[..., 0]
        
        correct_mask = (pred == label) | (label == 11)                                                      
        error_mask = ~correct_mask
        error_rate.append(error_mask.mean())
        unct_mean.append(unct.mean())

    error_sorted_names = [x for _, x in sorted(zip(error_rate, img_names), reverse=True)]
    unct_sorted_names = [x for _, x in sorted(
        zip(unct_mean, img_names), reverse=True)]

    return error_sorted_names, unct_sorted_names, unct_mean


def rank_evaluate(args, error_sorted_names, unct_sorted_names, acqu_func):
    # Kendall Evaluate
    kendall_tau = Kendall(error_sorted_names, unct_sorted_names)
    print 'Kendall tau', kendall_tau


    f = open(os.path.join(args.out_dir, 'ranking_metric_'+acqu_func+'.txt'), 'w')
    f.write('Kendall tau\n'+str(kendall_tau))


    # Retrieve Frame Evaluate 
    for choose_portion in [0.1, 0.3, 0.5, 0.7]:
        frame_num = len(error_sorted_names)
        num = int(frame_num * choose_portion)
        print 'choose', num, 'of frames'
        f.write('\nchoose '+ str(num) + ' of frames')

        count = 0
        for index, img in enumerate(unct_sorted_names):
            if error_sorted_names.index(img) < num:
                count += 1
            if index == num:
                break
        print 'match', count, 'frames, ', 'Rank IOU', count / float(num)
        f.write('\nmatch ' + str(count) + ' frames, ' + ' Rank IOU ' + str(count / float(num)))
    f.close()


def scores_unct(label_trues, label_preds, unct, n_class, percent):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    mask = unct < np.sort(unct)[int((unct.shape[0] - 1) * percent)]
    # print 'percent', percent, 'is', np.sort(unct)[int((unct.shape[0]-1)*percent)]
    label_trues = label_trues[mask]
    label_preds = label_preds[mask]
    hist = np.zeros((n_class, n_class))
    hist += _fast_hist(label_trues, label_preds, n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {'Overall Acc: \t': acc,
            'Mean Acc : \t': acc_cls,
            'FreqW Acc : \t': fwavacc,
            'Mean IoU : \t': mean_iu, }, cls_iu


def PR_Curve(args, pred_names, unct_names, label_names, acqu_func):
    x = np.array(range(1, 101))
    global_acc = []
    mean_acc = []
    freqw_acc = []
    Mean_IOU = []

    labels = np.array([], dtype='uint8')
    preds = np.array([], dtype='uint8')
    uncts = np.array([], dtype='uint8')
    for index, pred_name in enumerate(pred_names):
        pred = np.load(pred_name)
        unct = np.load(unct_names[index])
        label = cv2.imread(label_names[index])[..., 0]
        labels = np.append(labels, label.flatten())
        preds = np.append(preds, pred.flatten())
        uncts = np.append(uncts, unct.flatten())
    for i in range(1, 101):
        score, class_iou = scores_unct(labels, preds, uncts, 11, i / 100.0)
        #print i, '%' 'Overall Acc: \t', score['Overall Acc: \t'], 'Mean IoU : \t', score['Mean IoU : \t']
        global_acc.append(score['Overall Acc: \t'])
        mean_acc.append(score['Mean Acc : \t'])
        freqw_acc.append(score['FreqW Acc : \t'])
        Mean_IOU.append(score['Mean IoU : \t'])

    print 'global_acc', sum(global_acc)
    print 'mean_acc', sum(mean_acc)
    print 'freqw_acc', sum(freqw_acc)
    print 'Mean_IOU', sum(Mean_IOU)
    
    f = open(os.path.join(args.out_dir, 'pixel_level_PR_curve_'+acqu_func+'.txt'), 'w')
    f.write('global_acc\n'+str(global_acc)+' '+str(sum(global_acc))+'\n\nmean_acc\n'+str(mean_acc)+' '+str(sum(mean_acc))+\
            '\n\nfreqw_acc\n'+str(freqw_acc)+' '+str(sum(freqw_acc))+'\n\nMean_IOU\n'+str(Mean_IOU)+' '+str(sum(Mean_IOU)))
    f.close()

    return global_acc, mean_acc, freqw_acc, Mean_IOU


def analysis(args):
    root = json.load(open('config.json'))['camvid']['data_path']
    image_names = json.load(open(args.root + 'data_split.json', 'r'))['test']['labeled']
    label_names = [root + path.replace('test', 'test' + 'annot') for path in image_names]


    pred_names = glob.glob(os.path.join(args.out_pred_dir, '*'))
    pred_names.sort()
    unct_r_names = glob.glob(os.path.join(args.out_unct_dir_r, '*'))
    unct_r_names.sort()
    unct_e_names = glob.glob(os.path.join(args.out_unct_dir_e, '*'))
    unct_e_names.sort()
    unct_b_names = glob.glob(os.path.join(args.out_unct_dir_b, '*'))
    unct_b_names.sort()
    unct_v_names = glob.glob(os.path.join(args.out_unct_dir_v, '*'))
    unct_v_names.sort()
    warp_names = glob.glob(os.path.join(args.out_warp_dir, '*'))
    warp_names.sort()
    alpha_names = glob.glob(os.path.join(args.out_alpha_dir, '*'))
    alpha_names.sort()
    


    if not os.path.exists(args.out_error_dir):
        flow_error_names = None
    else:
        flow_error_names = glob.glob(os.path.join(args.out_error_dir, '*'))
        flow_error_names.sort()
    
    # CamVid has wrong label at this frame, we delete for reasonable result 
    del image_names[149]
    del pred_names[149]
    del unct_r_names[149]
    del unct_e_names[149]
    del unct_b_names[149]
    del unct_v_names[149]
    del label_names[149]
    print '========== Variation ratio =========='
    global_acc_r, mean_acc_r, freqw_acc_r, Mean_IOU_r = PR_Curve(args, pred_names, unct_r_names, label_names, 'r')
    print '============== Entropy =============='
    global_acc_e, mean_acc_e, freqw_acc_e, Mean_IOU_e = PR_Curve(args, pred_names, unct_e_names, label_names, 'e')
    print '=============== Bald ================'
    global_acc_b, mean_acc_b, freqw_acc_b, Mean_IOU_b = PR_Curve(args, pred_names, unct_b_names, label_names, 'b')
    print '============= Mean STD =============='
    global_acc_v, mean_acc_v, freqw_acc_v, Mean_IOU_v = PR_Curve(args, pred_names, unct_v_names, label_names, 'v')
    

    x = np.array([float(x)/100 for x in range(1, 101)])
    plt.plot(x, np.array(Mean_IOU_r), label="Variational Ratio")
    plt.plot(x, np.array(Mean_IOU_e), label="Entropy")
    plt.plot(x, np.array(Mean_IOU_b), label="BALD")
    plt.plot(x, np.array(Mean_IOU_v), label="Variance")
    plt.legend(loc='upper right')
    plt.grid(linestyle='-', linewidth=0.5)
    plt.title('PR-Curve of Mean IOU')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.savefig(os.path.join(args.out_dir, 'PR-curve.png'), bbox_inches='tight')


    print '========== Variation ratio =========='
    error_sorted_names, unct_sorted_names, unct_mean = frame_ranking(args, image_names, pred_names, unct_r_names, label_names)   
    rank_evaluate(args, error_sorted_names, unct_sorted_names, 'r')
    print '============== Entropy =============='
    error_sorted_names, unct_sorted_names, unct_mean = frame_ranking(args, image_names, pred_names, unct_e_names, label_names)
    rank_evaluate(args, error_sorted_names, unct_sorted_names, 'e')
    print '=============== Bald ================'
    error_sorted_names, unct_sorted_names, unct_mean = frame_ranking(args, image_names, pred_names, unct_b_names, label_names)
    rank_evaluate(args, error_sorted_names, unct_sorted_names, 'b')
    print '============= Mean STD =============='
    error_sorted_names, unct_sorted_names, unct_mean = frame_ranking(args, image_names, pred_names, unct_v_names, label_names)
    rank_evaluate(args, error_sorted_names, unct_sorted_names, 'v')

