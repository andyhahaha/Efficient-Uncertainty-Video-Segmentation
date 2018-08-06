import matplotlib
import numpy as np
from scipy.misc import imread
import torch
from torch.autograd import Variable

from FlowNet2_src import FlowNet2
from FlowNet2_src import flow_to_image
import matplotlib.pyplot as plt

import cv2
import pdb
import time
def demo(prev,after):
  # Prepare img pair
  #im1 = imread('FlowNet2_src/example/0img0.ppm')
  #im2 = imread('FlowNet2_src/example/0img1.ppm')
  im1 = cv2.resize(imread(prev),(480,360))
  im2 = cv2.resize(imread(after),(480,360))
 

  # B x 3(RGB) x 2(pair) x H x W
  ims = np.array([[im1, im2]]).transpose((0, 4, 1, 2, 3)).astype(np.float32)
  ims = torch.from_numpy(ims)
  ims_v = Variable(ims.cuda(), requires_grad=False)

  # Build model
  flownet2 = FlowNet2()
  path = '/home/mike/workspace/segnet/pytorch_flownet2/FlowNet2_src/pretrained/FlowNet2_checkpoint.pth.tar'
  pretrained_dict = torch.load(path)['state_dict']
  model_dict = flownet2.state_dict()
  pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
  model_dict.update(pretrained_dict)
  flownet2.load_state_dict(model_dict)
  flownet2.cuda()
  
  t1 = time.time()
  pred_flow = flownet2(ims_v).cpu().data
  t2 = time.time()
  print t2 - t1
  pred_flow = pred_flow[0].numpy().transpose((1,2,0))
  flow_im = flow_to_image(pred_flow)

  # Visualization
  plt.imshow(flow_im)
  plt.show()
  #pdb.set_trace()
  return pred_flow
  
if __name__ =='__main__':
  #flow = demo(prev,after)  
  flow = demo('/TOSHIBA_4TB/andyhaha/CamVid/video/Test_Video/01TP/01TP_003749.png', '/TOSHIBA_4TB/andyhaha/CamVid/video/Test_Video/01TP/01TP_003750.png') 

