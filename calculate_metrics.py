import os
from os.path import join
import time
from options.train_options import TrainOptions, TestOptions
from models import create_model
from util.visualizer import Visualizer

import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import trange, tqdm

from fusion_dataset import Fusion_Testing_Dataset
from util import util
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import multiprocessing
multiprocessing.set_start_method('spawn', True)

torch.backends.cudnn.benchmark = True

import cv2
import skimage
import pandas as pd

if __name__ == '__main__':
    opt = TestOptions().parse()
    save_img_path = opt.results_img_dir
    if os.path.isdir(save_img_path) is False:
        print('Create path: {0}'.format(save_img_path))
        os.makedirs(save_img_path)
    opt.batch_size = 1
    dataset = Fusion_Testing_Dataset(opt)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2)

    dataset_size = len(dataset)
    print('#Testing images = %d' % dataset_size)

    # model = create_model(opt)
    # model.setup_to_test('coco_finetuned_mask_256')
    # model.setup_to_test('coco_tiny_fusion')

    count_empty = 0
    all_psnr = np.array([])
    all_ssim = np.array([])
    all_ids = np.array([])
    for data_raw in tqdm(dataset_loader, dynamic_ncols=True):
        
        # calculate metrics
        # get true image
        true_img = cv2.imread(join('./', opt.test_img_dir, data_raw['file_id'][0] + '.jpg'))
        # resize true image
        true_img = cv2.resize(true_img, (256, 256), interpolation=cv2.INTER_AREA)

        # get test image
        test_img = cv2.imread(join('./', opt.results_img_dir, data_raw['file_id'][0] + '.png'))

        # calculate psnr
        psnr = skimage.metrics.peak_signal_noise_ratio(true_img, test_img)

        # calculate ssim
        ssim = skimage.metrics.structural_similarity(true_img, test_img, multichannel=True)

        # save to array
        all_ids = np.append(all_ids, data_raw['file_id'][0])
        all_psnr = np.append(all_psnr, psnr)
        all_ssim = np.append(all_ssim, ssim)

    result = pd.DataFrame([all_ids, all_psnr, all_ssim])
    result.to_csv(f'metrics_test_small/{opt.results_img_dir}.csv')