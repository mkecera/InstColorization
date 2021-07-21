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


def calculate_loss_metrics(directory_predicted, directory_label):
    all_psnr = np.array([])
    all_ssim = np.array([])
    for filename in os.listdir(directory_predicted):
        if filename.endswith(".png"):
            file_name = filename[:-4]
            true_img = cv2.imread(join('./', directory_label, file_name + '.jpg'))
            # resize true image
            true_img = cv2.resize(true_img, (256, 256), interpolation=cv2.INTER_AREA)

            # get test image
            test_img = cv2.imread(join('./', directory_predicted, file_name + '.png'))

            # calculate psnr
            psnr = skimage.metrics.peak_signal_noise_ratio(true_img, test_img)

            # calculate ssim
            ssim = skimage.metrics.structural_similarity(true_img, test_img, multichannel=True)

            # save to array
            all_psnr = np.append(all_psnr, psnr)
            all_ssim = np.append(all_ssim, ssim)

    print('{0} PSNR'.format(np.average(all_psnr)))
    print('{0} SSIM'.format(np.average(all_ssim)))


if __name__ == '__main__':
    directory_predicted = './data/colorized_images/'
    directory_label = './data/target/'
    calculate_loss_metrics(directory_predicted, directory_label)
