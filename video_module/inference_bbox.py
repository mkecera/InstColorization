from os.path import join, isfile, isdir
from os import listdir
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from argparse import ArgumentParser

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

import torch
from tqdm import tqdm

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

parser = ArgumentParser()
parser.add_argument("--test_img_dir", type=str, default='./data/training/', help='testing images folder')
parser.add_argument('--filter_no_obj', action='store_true')
args = parser.parse_args()

# root_path, dirs, files = os.walk(args.test_img_dir):
dirs = [f.path for f in os.scandir(args.test_img_dir) if f.is_dir() ]

for sequence_path in dirs:
    input_dir = sequence_path + '/input/'
    root_dir = sequence_path + '/'

    image_list = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
    output_path = "{0}_bbox".format(root_dir)

    # output_path = './data/output_npz/'
    if os.path.isdir(output_path) is False:
        print('Create path: {0}'.format(output_path))
        os.makedirs(output_path)

    for image_path in tqdm(image_list):
        img = cv2.imread(join(input_dir, image_path))
        lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_image)
        l_stack = np.stack([l_channel, l_channel, l_channel], axis=2)
        outputs = predictor(l_stack)
        save_path = join(output_path, image_path.split('.')[0])
        pred_bbox = outputs["instances"].pred_boxes.to(torch.device('cpu')).tensor.numpy()
        pred_scores = outputs["instances"].scores.cpu().data.numpy()
        if args.filter_no_obj is True and pred_bbox.shape[0] == 0:
            print('delete {0}'.format(image_path))
            os.remove(join(input_dir, image_path))
            continue
        np.savez(save_path, bbox = pred_bbox, scores = pred_scores)
