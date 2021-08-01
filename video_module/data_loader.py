import socket
import numpy as np
from torchvision import datasets, transforms
import os
import cv2
from torch.utils.data import Dataset, DataLoader
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import pickle
import torch


class DataSetLoader(Dataset):
    def __init__(self, is_training=True, data_root='./video_module/data/training', seq_len=20, image_size=256):
        self.path = data_root
        self.seq_len = seq_len
        self.image_size = image_size
        self.step_length = 0.1
        self.seed_is_set = False
        self.channels = 3
        self.dirs = [f.path for f in os.scandir(self.path) if (f.is_dir() and os.path.isfile(f.path + '/data_X.pkl'))]
        # Uncomment if you want to convert the training set again
        # self.convert_to_pickle()
        self.N = 1200
        self.dir_index = 0
        self.i = 0

    def __getitem__(self, index):
        if index == 0:
            self.dir_index = 0
            self.i = 0

        try:
            X, y = self.load_pickle(self.dirs[self.dir_index])
        except IndexError:
            self.dir_index = 0
            self.i = 0
            X, y = self.load_pickle(self.dirs[self.dir_index])

        if self.i == X.shape[0] - 2:
            self.i = 0
            self.dir_index += 1
        else:
            self.i += 1

        try:
            final = np.stack((X, y), axis=1)[self.i]
            final = torch.tensor(final).float()
        except TypeError:
            print('this sub directory is corrupted')
            self.dir_index += 1
            self.i = 0
            return self.__getitem__(index + 1)

        return final


    def __len__(self):
        return self.N

    def split_data(self, X, y):
        for i in range(0, len(X), self.seq_len):
            yield X[i:i + self.seq_len], y[i:i + self.seq_len]


    def convert_to_pickle(self):
        dirs = [f.path for f in os.scandir(self.path) if f.is_dir()]
        for sub_directory in dirs:
            if (os.path.isfile(sub_directory + '/data_X.pkl') and os.path.isfile(sub_directory + '/data_y.pkl')) and (
            not os.path.isfile(sub_directory + '/colorized_images/')):
                continue

            X = []
            y = []
            image_id_list = [f for f in listdir(sub_directory + '/colorized_images/') if
                             isfile(join(sub_directory + '/colorized_images/', f))]
            image_id_list.sort()

            for img in tqdm(image_id_list):
                input_img_path = sub_directory + '/colorized_images/' + img

                target_img_path = sub_directory + '/target/' + img[:-3] + 'jpg'

                if not os.path.isfile(input_img_path) or not os.path.isfile(target_img_path):
                    continue

                img = cv2.imread(input_img_path)
                img = cv2.resize(img, (self.image_size, self.image_size))

                target = cv2.imread(target_img_path)
                target = cv2.resize(target, (self.image_size, self.image_size))

                X.append(img)
                y.append(target)

            split_size = len(X) // self.seq_len
            X_mid = np.array_split(np.array(X), split_size)[1:]
            y_mid = np.array_split(np.array(y), split_size)[1:]

            self.save_pickle(np.array(X_mid), np.array(y_mid), sub_directory)


    def save_pickle(self, X, y, dir):
        path_X = dir + '/data_X.pkl'
        path_y = dir + '/data_y.pkl'
        with open(path_X, 'wb') as fp:
            pickle.dump(X, fp)

        with open(path_y, 'wb') as fp:
            pickle.dump(y, fp)

    def load_pickle(self, dir):
        path_X = dir + '/data_X.pkl'
        path_y = dir + '/data_y.pkl'
        with open(path_X, 'rb') as fp:
            X = pickle.load(fp)
        with open(path_y, 'rb') as fp:
            y = pickle.load(fp)
        return np.array(X), np.array(y)