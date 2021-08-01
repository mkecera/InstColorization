import multiprocessing
import sys

import torch

from options.train_options import TestOptions
from util import util

multiprocessing.set_start_method('spawn', True)

from skimage import io

torch.backends.cudnn.benchmark = True

sys.argv = [sys.argv[0]]
# opt = TestOptions().parse()

image = io.imread('example/000000046872.jpg')
image = torch.Tensor(image.transpose())
data_lab = util.xyz2lab(util.rgb2xyz(torch.unsqueeze(image, dim=0)))
a = data_lab[:,1,:,:]
b = data_lab[:,2,:,:]

##
a = a.reshape(250240)
b = b.reshape(250240)


##

