import glob
from options.train_options import TestOptions
from models import create_model
import torch
from tqdm import trange, tqdm
from fusion_dataset import Fusion_Testing_Dataset
from util import util
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import multiprocessing
import pandas as pd

multiprocessing.set_start_method('spawn', True)
torch.backends.cudnn.benchmark = True

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

    epoch_weights = glob.glob(f'checkpoints/coco_mask/[0-9]*.pth')
    epoch_weights = [i.split('/')[-1] for i in epoch_weights]

    model = create_model(opt)
    # model.setup_to_test('coco_finetuned_mask_256')

    g_list = []
    l1_list = []
    losses_list = []

    for weight_file in epoch_weights:
        model.setup_to_test(f'coco_mask/{weight_file}')
        count_empty = 0
        for data_raw in tqdm(dataset_loader, dynamic_ncols=True):
            data_raw['full_img'][0] = data_raw['full_img'][0].cuda()
            if data_raw['empty_box'][0] == 0:
                data_raw['cropped_img'][0] = data_raw['cropped_img'][0].cuda()
                box_info = data_raw['box_info'][0]
                box_info_2x = data_raw['box_info_2x'][0]
                box_info_4x = data_raw['box_info_4x'][0]
                box_info_8x = data_raw['box_info_8x'][0]
                cropped_data = util.get_colorization_data(data_raw['cropped_img'], opt, ab_thresh=0, p=opt.sample_p)
                full_img_data = util.get_colorization_data(data_raw['full_img'], opt, ab_thresh=0, p=opt.sample_p)
                model.set_input(cropped_data)
                model.set_fusion_input(full_img_data, [box_info, box_info_2x, box_info_4x, box_info_8x])
                model.forward()
            else:
                count_empty += 1
                full_img_data = util.get_colorization_data(data_raw['full_img'], opt, ab_thresh=0, p=opt.sample_p)
                model.set_forward_without_box(full_img_data)
        losses = model.get_current_losses()
        g_list.append(losses['G'])
        l1_list.append(losses['L1'])
    df = pd.DataFrame(l1_list, columns=['L1'])
    filename = f"./loss_results/validation_losses_lr_{str(opt.lr)}_epochs_{str(opt.niter)}.csv"
    df.to_csv(filename)
    print(f"Saved csv of validation results in {filename}")
