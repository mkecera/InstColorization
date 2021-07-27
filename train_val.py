import time
from options.train_options import TrainOptions
from models import create_model
from util.visualizer import Visualizer

import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import trange, tqdm

from fusion_dataset import *
from util import util
import os

if __name__ == '__main__':
    opt = TrainOptions().parse()
    if opt.stage == 'full':
        dataset = Training_Full_Dataset(opt)
    elif opt.stage == 'instance':
        dataset = Training_Instance_Dataset(opt)
    elif opt.stage == 'fusion':
        dataset = Training_Fusion_Dataset(opt)
    else:
        print('Error! Wrong stage selection!')
        exit()
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8)

    validation_dataset = Fusion_Testing_Dataset(opt)
    val_dataset_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, num_workers=2)

    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)
    print('#test images = %d' % len(validation_dataset))

    model = create_model(opt)
    val_model = create_model(opt)
    model.setup(opt)

    opt.display_port = 8098
    visualizer = Visualizer(opt)
    total_steps = 0

    if opt.stage == 'full' or opt.stage == 'instance':
        for epoch in trange(opt.epoch_count, opt.niter + opt.niter_decay, desc='epoch', dynamic_ncols=True):
            epoch_iter = 0

            for data_raw in tqdm(dataset_loader, desc='batch', dynamic_ncols=True, leave=False):
                total_steps += opt.batch_size
                epoch_iter += opt.batch_size

                data_raw['rgb_img'] = [data_raw['rgb_img']]
                data_raw['gray_img'] = [data_raw['gray_img']]

                input_data = util.get_colorization_data(data_raw['gray_img'], opt, p=1.0, ab_thresh=0)
                gt_data = util.get_colorization_data(data_raw['rgb_img'], opt, p=1.0, ab_thresh=10.0)
                if gt_data is None:
                    continue
                if(gt_data['B'].shape[0] < opt.batch_size):
                    continue
                input_data['B'] = gt_data['B']
                input_data['hint_B'] = gt_data['hint_B']
                input_data['mask_B'] = gt_data['mask_B']

                visualizer.reset()
                model.set_input(input_data)
                model.optimize_parameters()

                if total_steps % opt.display_freq == 0:
                    save_result = total_steps % opt.update_html_freq == 0
                    if opt.display_id > 0:
                        visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                if total_steps % opt.print_freq == 0:
                    losses = model.get_current_losses()
                    if opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

            if epoch % opt.save_epoch_freq == 0:
                model.save_networks('latest')
                model.save_networks(epoch)
            model.update_learning_rate()
    elif opt.stage == 'fusion':
        for epoch in trange(opt.epoch_count, opt.niter + opt.niter_decay, desc='epoch', dynamic_ncols=True):
            epoch_iter = 0

            for data_raw in tqdm(dataset_loader, desc='batch', dynamic_ncols=True, leave=False):
                # print(data_raw)
                total_steps += opt.batch_size
                epoch_iter += opt.batch_size
                box_info = data_raw['box_info'][0]
                box_info_2x = data_raw['box_info_2x'][0]
                box_info_4x = data_raw['box_info_4x'][0]
                box_info_8x = data_raw['box_info_8x'][0]
                cropped_input_data = util.get_colorization_data(data_raw['cropped_gray'], opt, p=1.0, ab_thresh=0)
                cropped_gt_data = util.get_colorization_data(data_raw['cropped_rgb'], opt, p=1.0, ab_thresh=10.0)
                full_input_data = util.get_colorization_data(data_raw['full_gray'], opt, p=1.0, ab_thresh=0)
                full_gt_data = util.get_colorization_data(data_raw['full_rgb'], opt, p=1.0, ab_thresh=10.0)
                if cropped_gt_data is None or full_gt_data is None:
                    continue
                cropped_input_data['B'] = cropped_gt_data['B']
                full_input_data['B'] = full_gt_data['B']
                visualizer.reset()
                model.set_input(cropped_input_data)
                model.set_fusion_input(full_input_data, [box_info, box_info_2x, box_info_4x, box_info_8x])
                model.optimize_parameters()

                if total_steps % opt.display_freq == 0:
                    save_result = total_steps % opt.update_html_freq == 0
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                if total_steps % opt.print_freq == 0:
                    losses = model.get_current_losses()
                    # print(losses)
                    if opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)
            if epoch % opt.save_epoch_freq == 0:
                model.save_fusion_epoch(epoch)
            model.update_learning_rate()

            val_model = model.copy()
            count_empty = 0
            for data_raw in tqdm(val_dataset_loader, dynamic_ncols=True):
                # if os.path.isfile(join(save_img_path, data_raw['file_id'][0] + '.png')) is True:
                #     continue
                data_raw['full_img'][0] = data_raw['full_img'][0].cuda()
                if data_raw['empty_box'][0] == 0:
                    data_raw['cropped_img'][0] = data_raw['cropped_img'][0].cuda()
                    box_info = data_raw['box_info'][0]
                    box_info_2x = data_raw['box_info_2x'][0]
                    box_info_4x = data_raw['box_info_4x'][0]
                    box_info_8x = data_raw['box_info_8x'][0]
                    cropped_data = util.get_colorization_data(data_raw['cropped_img'], opt, ab_thresh=0, p=opt.sample_p)
                    full_img_data = util.get_colorization_data(data_raw['full_img'], opt, ab_thresh=0, p=opt.sample_p)
                    val_model.set_input(cropped_data)
                    val_model.set_fusion_input(full_img_data, [box_info, box_info_2x, box_info_4x, box_info_8x])
                    val_model.forward()
                else:
                    count_empty += 1
                    full_img_data = util.get_colorization_data(data_raw['full_img'], opt, ab_thresh=0, p=opt.sample_p)
                    val_model.set_forward_without_box(full_img_data)
                # model.save_current_imgs(join(save_img_path, data_raw['file_id'][0] + '.png'))
            print('{0} images without bounding boxes'.format(count_empty))
            print(f'Validation Loss: {val_model.get_current_losses()}')
            # print(json.dumps(data, indent=4))

    else:
        print('Error! Wrong stage selection!')
        exit()
