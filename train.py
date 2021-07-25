import time
from options.train_options import TrainOptions
from models import create_model
from util.visualizer import Visualizer
from tqdm import trange, tqdm
import pandas as pd
from fusion_dataset import *
from util import util, plots

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

    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)

    opt.display_port = 8098
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

                model.set_input(input_data)
                model.optimize_parameters()

                if total_steps % opt.display_freq == 0:
                    save_result = total_steps % opt.update_html_freq == 0

                if total_steps % opt.print_freq == 0:
                    losses = model.get_current_losses()

            if epoch % opt.save_epoch_freq == 0:
                model.save_networks('latest')
                model.save_networks(epoch)
            model.update_learning_rate()
    elif opt.stage == 'fusion':
        g_list = []
        l1_list = []
        losses_list = []

        for epoch in trange(opt.epoch_count, opt.niter + opt.niter_decay, desc='epoch', dynamic_ncols=True):
            epoch_iter = 0

            for data_raw in tqdm(dataset_loader, desc='batch', dynamic_ncols=True, leave=False):
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
                model.set_input(cropped_input_data)
                model.set_fusion_input(full_input_data, [box_info, box_info_2x, box_info_4x, box_info_8x])
                model.optimize_parameters()

                if total_steps % opt.display_freq == 0:
                    save_result = total_steps % opt.update_html_freq == 0

                if total_steps % opt.print_freq == 0:
                    losses = model.get_current_losses()

            if epoch % opt.save_epoch_freq == 0:
                model.save_fusion_epoch(epoch)
            model.update_learning_rate()
            losses = model.get_current_losses()
            g_list.append(losses['G'])
            l1_list.append(losses['L1'])
        df = pd.DataFrame(l1_list, columns=['L1'])
        filename = f"./loss_results/{opt.name}_training_losses_lr_{str(opt.lr)}_epochs_{str(opt.niter)}.csv"
        df.to_csv(filename)
        print(f"Saved csv of training results in {filename}")
    else:
        print('Error! Wrong stage selection!')
        exit()
