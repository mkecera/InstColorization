mkdir ./checkpoints/coco_mask_tiny_neighbor
cp ./checkpoints/coco_finetuned_mask_256_ffs/latest_net_G.pth ./checkpoints/coco_mask_tiny_neighbor/latest_net_GF.pth

# Train
bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=10.7G -L /bin/bash -q gpu "source ~/.bashrc && $(which python) train_with_validation.py --stage fusion --name coco_mask_tiny_neighbor --sample_p 1.0 --niter 40 --niter_decay 60 --lr 0.00005 --model train --load_model --display_ncols 4 --fineSize 256 --batch_size 1 --display_freq 1 --print_freq 1 --save_epoch_freq 1 --train_img_dir train_tiny --no_html --display_id 0 --test_img_dir val_tiny"

# Inference Bounding box
bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=10.7G -L /bin/bash -q gpu "source ~/.bashrc && $(which python) inference_bbox.py --test_img_dir test_small"

# Test
bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=10.7G -L /bin/bash -q gpu "source ~/.bashrc && $(which python) test_fusion.py --name test_fusion_baseline_new --sample_p 1.0 --model fusion --fineSize 256 --test_img_dir unsplash --results_img_dir results_unsplash_baseline_new --model_dir coco_finetuned_mask_256"

bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=10.7G -L /bin/bash -q gpu "source ~/.bashrc && $(which python) test_fusion.py --name test_fusion_baseline_new --sample_p 1.0 --model fusion --fineSize 256 --test_img_dir test_small --results_img_dir results_test_small_baseline_new --model_dir coco_finetuned_mask_256"

bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=10.7G -L /bin/bash -q gpu "source ~/.bashrc && $(which python) calculate_metrics.py --test_img_dir test_small --results_img_dir results_test_small_baseline_new"