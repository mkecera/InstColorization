import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import skimage
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.io import imread

l1 = pd.read_csv('loss_log/coco_mask_tiny_l1_20210801_162629_loss.txt', index_col=0)
l2 = pd.read_csv('loss_log/coco_mask_tiny_l2_20210731_121023_loss.txt', index_col=0)
huber = pd.read_csv('loss_log/coco_mask_tiny_huber_20210731_114338_loss.txt', index_col=0)

ce = pd.read_csv('loss_log/coco_mask_tiny_ce_20210801_172231_loss.txt', index_col=0)
ce = ce[ce.train_G_loss<10]
# ce_only = pd.read_csv('loss_log/coco_mask_tiny_ce_only_20210801_160917_loss.txt', index_col=0)
neighbor = pd.read_csv('loss_log/coco_mask_tiny_neighbor_20210802_074236_loss.txt', index_col=0)

##
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
l1.train_L1_loss.plot.line(label='train_loss', ax=axes[0])
l1.val_L1_loss.plot.line(label='val_loss', ax=axes[0])
axes[0].set_title('Learning curve with loss function = L1')
plt.legend()

l2.train_L1_loss.plot.line(label='train_loss', ax=axes[1])
l2.val_L1_loss.plot.line(label='val_loss', ax=axes[1])
axes[1].set_title('Learning curve with loss function = L2')
plt.legend()

huber.train_L1_loss.plot.line(label='train_loss', ax=axes[2])
huber.val_L1_loss.plot.line(label='val_loss', ax=axes[2])
axes[2].set_title('Learning curve with loss function = Huber')
plt.legend()

plt.tight_layout()
plt.savefig('loss_fig/learning_curve_pix2pix_loss.png')

##
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
l1.train_psnr.plot.line(label='Loss = L1', ax=axes[0][0])
l2.train_psnr.plot.line(label='Loss = L2', ax=axes[0][0])
huber.train_psnr.plot.line(label='Loss = Huber', ax=axes[0][0])
axes[0][0].set_title('Train PSNR, pix2pix losses')
axes[0][0].legend()

l1.val_psnr.plot.line(label='Loss = L1', ax=axes[0][1])
l2.val_psnr.plot.line(label='Loss = L2', ax=axes[0][1])
huber.val_psnr.plot.line(label='Loss = Huber', ax=axes[0][1])
axes[0][1].set_title('Validation PSNR, pix2pix losses')
# axes[0][1].legend()

l1.train_ssim.plot.line(label='Loss = L1', ax=axes[1][0])
l2.train_ssim.plot.line(label='Loss = L2', ax=axes[1][0])
huber.train_ssim.plot.line(label='Loss = Huber', ax=axes[1][0])
axes[1][0].set_title('Train SSIM, pix2pix losses')
# axes[1][0].legend()

l1.val_ssim.plot.line(label='Loss = L1', ax=axes[1][1])
l2.val_ssim.plot.line(label='Loss = L2', ax=axes[1][1])
huber.val_ssim.plot.line(label='Loss = Huber', ax=axes[1][1])
axes[1][1].set_title('Validation SSIM, pix2pix losses')
# axes[1][1].legend()

plt.tight_layout()
plt.savefig('loss_fig/metrics_pix2pix_loss.png')

##
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
huber.train_L1_loss.plot.line(label='train_loss', ax=axes[0])
huber.val_L1_loss.plot.line(label='val_loss', ax=axes[0])
axes[0].set_title('Loss = Huber')

ce.train_G_loss.plot.line(label='train_loss', ax=axes[1])
ce.val_G_loss.plot.line(label='val_loss', ax=axes[1])
axes[1].set_title('Loss = Huber+CrossEntropy')

neighbor.train_G_loss.plot.line(label='train_loss', ax=axes[2])
neighbor.val_G_loss.plot.line(label='val_loss', ax=axes[2])
axes[2].set_title('Loss = Huber+CrossEntropy+Neighbour')
plt.legend()

plt.tight_layout()
plt.savefig('loss_fig/learning_curve_composite_loss.png')
##
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
huber.train_psnr.plot.line(label='Loss = Huber', ax=axes[0][0])
ce.train_psnr.plot.line(label='Loss = Huber+CrossEntropy', ax=axes[0][0])
neighbor.train_psnr.plot.line(label='Loss = Huber+CrossEntropy+Neighbour', ax=axes[0][0])
axes[0][0].set_title('Train PSNR, composite losses')
axes[0][0].legend()

huber.val_psnr.plot.line(label='Loss = Huber', ax=axes[0][1])
ce.val_psnr.plot.line(label='Loss = Huber+CrossEntropy', ax=axes[0][1])
neighbor.val_psnr.plot.line(label='Loss = Huber+CrossEntropy+Neighbour', ax=axes[0][1])
axes[0][1].scatter(6, huber.loc[6, 'val_psnr'], color='r')
axes[0][1].scatter(36, neighbor.loc[36, 'val_psnr'], color='r')
axes[0][1].set_title('Validation PSNR, composite losses')
# axes[0][1].legend()

huber.train_ssim.plot.line(label='Loss = Huber', ax=axes[1][0])
ce.train_ssim.plot.line(label='Loss = Huber+CrossEntropy', ax=axes[1][0])
neighbor.train_ssim.plot.line(label='Loss = Huber+CrossEntropy+Neighbour', ax=axes[1][0])
axes[1][0].set_title('Train SSIM, composite losses')
# axes[1][0].legend()

huber.val_ssim.plot.line(label='Loss = Huber', ax=axes[1][1])
ce.val_ssim.plot.line(label='Loss = Huber+CrossEntropy', ax=axes[1][1])
neighbor.val_ssim.plot.line(label='Loss = Huber+CrossEntropy+Neighbour', ax=axes[1][1])
axes[1][1].scatter(6, huber.loc[6, 'val_ssim'], color='r')
axes[1][1].scatter(36, neighbor.loc[36, 'val_ssim'], color='r')
axes[1][1].set_title('Validation SSIM, composite losses')
# axes[1][1].legend()

plt.tight_layout()
plt.savefig('loss_fig/metrics_composite_loss.png')
##
'''Examples'''
# for id in metric_neighbor_8.index:
#     # id = (metric_neighbor_8[1] - metric_baseline_new[1]).nlargest(24).index[i]
#     if id >= 100000:
#         id = '000000'+str(int(id))
#     elif id >= 10000:
#         id = '0000000' + str(int(id))
#     else:
#         id = '00000000' + str(int(id))
id = '000000274573'
# neighbor_1 = imread(f'results_test_small_neighbor_36/{id}.png')
neighbor_2 = imread(f'results_test_small_neighbor_8/{id}.png')
# fig, axes = plt.subplots(1, 2, figsize=(8, 4))
# axes[0].imshow(neighbor_1)
# axes[0].set_title('36')
# axes[1].imshow(neighbor_2)
# axes[1].set_title('8')

true_img = imread(f'test_small/{id}.jpg')
true_img = resize(true_img, (256, 256))
gray_img = rgb2gray(true_img)
baseline_img = imread(f'results_test_small_baseline_new/{id}.png')
huber_img = imread(f'results_test_small_huber_best/{id}.png')
neighbor_img = neighbor_2

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(true_img)
axes[0].set_title('Original')
# axes[1].imshow(gray_img, cmap=plt.cm.gray)
# axes[1].set_title('Grayscale')
axes[1].imshow(baseline_img)
axes[1].set_title('Baseline model')
axes[2].imshow(huber_img)
axes[2].set_title('Huber Loss, epoch 6')
axes[3].imshow(neighbor_img)
axes[3].set_title('Composite Loss, epoch 36')
for i in range(4):
    axes[i].axis('off')
fig.tight_layout()

plt.savefig(f'fig_comparison/{id}.png')
plt.close()
##

