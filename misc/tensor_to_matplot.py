from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Path to the TensorBoard log directory
# log_dir_1 = "/data/home/rim36739/disk/runs/seg/Cycle_val_last_try_IN_THESIS/val/True/50/fold_0001_216/with_gen/"
# log_dir_2 = "/data/home/rim36739/disk/runs/seg/Cycle_val_last_try_IN_THESIS/val/True/50/fold_0002_145/with_gen/"
# log_dir_3 = "/data/home/rim36739/disk/runs/seg/Cycle_val_last_try_IN_THESIS/val/True/50/fold_0003_148/with_gen/"
# log_dir_4 = "/data/home/rim36739/disk/runs/seg/Cycle_val_last_try_IN_THESIS/val/True/50/fold_0004_177/with_gen/"
# log_dir_5 = "/data/home/rim36739/disk/runs/seg/Cycle_val_last_try_IN_THESIS/val/True/50/fold_0005_201/with_gen/"
# log_dir_6 = "/data/home/rim36739/disk/runs/seg/Cycle_val_last_try_IN_THESIS/val/True/50/fold_0007_217/with_gen/"
log_dir_1 = "/data/home/rim36739/disk/runs/seg_gan/SEGGENNET_mehr_cycle_sonst_gleich/with_cycleTrue/0001_216/val/segmentation_sup"
log_dir_2 = "/data/home/rim36739/disk/runs/seg_gan/SEGGENNET_mehr_cycle_sonst_gleich/with_cycleTrue/0002_145/val/segmentation_sup"
log_dir_3 = "/data/home/rim36739/disk/runs/seg_gan/SEGGENNET_mehr_cycle_sonst_gleich/with_cycleTrue/0003_148/val/segmentation_sup"
log_dir_4 = "/data/home/rim36739/disk/runs/seg_gan/SEGGENNET_mehr_cycle_sonst_gleich/with_cycleTrue/0004_177/val/segmentation_sup"
log_dir_5 = "/data/home/rim36739/disk/runs/seg_gan/SEGGENNET_mehr_cycle_sonst_gleich/with_cycleTrue/0005_201/val/segmentation_sup"
log_dir_6 = "/data/home/rim36739/disk/runs/seg_gan/SEGGENNET_mehr_cycle_sonst_gleich/with_cycleTrue/0007_217/val/segmentation_sup"


log_dir_list = [log_dir_1, log_dir_2, log_dir_3, log_dir_4, log_dir_5, log_dir_6]

for idx, log_dir in enumerate(log_dir_list):
    # Load TensorBoard data
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    # Get scalar data from TensorBoard
    train_loss = event_acc.Scalars('heavy_smoke_domain/IoU/mIoU')
    #valid_loss = event_acc.Scalars('valid_loss')

    # Extract x and y values from scalar data
    train_steps, train_loss_values = zip(*[(s.step + 1, s.value) for s in train_loss])
    #valid_steps, valid_loss_values = zip(*[(s.step, s.value) for s in valid_loss])

    #sns.set_theme()
    sns.set()

    # Plot the data using matplotlib
    plt.plot(train_steps, train_loss_values, label=f'Fold {idx + 1}')
    #plt.plot(valid_steps, valid_loss_values, label='Validation Loss')
    plt.xticks(fontsize=16)

    # set the font size of the y-axis tick labels
    plt.yticks(fontsize=16)
    plt.legend(fontsize=13, loc = "upper left")
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('mIoU',fontsize=16)
plt.savefig('seggan_iou_hs.pdf', format='pdf',bbox_inches='tight')
    #plt.show()
############################################# behalten für schrift größer machen

"""# Path to the TensorBoard log directory
log_dir_1 = "/data/home/rim36739/disk/runs/star_ssim/fold_0"
log_dir_2 = "/data/home/rim36739/disk/runs/star_ssim/fold_1"
log_dir_3 = "/data/home/rim36739/disk/runs/star_ssim/fold_2"
log_dir_4 = "/data/home/rim36739/disk/runs/star_ssim/fold_3"
log_dir_5 = "/data/home/rim36739/disk/runs/star_ssim/fold_4"
log_dir_6 = "/data/home/rim36739/disk/runs/star_ssim/fold_5"

log_dir_list = [log_dir_1, log_dir_2, log_dir_3, log_dir_4, log_dir_5, log_dir_6]

for idx, log_dir in enumerate(log_dir_list):
    # Load TensorBoard data
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    # Get scalar data from TensorBoard
    train_loss = event_acc.Scalars('fid')
    #valid_loss = event_acc.Scalars('valid_loss')

    # Extract x and y values from scalar data
    train_steps, train_loss_values = zip(*[(s.step, s.value) for s in train_loss])
    #valid_steps, valid_loss_values = zip(*[(s.step, s.value) for s in valid_loss])

    #sns.set_theme()
    sns.set()

    # Plot the data using matplotlib
    plt.plot(train_steps, train_loss_values, label=f'Fold {idx + 1}')
    #plt.plot(valid_steps, valid_loss_values, label='Validation Loss')
    plt.xticks(fontsize=16)

    # set the font size of the y-axis tick labels
    plt.yticks(fontsize=16)
    plt.legend(fontsize=14,loc = "upper left")
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('FID',fontsize=16)
plt.savefig('star_fid.pdf', format='pdf',bbox_inches='tight')"""
    #plt.show()
############################################# behalten für schrift größer machen

    #sinplot()
    #sinplot()