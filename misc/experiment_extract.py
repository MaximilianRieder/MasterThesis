import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

#########################################################################################################################################  
    # Path to the TensorBoard log directory
# log_dir_1 = "/data/home/rim36739/disk/runs/seg/Cycle_val_last_try/val/True/50/fold_0001_216/without_gen"
# log_dir_2 = "/data/home/rim36739/disk/runs/seg/Cycle_val_last_try/val/True/50/fold_0002_145/without_gen"
# log_dir_3 = "/data/home/rim36739/disk/runs/seg/Cycle_val_last_try/val/True/50/fold_0003_148/without_gen"
# log_dir_4 = "/data/home/rim36739/disk/runs/seg/Cycle_val_last_try/val/True/50/fold_0004_177/without_gen"
# log_dir_5 = "/data/home/rim36739/disk/runs/seg/Cycle_val_last_try/val/True/50/fold_0005_201/without_gen"
# log_dir_6 = "/data/home/rim36739/disk/runs/seg/Cycle_val_last_try/val/True/50/fold_0007_217/without_gen"
log_dir_1 = "/data/home/rim36739/disk/runs/seg_gan/SEGGENNET_mehr_cycle_sonst_gleich/with_cycleTrue/0001_216/val/segmentation_sup"
log_dir_2 = "/data/home/rim36739/disk/runs/seg_gan/SEGGENNET_mehr_cycle_sonst_gleich/with_cycleTrue/0002_145/val/segmentation_sup"
log_dir_3 = "/data/home/rim36739/disk/runs/seg_gan/SEGGENNET_mehr_cycle_sonst_gleich/with_cycleTrue/0003_148/val/segmentation_sup"
log_dir_4 = "/data/home/rim36739/disk/runs/seg_gan/SEGGENNET_mehr_cycle_sonst_gleich/with_cycleTrue/0004_177/val/segmentation_sup"
log_dir_5 = "/data/home/rim36739/disk/runs/seg_gan/SEGGENNET_mehr_cycle_sonst_gleich/with_cycleTrue/0005_201/val/segmentation_sup"
log_dir_6 = "/data/home/rim36739/disk/runs/seg_gan/SEGGENNET_mehr_cycle_sonst_gleich/with_cycleTrue/0007_217/val/segmentation_sup"

domains = ["all_domains/Dice/mDice","heavy_smoke_domain/Dice/mDice","slightly_smoked_domain/Dice/mDice","no_smoke_domain/Dice/mDice"] 
log_dir_list = [log_dir_1, log_dir_2, log_dir_3, log_dir_4, log_dir_5, log_dir_6]
for d in domains:
    acc_dom = 0
    for idx, log_dir in enumerate(log_dir_list):
        
        event_file = log_dir
        tag = d
        step = 29

        event_acc = EventAccumulator(event_file)
        event_acc.Reload()

        # Get the values of the tag at the specified step
        values = event_acc.Scalars(tag)
        value_at_step = next((v for v in values if v.step == step), None)

        acc_dom += value_at_step.value
        if value_at_step is not None:
            print(f"Value of {tag} at for video {idx + 1} step {step}: {round(value_at_step.value*100,1)}")
        else:
            print(f"No values found for tag {tag} at step {step}")
    print(f"accumulated for dom {d} : {round((acc_dom / 6) * 100,1)}")
    print(f"accumulated for dom {d} percent: {(acc_dom / 6) * 100}")

"""#########################################################################################################################################  
    # Path to the TensorBoard log directory
log_dir_1 = "/data/home/rim36739/disk/saved_old_results/Segmentation/Baseline_cycle_final_912_513/val/fold_0001_216/with_gen"
log_dir_2 = "/data/home/rim36739/disk/saved_old_results/Segmentation/Baseline_cycle_final_912_513/val/fold_0002_145/with_gen"
log_dir_3 = "/data/home/rim36739/disk/saved_old_results/Segmentation/Baseline_cycle_final_912_513/val/fold_0003_148/with_gen"
log_dir_4 = "/data/home/rim36739/disk/saved_old_results/Segmentation/Baseline_cycle_final_912_513/val/fold_0004_177/with_gen"
log_dir_5 = "/data/home/rim36739/disk/saved_old_results/Segmentation/Baseline_cycle_final_912_513/val/fold_0005_201/with_gen"
log_dir_6 = "/data/home/rim36739/disk/saved_old_results/Segmentation/Baseline_cycle_final_912_513/val/fold_0007_217/with_gen"

log_dir_list = [log_dir_1, log_dir_2, log_dir_3, log_dir_4, log_dir_5, log_dir_6]

for idx, log_dir in enumerate(log_dir_list):
    
    event_file = log_dir
    tag = "no_smoke_domain/IoU/mIoU"
    step = 59

    event_acc = EventAccumulator(event_file)
    event_acc.Reload()

    # Get the values of the tag at the specified step
    values = event_acc.Scalars(tag)
    value_at_step = next((v for v in values if v.step == step), None)

    if value_at_step is not None:
        print(f"Value of {tag} at for video {idx + 1} step {step}: {value_at_step.value}")
    else:
        print(f"No values found for tag {tag} at step {step}")"""

"""    # Get scalar data from TensorBoard
    train_loss = event_acc.Scalars('avg/generator_loss/no_smoke')
    #valid_loss = event_acc.Scalars('valid_loss')

    # Extract x and y values from scalar data
    train_steps, train_loss_values = zip(*[(s.step, s.value) for s in train_loss])
    #valid_steps, valid_loss_values = zip(*[(s.step, s.value) for s in valid_loss])

    #sns.set_theme()
    sns.set()

    # Plot the data using matplotlib
    plt.plot(train_steps, train_loss_values, label=f'Fold {idx + 1}')
    #plt.plot(valid_steps, valid_loss_values, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')"""
#plt.savefig('gen_ns_loss.pdf', format='pdf')