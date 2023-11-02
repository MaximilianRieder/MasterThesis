from average_meter import AverageMeter
import config_seg_gan as config_seg
import sys

def calculate_iou_dice(global_conf_mat):
### Calculate IoU for each class from the global confusion matrix. From: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8569372. 
    ### global_conf_mat has shape 19 x 19. Define dicts that store the iou and dice values for the single classes.
    class_ious = {}
    class_dices = {}
    ### Sum the true positive values, i. e. the values on the diagonal of the conf matrix, up over all classes.
    tp_sum = 0
    ### Get the sum of all entries.
    entries_sum = global_conf_mat.sum()
    mean_iou_meter = AverageMeter()
    mean_dice_meter = AverageMeter()
    ### Loop over classes. From 0 to 18.
    for c in range(global_conf_mat.shape[0]):
        ### Get true positives for current class.
        tp = global_conf_mat[c, c]
        tp_sum += tp
        ### Get false positives for current class including tp, i. e. the sum of the column entries for this class.
        fp = global_conf_mat[ : , c].sum()
        ### Get false negatives for current class including tp, i. e. the sum of the row entries for this class.
        fn = global_conf_mat[c, : ].sum()

        ### Calculate IoU for this class.
        iou_denom = float(fp + fn - tp)
        ### Avoid division by zero by setting the denominator to a small epsilon value. 
        if(iou_denom == 0.00):
            iou_denom = float(sys.float_info.epsilon)
        iou = float(tp) / iou_denom

        ### Calculate Dice for this class.
        dice_denom = float(fp + fn)
        ### Avoid division by zero by setting the denominator to a small epsilon value. 
        if(dice_denom == 0.00):
            dice_denom = float(sys.float_info.epsilon)
        dice = float(2 * tp) / dice_denom 

        ### Calculate global accuracy
        global_acc = float(tp_sum) / float(entries_sum)

        class_ious[c] = iou
        class_dices[c] = dice

    return global_acc, class_ious, class_dices, mean_iou_meter, mean_dice_meter


def calculate_mean_metrics(global_conf_mat, global_acc, class_ious, class_dices, mean_iou_meter, mean_dice_meter, unavailable_class_indices, scheduler, loss_meter, perform_metrics):
    metrics_dict = {}

    if(loss_meter is not None):
        metrics_dict['Loss'] = loss_meter.avg

    if(perform_metrics == True):

        metrics_dict['Confusion_Matrix'] = global_conf_mat

        for key in class_ious:
            metrics_dict['IoU/' + str(key)] = class_ious[key]
            ### Calculate mean IoU value, excluding the background and the classes that are not present in the video sequence at all.
            if(key not in unavailable_class_indices
                and key != 0 and str(key) != '0'):
                # TODO batch size as n?
                mean_iou_meter.update(class_ious[key], n=config_seg.BATCH_SIZE)

        metrics_dict['IoU/mIoU'] = mean_iou_meter.avg

        for key in class_dices:
            metrics_dict['Dice/' + str(key)] = class_dices[key]
            ### Calculate mean Dice value, excluding the background and the classes that are not present in the video sequence at all.
            if(key not in unavailable_class_indices
                and key != 0 and str(key) != '0'):
                mean_dice_meter.update(class_dices[key], n=config_seg.BATCH_SIZE)

        metrics_dict['Dice/mDice'] = mean_dice_meter.avg

        metrics_dict['Global_Acc'] = global_acc

        # At the end of the epoch, print all calculated metrics. 
        for metric_k in metrics_dict:
            if(metric_k == 'IoU/mIoU' or metric_k == 'Dice/mDice' or metric_k == 'Global_Acc' or metric_k == 'Loss'):
                print(metric_k, metrics_dict[metric_k])

    if(scheduler is not None):
        curr_lr = scheduler.get_last_lr()
        metrics_dict['Learning_Rate'] = curr_lr[0]


    return metrics_dict