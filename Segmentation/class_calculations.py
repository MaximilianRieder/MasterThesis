import torch
from tqdm import tqdm

#######################
### @Tobias Rückert ###
########################################################################################################################################################################
###  Get the weights whith which each class is weighted in the cross entropy loss function. Therefore, loop one time over the training dataset.
########################################################################################################################################################################
def get_class_weights(train_loader, num_classes, device, five_crop):
    print('#####################################')
    print('### Calculate weights for each class.')
    print('#####################################')

    # The count of each class is summed up over all targets, and the inverse class weighting is applied in the following cross entropy loss function.
    value_counts = {}

    for c_idx in range(int(num_classes + 1)):
        value_counts[c_idx] = torch.as_tensor(0.0).to(device)

    for i, data in enumerate(tqdm(train_loader), 0):

        mask = data[1]
        if(five_crop == 'YES'):
            mask = mask.view(mask.shape[0] * mask.shape[1], mask.shape[2], mask.shape[3])
        
        mask = mask.long().to(device)

        unique_counts = torch.unique(mask, sorted=True, return_inverse=False, return_counts=True)
        values = unique_counts[0]
        counts = unique_counts[1]

        for index, unique_v in enumerate(values):
            value_counts[int(unique_v)] = value_counts[int(unique_v)] + counts[index]

    relative_class_frequencies = []
    total_values = torch.tensor(0.0)
    for c in value_counts:
        total_values += value_counts[c].item()

    for c in value_counts:
        if(value_counts[c].item() == 0.0):
            relative_class_frequencies.append(0.0)
        else:
            relative_class_frequencies.append(1.0 - (value_counts[c].item() / total_values))

    relative_class_frequencies = torch.Tensor(relative_class_frequencies).to(device)
    
    return relative_class_frequencies




########################################################################################################################################################################
###  Returns the indices for classes that are not available in the train and validation dataset. Returns one list of indices for each.
########################################################################################################################################################################
def get_unavailable_class_indices(train_loader, val_loader, num_classes, device, five_crop):
    print('##########################################################################')
    print('### Returns the indices for unavailable train and validation classes.')
    print('##########################################################################')

    # Initialize lists with all classes.
    not_available_classes_train = {}
    not_available_classes_val = {}

    ### Create a dict with the class numbers as keys for train and validation, and remove the key of a class from the corresponding dict in case a value in a target mask is found for this class number.
    for c_idx in range(int(num_classes + 1)):
        not_available_classes_train[c_idx] = torch.as_tensor(1).to(device)
        not_available_classes_val[c_idx] = torch.as_tensor(1).to(device)

    ### For train data.
    for i, data in enumerate(tqdm(train_loader), 0):

        mask = data[1]

        if(len(mask.shape) == 1):
            continue

        if(five_crop == 'YES'):
            mask = mask.view(mask.shape[0] * mask.shape[1], mask.shape[2], mask.shape[3])
        
        mask = mask.long().to(device)

        unique_counts = torch.unique(mask, sorted=True, return_inverse=False, return_counts=True)
        values = unique_counts[0]
        counts = unique_counts[1]

        for index, unique_v in enumerate(values):
            if unique_v == 14:
                print("fail: " + str(values) + str(counts))
            if(int(unique_v) in not_available_classes_train):
                not_available_classes_train.pop(int(unique_v))

    ### For validation data.
    for i, data in enumerate(tqdm(val_loader), 0):

        mask = data[1]

        if(len(mask.shape) == 1):
            continue
        
        mask = mask.long().to(device)

        mask = mask.squeeze(1)

        unique_counts = torch.unique(mask, sorted=True, return_inverse=False, return_counts=True)
        values = unique_counts[0]
        counts = unique_counts[1]

        for index, unique_v in enumerate(values):
            if(int(unique_v) in not_available_classes_val):
                not_available_classes_val.pop(int(unique_v))

        

    ### Collect the remaining keys from the dictionaries, i. e. the indices of the classes which are not present, and return them.
    not_available_classes_train_final = []
    not_available_classes_val_final = []

    for k in not_available_classes_train:
        not_available_classes_train_final.append(k)

    for k in not_available_classes_val:
        not_available_classes_val_final.append(k)

    not_available_classes_train_final = torch.ByteTensor(not_available_classes_train_final).to(device)
    not_available_classes_val_final = torch.ByteTensor(not_available_classes_val_final).to(device)

    print('Not available classes train: ' + str(not_available_classes_train_final))
    print('Not available classes val: ' + str(not_available_classes_val_final))

    return_dict = {
        'train' : not_available_classes_train_final,
        'val' : not_available_classes_val_final
    }
    
    return return_dict
