from pandas_ods_reader import read_ods
import os
import sys
import shutil
from tqdm import tqdm

# Path to videos
frames_root_path = '/data/home/rim36739/images/Frames/Smoke_Annotations/'

# Name of each label file
label_files = ['1_labels_v2.ods','2_labels_v2.ods', '3_labels_v2.ods', '4_labels_v2.ods', '5_labels_v2.ods', '6_labels_v2.ods']

complete = 0
complete_hs = 0
complete_ns = 0
complete_ms = 0
for label_file in label_files:

    # Absolute path of file
    path = os.path.join(frames_root_path, label_file)
    df = read_ods(path)

    ### Load a file that does not contain a header row.
    ### If no columns are provided, they will be numbered.
    df = read_ods(path, 1, headers=False)

    ns_count = 0
    ms_count = 0
    hs_count = 0
    not_used_count = 0
    prev_frame = 0
    slightly_smoked = "s"
    heavily_smoked = "h"
    not_smoked = "n"
    prev_class = None

    for entry in range(1, len(df['column.0'])):
        ### Get some attributes from the ODS table.
        frame = int(df['column.0'][entry])
        not_smoked_column = df['column.1'][entry]
        slightly_smoked_column = df['column.2'][entry]
        heavily_smoked_column = df['column.3'][entry]
        smeared = df['column.4'][entry]
        do_not_use = df['column.6'][entry]
        
        if(prev_class is None):
            not_used_count += (frame - prev_frame)
        elif(prev_class is not_smoked):
            ns_count += (frame - prev_frame)
        elif(prev_class is slightly_smoked):
            ms_count += (frame - prev_frame)
        elif(prev_class is heavily_smoked):
            hs_count += (frame - prev_frame)
        else:
            print("error")
            
        ### count occurences
        if(do_not_use is not None):
            prev_class = None
        elif(not_smoked_column is not None):
            prev_class = not_smoked
        elif(slightly_smoked_column is not None):
            prev_class = slightly_smoked
        elif(heavily_smoked_column is not None):
            prev_class = heavily_smoked
        else:
            print("error")

        prev_frame = frame
    complete_hs += hs_count
    complete_ms += ms_count
    complete_ns += ns_count
    complete += (hs_count + ms_count + ns_count)
    
    print(f'{label_file}: \n heavy smoked count: {hs_count} \n slightly smoked count: {ms_count} \n not smoked count: {ns_count} \n not used count: {not_used_count} \n')
    
print(f"hs:{complete_hs} ms:{complete_ms} ns:{complete_ns} insg:{complete}")
"""
    1_labels_v2.ods: 
    heavy smoked count: 13072 
    slightly smoked count: 5445 
    not smoked count: 16753 
    not used count: 22119 

    2_labels_v2.ods: 
    heavy smoked count: 7516 
    slightly smoked count: 9750 
    not smoked count: 46586 
    not used count: 11313 

    3_labels_v2.ods: 
    heavy smoked count: 13752 
    slightly smoked count: 4920 
    not smoked count: 15508 
    not used count: 16755 

    4_labels_v2.ods: 
    heavy smoked count: 14211 
    slightly smoked count: 9698 
    not smoked count: 40512 
    not used count: 10720 

    5_labels_v2.ods: 
    heavy smoked count: 6323 
    slightly smoked count: 6335 
    not smoked count: 23730 
    not used count: 7726 

    6_labels_v2.ods: 
    heavy smoked count: 6473 
    slightly smoked count: 3145 
    not smoked count: 17762 
    not used count: 6864 """