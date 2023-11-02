from pandas_ods_reader import read_ods
import os
import sys
import shutil
from tqdm import tqdm

### Path to the annotations.
path_list = ['1_labels_v2.ods', '2_labels_v2.ods', '3_labels_v2.ods', '4_labels_v2.ods', '5_labels_v2.ods', '6_labels_v2.ods']
### Path to the original frames, from inside the container. Has to be modified if someone else is executing the script. 
frames_base_path = '/workspace/data/Frames/'
### Path and name of the original Videos.
video_path_list = ['Videos_25fps/0001_216', 'Videos_25fps/0002_145', 'Videos_25fps/0003_148', 'Videos_25fps/0004_177', 'Videos_25fps/0005_201', 'Videos_25fps/0007_217']

for vid in range(1):

    print('Start sorting frames of video: ' + str(vid))

    path = path_list[vid]
    video_path = video_path_list[vid]

    not_smoked_path = os.path.join(video_path, 'not_smoked')
    slightly_smoked_path = os.path.join(video_path, 'slightly_smoked')
    heavily_smoked_path = os.path.join(video_path, 'heavily_smoked')

    ### By default the first sheet is imported.
    df = read_ods(path)

    ### Load a file that does not contain a header row.
    ### If no columns are provided, they will be numbered.
    df = read_ods(path, 1, headers=False)

    ### Create directories if they are not already there.
    if('Videos_25fps' in path):
        if(not os.path.exists('Videos_25fps')):
            os.makedirs('Videos_25fps')
    if('Videos_50fps' in path):
        if(not os.path.exists('Videos_50fps')):
            os.makedirs('Videos_50fps')

    for curr_path in [video_path, not_smoked_path, slightly_smoked_path, heavily_smoked_path]:
        if(not os.path.exists(curr_path)):
            os.makedirs(curr_path)


    ### Loop through the lines of the ODS table. 
    prev_frame = 0
    prev_status = ''
    for entry in tqdm(range(1, len(df['column.0']))):
        ### Get some attributes from the ODS table.
        frame = int(df['column.0'][entry])
        not_smoked = df['column.1'][entry]
        slightly_smoked = df['column.2'][entry]
        heavily_smoked = df['column.3'][entry]
        smeared = df['column.4'][entry]
        do_not_use = df['column.6'][entry]


        if(do_not_use is not None):
            status = 'nu'
        elif(not_smoked is not None):
            status = 'ns'
        elif(slightly_smoked is not None):
            status = 'ss'
        elif(heavily_smoked is not None):
            status = 'hs'
        else:
            print("error")



        if(prev_status == 'nu'):
            curr_smoke_save_path = ''

            prev_frame = frame
            prev_status = status

            continue

        elif(prev_status == 'ns'):
            curr_smoke_save_path = not_smoked_path
        elif(prev_status == 'ss'):
            curr_smoke_save_path = slightly_smoked_path
        elif(prev_status == 'hs'):
            curr_smoke_save_path = heavily_smoked_path
        

        if(entry == 1):
            prev_frame = frame
            prev_status = status
            continue
        

        frame_range = range(prev_frame, frame, 1)

        ### Loop through all frames of the current interval / range of frames.
        for f in frame_range:
            
            ### Determine the subfolder, i. e. the folder that contains the current frame in the original frames directory. 
            subfolder = 0
            for s in range(0, 1000000, 1000):
                if(f >= s and f < s + 1000):
                    subfolder = s
                    break
            
            ### Create target directory, if it does not exist yet.
            target_folder = os.path.join(curr_smoke_save_path, str(subfolder))
            if(not os.path.exists(target_folder)):
                os.makedirs(target_folder)

            src_path = os.path.join(frames_base_path, video_path, str(subfolder), 'f_' + str(f) + '.png')
            target_path = os.path.join(curr_smoke_save_path, str(subfolder), 'f_' + str(f) + '.png')
            
            ### Copy original frame to the corresponding target directory.
            shutil.copyfile(src_path, target_path)

        prev_frame = frame
        prev_status = status