import os
import sys
import pickle
import datasets.data_utils as data_utils
from datasets.base_dataset import BaseDataset
import numpy as np
import cv2

class MEADDataset(BaseDataset):
    def __init__(self, data_list, config, test=False):
        super().__init__(data_list, config, test)
        self.name = 'MEAD'

    def __len__(self):
        return len(self.data_list)

    def __getitem_aux__(self, index):
        sample = self.data_list[index]

        landmarks_filename = sample[1]
        video_path = sample[0]
        mediapipe_landmarks_path = sample[2]

        with open(landmarks_filename, "rb") as pkl_file:
            landmarks = pickle.load(pkl_file)
            preprocessed_landmarks = data_utils.landmarks_interpolate(landmarks)
            if preprocessed_landmarks is None:
                raise Exception('Video %s has no landmarks'%(sample))
            
        if not os.path.exists(mediapipe_landmarks_path):
            print('Mediapipe landmarks not found for %s'%(sample))
            return None

        mediapipe_landmarks = np.load(mediapipe_landmarks_path)

        video = cv2.VideoCapture(video_path)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # select randomly one file from this subject
        if num_frames == 0:
            print('Video %s has no frames'%(sample))
            return None


        # pick random frame
        frame_idx = np.random.randint(0, num_frames)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, image = video.read()
        if not ret:
            raise Exception('Video %s has no frames'%(sample))
        
        landmarks_fan = preprocessed_landmarks[frame_idx]
        landmarks_mediapipe = mediapipe_landmarks[frame_idx]

        data_dict = self.prepare_data(image=image, landmarks_fan=landmarks_fan, landmarks_mediapipe=landmarks_mediapipe)

        return data_dict
    


def get_datasets_MEAD(config=None):
    # Assuming you're currently in the directory where the files are located
    files = [f for f in os.listdir(config.dataset.MEAD_fan_landmarks_path)]

    # this is the split used in the paper, randomly selected
    train_subjects = ['M003', 'M007', 'M009', 'M011', 'M012', 'M019', 'M024', 'M025', 'M026', 'M027', 'M029', 'M030', 'M031', 'M032', 'M033', 'M034', 'M035', 'M037', 'M039', 'M040', 'M041', 'W009', 'W011', 'W014', 'W015', 'W016', 'W019', 'W021', 'W023', 'W024', 'W025', 'W026', 'W035', 'W036', 'W037', 'W038', 'W040']
    val_subjects = ['M013', 'M023', 'M042', 'W018', 'W028']
    test_subjects = ['M005', 'M022', 'M028', 'W029', 'W033']

    # assert each subject is in exactly one split
    assert len(set(train_subjects).intersection(val_subjects)) == 0
    assert len(set(train_subjects).intersection(test_subjects)) == 0
    assert len(set(val_subjects).intersection(test_subjects)) == 0

    train_list = []
    for file in files:
        if file.split('_')[0] in train_subjects:
            landmarks_path = os.path.join(config.dataset.MEAD_fan_landmarks_path, file.split(".")[0] + ".pkl")
            folder_path = os.path.join(config.dataset.MEAD_path,file.split(".")[0]+".mp4")
            mediapipe_landmarks_path = os.path.join(config.dataset.MEAD_mediapipe_landmarks_path, file.split(".")[0] + ".npy")
            train_list.append([folder_path, landmarks_path, mediapipe_landmarks_path, file.split('_')[0]])

    val_list = []
    for file in files:
        if file.split('_')[0] in val_subjects:
            landmarks_path = os.path.join(config.dataset.MEAD_fan_landmarks_path, file.split(".")[0] + ".pkl")
            folder_path = os.path.join(config.dataset.MEAD_path,file.split(".")[0]+".mp4")
            mediapipe_landmarks_path = os.path.join(config.dataset.MEAD_mediapipe_landmarks_path, file.split(".")[0] + ".npy")
            val_list.append([folder_path, landmarks_path, mediapipe_landmarks_path, file.split('_')[0]])

    test_list = []
    for file in files:
        if file.split('_')[0] in test_subjects:
            landmarks_path = os.path.join(config.dataset.MEAD_fan_landmarks_path, file.split(".")[0] + ".pkl")
            folder_path = os.path.join(config.dataset.MEAD_path,file.split(".")[0]+".mp4")
            mediapipe_landmarks_path = os.path.join(config.dataset.MEAD_mediapipe_landmarks_path, file.split(".")[0] + ".npy")
            test_list.append([folder_path, landmarks_path, mediapipe_landmarks_path, file.split('_')[0]])


    return MEADDataset(train_list, config), MEADDataset(val_list, config, test=True), MEADDataset(test_list, config, test=True) #, train_list, val_list, test_list

