import os
import numpy as np
import cv2
import sys
from datasets.base_dataset import BaseDataset
import sys

class CelebADataset(BaseDataset):
    def __init__(self, data_list, config, test=False):
        super().__init__(data_list, config, test)
        self.keys = list(data_list.keys())
        self.name = 'CelebA'

    def __len__(self):
        return len(self.keys)

    def __getitem_aux__(self, index):
        key = self.keys[index]

        fan_landmarks_path = os.path.join(self.config.dataset.CelebA_fan_landmarks_path)
        mediapipe_landmarks_path = os.path.join(self.config.dataset.CelebA_mediapipe_landmarks_path)
        folder_path = os.path.join(self.config.dataset.CelebA_path)

        files_list = self.data_list[key]

        fan_landmarks_list = []
        mediapipe_landmarks_list = []
        for file in files_list:
            fan_landmarks_files = os.path.join(fan_landmarks_path, file.replace('.jpg','.npy').replace(".png",".npy"))
            fan_landmarks_list.append(fan_landmarks_files)

            mediapipe_landmarks_files = os.path.join(mediapipe_landmarks_path, file.replace('.jpg','.npy').replace(".png",".npy"))
            mediapipe_landmarks_list.append(mediapipe_landmarks_files)

        files_list = [os.path.join(folder_path, file) for file in files_list]

        data_dict = self.sample_frames(files_list, fan_landmarks_list, mediapipe_landmarks_list)

        data_dict['subject'] = ""
        data_dict['filename'] = ""

        return data_dict



def get_datasets_CelebA(config=None):
    file = "datasets/identity_CelebA.txt"
    with open(file) as f:
        lines = f.readlines()

    subjects = [x.split()[1].strip() for x in lines]
    files = [x.split()[0] for x in lines]

    train_dict = {}
    num_files = 0
    for subject, file in zip(subjects, files):
        if subject not in train_dict:
            train_dict[subject] = []

        if not os.path.exists(os.path.join(config.dataset.CelebA_mediapipe_landmarks_path, file.replace('.jpg','.npy').replace(".png",".npy"))):
            continue

        train_dict[subject].append(file)
        num_files += 1

    print("Number of subjects CeleBA: ", len(train_dict.keys()))
    print("Number of files CeleBA: ", num_files)

    return CelebADataset(train_dict, config)

