import os
import pickle
import datasets.data_utils as data_utils

import albumentations as A
from sklearn.model_selection import train_test_split
from datasets.base_dataset import BaseDataset
import numpy as np

class LRS3Dataset(BaseDataset):
    def __init__(self, data_list, config, test=False):
        super().__init__(data_list, config, test)
        self.name = 'LRS3'

    def __len__(self):
        return len(self.data_list)

    def __getitem_aux__(self, index):
        sample = self.data_list[index]

        landmarks_filename = sample[1]
        mediapipe_landmarks_filename = sample[2]
        video_path = sample[0]

        if not os.path.exists(landmarks_filename):
            raise Exception('Video %s has no landmarks'%(sample))

        with open(landmarks_filename, "rb") as pkl_file:
            landmarks = pickle.load(pkl_file)
            preprocessed_landmarks = data_utils.landmarks_interpolate(landmarks)
            if preprocessed_landmarks is None:
                raise Exception('Video %s has no landmarks'%(sample))
        
        mediapipe_landmarks = np.load(mediapipe_landmarks_filename)

        out_dict = self.sample_frames(video_path, preprocessed_landmarks, mediapipe_landmarks)
        
        out_dict['subject'] = video_path.split('/')[-2]
        out_dict['filename'] = video_path.split('/')[-1].split('.')[0]
        return out_dict



def get_datasets_LRS3(config):
    if not os.path.exists('assets/LRS3_lists.pkl'):
        print('Creating train, validation, and test lists for LRS3... (This only happens once)')

        from .data_utils import create_LRS3_lists
        create_LRS3_lists(config.dataset.LRS3_path, config.dataset.LRS3_landmarks_path)


    lists = pickle.load(open("assets/LRS3_lists.pkl", "rb"))
    train_list = lists[0]
    val_list = lists[1]
    test_list = lists[2]
    return LRS3Dataset(train_list, config=config), LRS3Dataset(val_list, config=config, test=True), LRS3Dataset(test_list,
                                                                                               config=config,
                                                                                               test=True)




def get_LRS3_test(config):
    test_folder_list = list(os.listdir(f"{config.dataset.LRS3_path}/test"))


    def gather_LRS3_split(folder_list, split="trainval"):
        list_ = []
        for folder in folder_list:
            for file in os.listdir(os.path.join(f"{config.dataset.LRS3_path}/{split}", folder)):
                if file.endswith(".txt"):
                    file_without_extension = file.split(".")[0]
                    file_inner_path = f"{split}/{folder}/{file_without_extension}"

                    landmarks_filename = os.path.join(config.dataset.LRS3_landmarks_path, file_inner_path+".pkl")
                    subject = folder
                    mediapipe_landmarks_filepath = os.path.join(config.dataset.LRS3_path, file_inner_path+".npy")

                    list_.append([os.path.join(config.dataset.LRS3_path, file_inner_path + ".mp4"), os.path.join(config.dataset.LRS3_landmarks_path, file_inner_path+".pkl"), 
                                    mediapipe_landmarks_filepath,
                                    subject])
        return list_

    test_list = gather_LRS3_split(test_folder_list, split="test")


    return LRS3Dataset(test_list, config=config, test=True, sample_full_video_for_testing=True)