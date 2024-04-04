import os
import torch
from datasets.base_dataset import BaseDataset


class FFHQDataset(BaseDataset):
    def __init__(self, data_list, config, test=False):
        super().__init__(data_list, config, test)
        self.name = 'FFHQ'

    def __getitem_aux__(self, index):
        data_dict = self.sample_frames([self.data_list[index][0]], [self.data_list[index][1]], [self.data_list[index][2]])

        data_dict['subject'] = ""
        data_dict['filename'] = ""
        
        return data_dict


    
def get_datasets_FFHQ(config):
    if config.K > 1:
        print('Warning: K > 1 not supported for FFHQ dataset. Make sure you have FFHQ percentage set to 0.')

    train_list = []

    for image in os.listdir(config.dataset.FFHQ_path):
        if image.endswith(".png"):
            image_path = os.path.join(config.dataset.FFHQ_path, image)
            fan_landmarks_path = os.path.join(config.dataset.FFHQ_fan_landmarks_path, image.split(".")[0] + ".npy")
            mediapipe_landmarks_path = os.path.join(config.dataset.FFHQ_mediapipe_landmarks_path, image.split(".")[0] + ".npy")

            train_list.append([image_path, fan_landmarks_path, mediapipe_landmarks_path])

    dataset = FFHQDataset(train_list, config, test=False)
    return dataset




