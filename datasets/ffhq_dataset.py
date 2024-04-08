import os
import numpy as np
from datasets.base_dataset import BaseDataset
import cv2
import numpy as np


class FFHQDataset(BaseDataset):
    def __init__(self, data_list, config, test=False):
        super().__init__(data_list, config, test)
        self.name = 'FFHQ'
        

    def __getitem_aux__(self, index):
        image = cv2.imread(self.data_list[index][0])

        # check if paths exist
        if not os.path.exists(self.data_list[index][2]):
            print('Mediapipe landmarks not found for %s'%(self.data_list[index]))
            return None
        
        if not os.path.exists(self.data_list[index][1]):
            print('Fan landmarks not found for %s'%(self.data_list[index]))
            return None


        landmarks_fan = np.load(self.data_list[index][1], allow_pickle=True)
        if landmarks_fan is None or landmarks_fan.size == 1:
            return None
        
        landmarks_fan = landmarks_fan[0] # first found face
        
        landmarks_mediapipe = np.load(self.data_list[index][2], allow_pickle=True)

        data_dict = self.prepare_data(image=image, landmarks_fan=landmarks_fan, landmarks_mediapipe=landmarks_mediapipe)
        
        return data_dict


    
def get_datasets_FFHQ(config):
    train_list = []

    for image in os.listdir(config.dataset.FFHQ_path):
        if image.endswith(".png"):
            image_path = os.path.join(config.dataset.FFHQ_path, image)
            fan_landmarks_path = os.path.join(config.dataset.FFHQ_fan_landmarks_path, image.split(".")[0] + ".npy")
            mediapipe_landmarks_path = os.path.join(config.dataset.FFHQ_mediapipe_landmarks_path, image.split(".")[0] + ".npy")

            train_list.append([image_path, fan_landmarks_path, mediapipe_landmarks_path])

    dataset = FFHQDataset(train_list, config, test=False)
    return dataset




