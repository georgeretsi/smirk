import os
from datasets.base_dataset import BaseDataset
import numpy as np
import cv2

class CelebADataset(BaseDataset):
    def __init__(self, data_list, config, test=False):
        super().__init__(data_list, config, test)
        self.keys = list(data_list.keys())
        self.name = 'CelebA'

    def __len__(self):
        return len(self.keys)

    def __getitem_aux__(self, index):
        key = self.keys[index]

        files_list = self.data_list[key]

        # select randomly one file from this subject
        if len(files_list) == 0:
            print('No files found for %s'%(key))
            return None
        
        k = np.random.randint(0, len(files_list))

        image_filepath = os.path.join(self.config.dataset.CelebA_path, files_list[k])
        landmarks_fan_filepath = os.path.join(self.config.dataset.CelebA_fan_landmarks_path, files_list[k].replace('.jpg','.npy'))
        landmarks_mediapipe_filepath = os.path.join(self.config.dataset.CelebA_mediapipe_landmarks_path, files_list[k].replace('.jpg','.npy'))

        if not os.path.exists(landmarks_mediapipe_filepath):
            print('Mediapipe landmarks not found for %s'%(files_list[k]))
            return None
    
        if not os.path.exists(landmarks_fan_filepath):
            print('Fan landmarks not found for %s'%(files_list[k]))
            return None
        
        image = cv2.imread(image_filepath)
        landmarks_fan = np.load(landmarks_fan_filepath, allow_pickle=True)
        if landmarks_fan is None or landmarks_fan.size == 1:
            return None
        
        landmarks_fan = landmarks_fan[0]

        landmarks_mediapipe = np.load(landmarks_mediapipe_filepath, allow_pickle=True)


        data_dict = self.prepare_data(image=image, landmarks_fan=landmarks_fan, landmarks_mediapipe=landmarks_mediapipe)
        return data_dict



def get_datasets_CelebA(config=None):
    file = "datasets/identity_CelebA.txt" # provided by CelebA dataset
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

    return CelebADataset(train_dict, config)

