import os
import pickle
import datasets.data_utils as data_utils
from datasets.base_dataset import BaseDataset
import numpy as np

class MEADSidesDataset(BaseDataset):
    def __init__(self, data_list, config, test=False):
        super().__init__(data_list, config, test)
        self.name = 'MEAD_SIDES'

    def __len__(self):
        return len(self.data_list)

    def __getitem_aux__(self, index):
        sample = self.data_list[index]

        landmarks_filename = sample[1]
        video_path = sample[0]

        if not os.path.exists(landmarks_filename):
            raise Exception('Video %s has no landmarks'%(sample))

        landmarks = np.load(landmarks_filename)

        data_dict = self.sample_frames(video_path, None, landmarks)

        data_dict['subject'] = ""
        data_dict['filename'] = ""

        return data_dict
    


def get_datasets_MEAD_sides(config=None):

    # this is the split used in the paper, randomly selected
    train_subjects = ['M003', 'M007', 'M009', 'M011', 'M012', 'M019', 'M024', 'M025', 'M026', 'M027', 'M029', 'M030', 'M031', 'M032', 'M033', 'M034', 'M035', 'M037', 'M039', 'M040', 'M041', 'W009', 'W011', 'W014', 'W015', 'W016', 'W019', 'W021', 'W023', 'W024', 'W025', 'W026', 'W035', 'W036', 'W037', 'W038', 'W040']
    val_subjects = ['M013', 'M023', 'M042', 'W018', 'W028']
    test_subjects = ['M005', 'M022', 'M028', 'W029', 'W033']

    # assert each subject is in exactly one split
    assert len(set(train_subjects).intersection(val_subjects)) == 0
    assert len(set(train_subjects).intersection(test_subjects)) == 0
    assert len(set(val_subjects).intersection(test_subjects)) == 0

    if not os.path.exists("assets/MEAD_lists.pkl"):
        print('Creating train, validation, and test lists for MEAD Sides... (This only happens once)')

        train_list = []
        val_list = []
        test_list = []

        for view in ['videos_left_30', 'videos_left_60', 'videos_right_30', 'videos_right_60']:

            # Assuming you're currently in the directory where the files are located
            files = [f for f in os.listdir(os.path.join(config.dataset.MEAD_sides_path,view))]

            files = [f for f in files if f.endswith('.mp4') and "test" not in f]

            for file in files:
                if file.split('_')[0] in train_subjects:
                    folder_path = os.path.join(config.dataset.MEAD_sides_path,view,file.split(".")[0]+".mp4")
                    landmarks_path = os.path.join(config.dataset.MEAD_sides_path,view,file.split(".")[0]+".npy")
                    if not os.path.exists(landmarks_path):
                        continue
                    train_list.append([folder_path, landmarks_path, file.split('_')[0]])

            for file in files:
                if file.split('_')[0] in val_subjects:
                    folder_path = os.path.join(config.dataset.MEAD_sides_path,view,file.split(".")[0]+".mp4")
                    landmarks_path = os.path.join(config.dataset.MEAD_sides_path,view,file.split(".")[0]+".npy")
                    if not os.path.exists(landmarks_path):
                        continue
                    val_list.append([folder_path, landmarks_path, file.split('_')[0]])

            for file in files:
                if file.split('_')[0] in test_subjects:
                    folder_path = os.path.join(config.dataset.MEAD_sides_path,view,file.split(".")[0]+".mp4")
                    landmarks_path = os.path.join(config.dataset.MEAD_sides_path,view,file.split(".")[0]+".npy")
                    if not os.path.exists(landmarks_path):
                        continue
                    test_list.append([folder_path, landmarks_path, file.split('_')[0]])
            pickle.dump([train_list,val_list,test_list], open(f"assets/MEAD_lists.pkl", "wb"))
    else:
        train_list, val_list, test_list = pickle.load(open("assets/MEAD_lists.pkl", "rb"))

    # print("MEAD Sides Train: ", len(train_list))
    # print("MEAD Sides Val: ", len(val_list))
    # print("MEAD Sides Test: ", len(test_list))

    return MEADSidesDataset(train_list, config), MEADSidesDataset(val_list, config, test=True), MEADSidesDataset(test_list, config, test=True)





