import os
import sys
import pickle
import datasets.data_utils as data_utils
from datasets.base_dataset import BaseDataset
import numpy as np

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

        if not os.path.exists(landmarks_filename):
            raise Exception('Video %s has no landmarks'%(sample))

        with open(landmarks_filename, "rb") as pkl_file:
            landmarks = pickle.load(pkl_file)
            preprocessed_landmarks = data_utils.landmarks_interpolate(landmarks)
            if preprocessed_landmarks is None:
                raise Exception('Video %s has no landmarks'%(sample))

        mediapipe_landmarks = np.load(mediapipe_landmarks_path)

        data_dict = self.sample_frames(video_path, preprocessed_landmarks, mediapipe_landmarks)

        data_dict['subject'] = ""
        data_dict['filename'] = ""


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

    # print("Train Files:", len(train_list))
    # print("Validation Files:", len(val_list))
    # print("Test Files:", len(test_list))

    return MEADDataset(train_list, config), MEADDataset(val_list, config, test=True), MEADDataset(test_list, config, test=True) #, train_list, val_list, test_list


def get_datasets_MEAD_intrasubject(config=None):

    files = [f for f in os.listdir(config.dataset.MEAD_fan_landmarks_path)]


    # this is the split used in the paper, randomly selected
    train_subjects = ['M003', 'M007', 'M009', 'M011', 'M012', 'M019', 'M024', 'M025', 'M026', 'M027', 'M029', 'M030', 'M031', 'M032', 'M033', 'M034', 'M035', 'M037', 'M039', 'M040', 'M041', 'W009', 'W011', 'W014', 'W015', 'W016', 'W019', 'W021', 'W023', 'W024', 'W025', 'W026', 'W035', 'W036', 'W037', 'W038', 'W040']
    val_subjects = ['M013', 'M023', 'M042', 'W018', 'W028']
    test_subjects = ['M005', 'M022', 'M028', 'W029', 'W033']

    all_subjects = train_subjects + val_subjects + test_subjects

    all_list = []
    for file in files:
        if file.split('_')[0] in all_subjects:
            landmarks_path = os.path.join(config.dataset.MEAD_fan_landmarks_path, file.split(".")[0] + ".pkl")
            folder_path = os.path.join(config.dataset.MEAD_path,file.split(".")[0]+".mp4")
            mediapipe_landmarks_path = os.path.join(config.dataset.MEAD_mediapipe_landmarks_path, file.split(".")[0] + ".npy")
            all_list.append([folder_path, landmarks_path, mediapipe_landmarks_path, file.split('_')[0]])

    from sklearn.model_selection import train_test_split

    # print([f[3] for f in all_list])
    train_list, test_list = train_test_split(all_list, test_size=0.2, random_state=42, stratify=[f[3] for f in all_list])

    return MEADDataset(train_list, config), MEADDataset(test_list, config, test=True)




if __name__ == '__main__':
    from omegaconf import OmegaConf
    import sys
    import cv2
    config = OmegaConf.load(sys.argv[1])

    OmegaConf.set_struct(config, True)

    # Remove the configuration file name from sys.argv
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    # merge config with cli args
    config.merge_with_cli()


    # this is the split used in the paper, randomly selected
    train_subjects = ['M003', 'M007', 'M009', 'M011', 'M012', 'M019', 'M024', 'M025', 'M026', 'M027', 'M029', 'M030', 'M031', 'M032', 'M033', 'M034', 'M035', 'M037', 'M039', 'M040', 'M041', 'W009', 'W011', 'W014', 'W015', 'W016', 'W019', 'W021', 'W023', 'W024', 'W025', 'W026', 'W035', 'W036', 'W037', 'W038', 'W040']
    val_subjects = ['M013', 'M023', 'M042', 'W018', 'W028']
    test_subjects = ['M005', 'M022', 'M028', 'W029', 'W033']

    # assert each subject is in exactly one split
    assert len(set(train_subjects).intersection(val_subjects)) == 0
    assert len(set(train_subjects).intersection(test_subjects)) == 0
    assert len(set(val_subjects).intersection(test_subjects)) == 0

    files = [f for f in os.listdir(config.dataset.MEAD_fan_landmarks_path)]

    for split in ["train", "val", "test"]:
        num = 0 
        if split == "train":
            s = train_subjects
        elif split == "val":
            s = val_subjects
        elif split == "test":
            s = test_subjects 

        for subject in s:
            subject_images = []
            for file in files:
                if file.split('_')[0] == subject:
                    # folder_path = os.path.join(config.dataset.MEAD_path,file.split(".")[0]+".mp4")
                    # vid = cv2.VideoCapture(folder_path)

                    # # get midframe
                    # frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
                    # vid.set(cv2.CAP_PROP_POS_FRAMES, frame_count//2)
                    # ret, image = vid.read()

                    # image = cv2.resize(image, (48, 48))

                    # subject_images.append(image)
                
                    num += 1

            # # create grid
            # grid = []
            # i = 0
            # while i < len(subject_images):
            #     ims = subject_images[i:i+30]
            #     if len(ims) < 30:
            #         ims += [np.zeros_like(ims[0]) for _ in range(30-len(ims))]
            #     grid.append(np.concatenate(ims, axis=1))
            #     i += 30

            # grid = np.concatenate(grid, axis=0)

            # os.makedirs("datasets/mead_test", exist_ok=True)

            # cv2.imwrite("datasets/mead_test/MEAD_%s_%s.png"%(split,subject), grid)
        print(split, num)



    train_dataset, val_dataset, test_dataset, train_list, val_list, test_list = get_datasets_MEAD(config)
    print(len(train_dataset), len(val_dataset), len(test_dataset))
    
    # train_subjects = []
    # val_subjects = []
    # test_subjects = []
    # for sample in train_list:
    #     subject = sample[0].split("/")[-1].split("_")[0]
    #     if subject not in train_subjects:
    #         train_subjects.append(subject)

    # for sample in val_list:
    #     subject = sample[0].split("/")[-1].split("_")[0]
    #     if subject not in val_subjects:
    #         val_subjects.append(subject)

    # for sample in test_list:
    #     subject = sample[0].split("/")[-1].split("_")[0]
    #     if subject not in test_subjects:
    #         test_subjects.append(subject)



    # print("Train Subjects:", sorted(train_subjects))
    # print("Validation Subjects:", sorted(val_subjects))
    # print("Test Subjects:", sorted(test_subjects))


