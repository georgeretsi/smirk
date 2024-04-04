import torch.utils.data
from skimage.transform import estimate_transform, warp
import albumentations as A
import numpy as np
from skimage import transform as trans
import cv2
import random
import os


def create_mask(landmarks, shape):
    # Convert landmarks to the required format for convexHull function
    landmarks = landmarks.astype(np.int32)[...,:2]

    # Get the convex hull
    hull = cv2.convexHull(landmarks)

    # Create an empty mask of the same size as the input shape
    mask = np.ones(shape, dtype=np.uint8)

    # Fill the inside of the convex hull with 0 in the mask
    cv2.fillConvexPoly(mask, hull, 0)

    return mask


mediapipe_indices = [276, 282, 283, 285, 293, 295, 296, 300, 334, 336,  46,  52,  53,
        55,  63,  65,  66,  70, 105, 107, 249, 263, 362, 373, 374, 380,
       381, 382, 384, 385, 386, 387, 388, 390, 398, 466,   7,  33, 133,
       144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246,
       168,   6, 197, 195,   5,   4, 129,  98,  97,   2, 326, 327, 358,
         0,  13,  14,  17,  37,  39,  40,  61,  78,  80,  81,  82,  84,
        87,  88,  91,  95, 146, 178, 181, 185, 191, 267, 269, 270, 291,
       308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409,
       415]

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, config, test=False):
        self.data_list = data_list
        self.config = config
        self.image_size = config.image_size
        self.test = test
        self.K = config.K
        self.sample_full_video_for_testing = config.dataset.sample_full_video_for_testing

        if not self.test:
            self.scale = [config.train.train_scale_min, config.train.train_scale_max] 
        else:
            self.scale = config.train.test_scale
        
        self.transform = A.ReplayCompose([
                # color ones
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
                A.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05, p=0.25),
                A.CLAHE(p=0.255),
                #A.HueSaturationValue(p=0.25),  
                A.RGBShift(p=0.25),
                A.Blur(p=0.1),
                A.GaussNoise(p=0.5),
                # affine ones
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, border_mode=0, p=0.9),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),  additional_targets={'mediapipe_keypoints': 'keypoints'})


        self.arcface_dst = np.array(
            [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
            [41.5493, 92.3655], [70.7299, 92.2041]],
            dtype=np.float32)

        # self.arcface_dst = self.arcface_dst/112.0*224.0


    def estimate_norm(self, lmk, image_size=112,mode='arcface'):
        assert lmk.shape == (5, 2)
        assert image_size%112==0 or image_size%128==0
        if image_size%112==0:
            ratio = float(image_size)/112.0
            diff_x = 0
        else:
            ratio = float(image_size)/128.0
            diff_x = 8.0*ratio
        dst = self.arcface_dst * ratio
        dst[:,0] += diff_x
        tform = trans.SimilarityTransform()
        tform.estimate(lmk, dst)
        M = tform.params[0:2, :]
        return M

    @staticmethod
    def crop_face(frame, landmarks, scale=1.0, image_size=224):
        left = np.min(landmarks[:, 0])
        right = np.max(landmarks[:, 0])
        top = np.min(landmarks[:, 1])
        bottom = np.max(landmarks[:, 1])

        h, w, _ = frame.shape
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])

        size = int(old_size * scale)

        # crop image
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        return tform

    def __len__(self):
        return len(self.data_list)

    def __getitem_aux__(self, index):
        pass


    def __getitem__(self, index):
        landmarks_not_checked = True
        while landmarks_not_checked:
            try:
                data_dict = self.__getitem_aux__(index)
                # check if landmarks are not None
                if data_dict is not None:
                    landmarks = data_dict['landmarks_fan']
                    if landmarks is not None and (landmarks.shape[-2] == 68):
                        landmarks_not_checked = False
                        break
                #else:
                print("Error in loading data. Trying again...")
                index = np.random.randint(0, len(self.data_list))
            except Exception as e:
                # raise e
                print('Error in loading data. Trying again...', e)
                index = np.random.randint(0, len(self.data_list))


        return data_dict


    def sample_frames(self, imagePathsOrVideo, landmarksPathsOrNumpyArrayFan, landmarksPathsOrNumpyArrayMediapipe):
        images_list = []; fan_kpt_list = []; masked_images_list = []; masks_list = [];
        mediapipe_kpt_list = [];
        mica_list = []

        if isinstance(imagePathsOrVideo, str):
            isVideo = True
            video = cv2.VideoCapture(imagePathsOrVideo)
            num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        elif isinstance(imagePathsOrVideo, list):
            isVideo = False
            num_frames = len(imagePathsOrVideo)
        
        if self.test and self.sample_full_video_for_testing:
            K_useful = True
            frame_indices = np.arange(0,num_frames,5)
            # frame_indices = np.array([0,num_frames//2])

        elif self.test:
            # pick the middle frame for testing ? 
            K_useful = True
            frame_indices = np.array([num_frames//2])

        else:
            if self.name == 'FFHQ':
                # repeat the first frame K times
                K_useful = False
                frame_indices = np.zeros(self.K, dtype=np.int32)
            elif self.name == 'LRS3' and self.config.dataset.LRS3_temporal_sampling is True:
                K_useful = True
                start_idx = random.randint(0, num_frames - self.K)
                end_idx = start_idx + self.K
                frame_indices = list(range(start_idx,end_idx))
            else:
                K_useful = True
                frame_indices = np.random.randint(0, num_frames, size=self.K)

        if isinstance(self.scale, list):
            scale = np.random.rand() * (self.scale[1] - self.scale[0]) + self.scale[0]
        else:
            scale = self.scale
            
        
        # set initial transformed for replay usages in LRS3 when temporal sampling
        replay_data = None

        for frame_idx in frame_indices:
            if isVideo:
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = video.read()
            else:
                frame = cv2.imread(imagePathsOrVideo[frame_idx])

            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # ------------ load fan landmarks ------------ #
            if landmarksPathsOrNumpyArrayFan is None:
                flag_landmarks_fan = False
                kpt_fan = np.zeros((68,2))
            else:
                flag_landmarks_fan = True
                if isinstance(landmarksPathsOrNumpyArrayFan, list):
                    if isinstance(landmarksPathsOrNumpyArrayFan[frame_idx], str):
                        kpt_filename = landmarksPathsOrNumpyArrayFan[frame_idx]            
                        kpt_fan = np.load(kpt_filename, allow_pickle=True)
                        if kpt_fan is None or kpt_fan.size == 1:
                            return None
                        
                        kpt_fan = kpt_fan[0]

                    else:
                        kpt_fan = landmarksPathsOrNumpyArrayFan[frame_idx]
                elif isinstance(landmarksPathsOrNumpyArrayFan, np.ndarray):
                    kpt_fan = landmarksPathsOrNumpyArrayFan[frame_idx]
                else:
                    raise Exception('landmarksPathsOrNumpyArrayFan should be a list of paths or a numpy array')
            
            # ------------ load mediapipe landmarks ------------ #
            if isinstance(landmarksPathsOrNumpyArrayMediapipe, list):
                if isinstance(landmarksPathsOrNumpyArrayMediapipe[frame_idx], str):
                    kpt_filename = landmarksPathsOrNumpyArrayMediapipe[frame_idx]            
                    kpt_mediapipe = np.load(kpt_filename, allow_pickle=True)
                else:
                    kpt_mediapipe = landmarksPathsOrNumpyArrayMediapipe[frame_idx]
            elif isinstance(landmarksPathsOrNumpyArrayMediapipe, np.ndarray):
                kpt_mediapipe = landmarksPathsOrNumpyArrayMediapipe[frame_idx]
            else:
                raise Exception('landmarksPathsOrNumpyArrayNediapipe should be a list of paths or a numpy array')

            tform = self.crop_face(frame,kpt_mediapipe,scale,image_size=self.image_size)
            
            kpt_mediapipe = kpt_mediapipe[...,:2]
            

            cropped_image = warp(frame, tform.inverse, output_shape=(self.image_size, self.image_size), preserve_range=True).astype(np.uint8)
            cropped_kpt_fan = np.dot(tform.params, np.hstack([kpt_fan, np.ones([kpt_fan.shape[0],1])]).T).T
            cropped_kpt_fan = cropped_kpt_fan[:,:2]


            cropped_kpt_mediapipe = np.dot(tform.params, np.hstack([kpt_mediapipe, np.ones([kpt_mediapipe.shape[0],1])]).T).T
            cropped_kpt_mediapipe = cropped_kpt_mediapipe[:,:2]

            # find convex hull
            hull_mask = create_mask(cropped_kpt_mediapipe, (self.image_size, self.image_size))

            cropped_kpt_mediapipe = cropped_kpt_mediapipe[mediapipe_indices,:2]


            # augment
            if not self.test:
                try:
                    if self.name == 'LRS3' and self.config.dataset.LRS3_temporal_sampling is True:
                        # appply the same transform as the previous frame
                        if replay_data is not None:
                            transformed = A.ReplayCompose.replay(replay_data, image=cropped_image, mask= 1 - hull_mask, keypoints=cropped_kpt_fan, mediapipe_keypoints=cropped_kpt_mediapipe)
                        else:
                            transformed = self.transform(image=cropped_image, mask= 1 - hull_mask, keypoints=cropped_kpt_fan, mediapipe_keypoints=cropped_kpt_mediapipe)
                            replay_data = transformed['replay']
                    else:
                        transformed = self.transform(image=cropped_image, mask= 1 - hull_mask, keypoints=cropped_kpt_fan, mediapipe_keypoints=cropped_kpt_mediapipe)
    
                    cropped_image = (transformed['image']/255.0).astype(np.float32)
                    cropped_kpt_fan = np.array(transformed['keypoints']).astype(np.float32)
                    cropped_kpt_mediapipe = np.array(transformed['mediapipe_keypoints']).astype(np.float32)
                    hull_mask = 1 - transformed['mask']
                except ValueError: # this from albumentation
                    print("Error in albumentations...")
                    return None
                # hull_mask_rev = transformed['revmask']
            else: 
                cropped_image = (cropped_image/255.0).astype(np.float32)
                cropped_kpt_fan = cropped_kpt_fan.astype(np.float32)
                cropped_kpt_mediapipe = cropped_kpt_mediapipe.astype(np.float32)
                


            cropped_kpt_fan[:,:2] = cropped_kpt_fan[:,:2]/self.image_size * 2  - 1
            cropped_kpt_mediapipe[:,:2] = cropped_kpt_mediapipe[:,:2]/self.image_size * 2  - 1

            masked_cropped_image = cropped_image * hull_mask[...,None]

            images_list.append(cropped_image.transpose(2,0,1))
            fan_kpt_list.append(cropped_kpt_fan)
            mediapipe_kpt_list.append(cropped_kpt_mediapipe)
            masked_images_list.append(masked_cropped_image.transpose(2,0,1))
            masks_list.append(hull_mask[...,None])


            # ----------- mica images ---------------- #
            kpt2 = kpt_fan[[36,45,32,48,54]].copy()
            kpt2[0] = (kpt_fan[36] + kpt_fan[39])/2
            kpt2[1] = (kpt_fan[42] + kpt_fan[45])/2

            # tform = self.crop_face(image,kpt,scale)
            tform = self.estimate_norm(kpt2, 112)

            frame = frame/255.
            # dst_image = warp(image, tform.inverse, output_shape=(self.crop_size, self.crop_size))
            mica_image = cv2.warpAffine(frame, tform, (112, 112), borderValue=0.0)
            mica_image = mica_image.transpose(2,0,1)
            mica_list.append(mica_image)



        images_array = torch.from_numpy(np.array(images_list)).type(dtype = torch.float32) #K,224,224,3
        masked_images_array = torch.from_numpy(np.array(masked_images_list)).type(dtype = torch.float32) #K,224,224,3

        try:
            kpt_array_fan = torch.from_numpy(np.array(fan_kpt_list)).type(dtype = torch.float32) #K,224,224,3
            kpt_array_mediapipe = torch.from_numpy(np.array(mediapipe_kpt_list)).type(dtype = torch.float32) #K,224,224,3
        except:
            return None
        masks_array = torch.from_numpy(np.array(masks_list)).type(dtype = torch.float32).permute(0, 3, 1, 2) #K,1, 224,224
        images_array_mica = torch.from_numpy(np.array(mica_list)).type(dtype = torch.float32) #K,3,224,224

        data_dict = {
            'img': images_array,
            'landmarks_fan': kpt_array_fan[...,:2],
            'landmarks_mp': kpt_array_mediapipe[...,:2],
            'masked_img': masked_images_array,
            'mask': masks_array,
            'img_mica': images_array_mica,
            'flag_landmarks_fan': flag_landmarks_fan,
            'K_useful': K_useful,
            'dataset_name': self.name
        }


        return data_dict
