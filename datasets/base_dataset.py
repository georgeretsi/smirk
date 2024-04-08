import torch.utils.data
from skimage.transform import estimate_transform, warp
import albumentations as A
import numpy as np
from skimage import transform as trans
import cv2


def create_mask(landmarks, shape):
    landmarks = landmarks.astype(np.int32)[...,:2]
    hull = cv2.convexHull(landmarks)
    mask = np.ones(shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 0)

    return mask

# these are the indices of the mediapipe landmarks that correspond to the mediapipe landmark barycentric coordinates provided by FLAME2020
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

        if not self.test:
            self.scale = [config.train.train_scale_min, config.train.train_scale_max] 
        else:
            self.scale = config.train.test_scale
        
        self.transform = A.Compose([
                # color ones
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
                A.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05, p=0.25),
                A.CLAHE(p=0.255),
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

    def prepare_data(self, image, landmarks_fan, landmarks_mediapipe):
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


        if landmarks_fan is None:
            flag_landmarks_fan = False
            landmarks_fan = np.zeros((68,2))
        else:
            flag_landmarks_fan = True
            
        # crop the image using the landmarks
        if isinstance(self.scale, list):
            # select randomly a zoom scale during training for cropping
            scale = np.random.rand() * (self.scale[1] - self.scale[0]) + self.scale[0]
        else:
            scale = self.scale

        tform = self.crop_face(image,landmarks_mediapipe,scale,image_size=self.image_size)
        
        landmarks_mediapipe = landmarks_mediapipe[...,:2]
        
        cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size), preserve_range=True).astype(np.uint8)
        cropped_landmarks_fan = np.dot(tform.params, np.hstack([landmarks_fan, np.ones([landmarks_fan.shape[0],1])]).T).T
        cropped_landmarks_fan = cropped_landmarks_fan[:,:2]


        cropped_landmarks_mediapipe = np.dot(tform.params, np.hstack([landmarks_mediapipe, np.ones([landmarks_mediapipe.shape[0],1])]).T).T
        cropped_landmarks_mediapipe = cropped_landmarks_mediapipe[:,:2]

        # find convex hull for masking the face 
        hull_mask = create_mask(cropped_landmarks_mediapipe, (self.image_size, self.image_size))

        cropped_landmarks_mediapipe = cropped_landmarks_mediapipe[mediapipe_indices,:2]


        # augment
        if not self.test:
            transformed = self.transform(image=cropped_image, mask= 1 - hull_mask, keypoints=cropped_landmarks_fan, mediapipe_keypoints=cropped_landmarks_mediapipe)

            cropped_image = (transformed['image']/255.0).astype(np.float32)
            cropped_landmarks_fan = np.array(transformed['keypoints']).astype(np.float32)
            cropped_landmarks_mediapipe = np.array(transformed['mediapipe_keypoints']).astype(np.float32)
            hull_mask = 1 - transformed['mask']
        else: 
            cropped_image = (cropped_image/255.0).astype(np.float32)
            cropped_landmarks_fan = cropped_landmarks_fan.astype(np.float32)
            cropped_landmarks_mediapipe = cropped_landmarks_mediapipe.astype(np.float32)
            

        cropped_landmarks_fan[:,:2] = cropped_landmarks_fan[:,:2]/self.image_size * 2  - 1
        cropped_landmarks_mediapipe[:,:2] = cropped_landmarks_mediapipe[:,:2]/self.image_size * 2  - 1
        masked_cropped_image = cropped_image * hull_mask[...,None]
        

        cropped_image = cropped_image.transpose(2,0,1)
        masked_cropped_image = masked_cropped_image.transpose(2,0,1)
        hull_mask = hull_mask[...,None]
        hull_mask = hull_mask.transpose(2,0,1)


        # ----------- mica images ---------------- #
        landmarks_arcface_crop = landmarks_fan[[36,45,32,48,54]].copy()
        landmarks_arcface_crop[0] = (landmarks_fan[36] + landmarks_fan[39])/2
        landmarks_arcface_crop[1] = (landmarks_fan[42] + landmarks_fan[45])/2

        tform = self.estimate_norm(landmarks_arcface_crop, 112)

        image = image/255.
        mica_image = cv2.warpAffine(image, tform, (112, 112), borderValue=0.0)
        mica_image = mica_image.transpose(2,0,1)



        image = torch.from_numpy(cropped_image).type(dtype = torch.float32) 
        masked_image = torch.from_numpy(masked_cropped_image).type(dtype = torch.float32)
        landmarks_fan = torch.from_numpy(cropped_landmarks_fan).type(dtype = torch.float32) 
        landmarks_mediapipe = torch.from_numpy(cropped_landmarks_mediapipe).type(dtype = torch.float32)
        hull_mask = torch.from_numpy(hull_mask).type(dtype = torch.float32)
        mica_image = torch.from_numpy(mica_image).type(dtype = torch.float32) 


        data_dict = {
            'img': image,
            'landmarks_fan': landmarks_fan[...,:2],
            'flag_landmarks_fan': flag_landmarks_fan, # if landmarks are not available
            'landmarks_mp': landmarks_mediapipe[...,:2],
            'mask': hull_mask,
            'img_mica': mica_image
        }


        return data_dict
