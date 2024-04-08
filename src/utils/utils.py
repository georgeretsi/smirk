import os
import torch
import numpy as np

def load_templates():
    templates_path = "assets/expression_templates_famos"
    classes_to_load = ["lips_back", "rolling_lips", "mouth_side", "kissing", "high_smile", "mouth_up",
                       "mouth_middle", "mouth_down", "blow_cheeks", "cheeks_in", "jaw", "lips_up"]
    templates = {}
    for subject in os.listdir(templates_path):
        if os.path.isdir(os.path.join(templates_path, subject)):
            for template in os.listdir(os.path.join(templates_path, subject)):
                if template.endswith(".mp4"):
                    continue
                if template not in classes_to_load:
                    continue
                exps = []
                for npy_file in os.listdir(os.path.join(templates_path, subject, template)):
                    params = np.load(os.path.join(templates_path, subject, template, npy_file), allow_pickle=True)
                    exp = params.item()['expression'].squeeze()
                    exps.append(exp)
                templates[subject+template] = np.array(exps)
    print('Number of expression templates loaded: ', len(templates.keys()))
    
    return templates



def tensor_to_image(image_tensor):
    """Converts a tensor to a numpy image."""
    image = image_tensor.permute(1,2,0).cpu().numpy()*255.0
    image = np.clip(image, 0, 255)
    image = image.astype(np.uint8)
    return image

def image_to_tensor(image):
    """Converts a numpy image to a tensor."""
    image = torch.from_numpy(image).permute(2,0,1).float()/255.0
    return image


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_module(module, module_name=None):
    
    for param in module.parameters():
        param.requires_grad_(False)

    module.eval()


def unfreeze_module(module, module_name=None):
    
    for param in module.parameters():
        param.requires_grad_(True)

    module.train()

import cv2
from torchvision.utils import make_grid


def batch_draw_keypoints(images, landmarks, color=(255, 255, 255), radius=1):
    if isinstance(landmarks, torch.Tensor):
        landmarks = landmarks.cpu().numpy()
        landmarks = landmarks.copy()*112 + 112

    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy().transpose(0, 2, 3, 1)
        images = (images * 255).astype('uint8')
        images = np.ascontiguousarray(images[..., ::-1])

    plotted_images = []
    for image, landmark in zip(images, landmarks):
        for point in landmark:
            image = cv2.circle(image, (int(point[0]), int(point[1])), radius, color, -1)
        plotted_images.append(image)

    return plotted_images

def make_grid_from_opencv_images(images, nrow=12):
    """ Create a grid of images from the list of cv2 images in images"""
    images = np.array(images)
    images = images[..., ::-1]
    images = np.array(images)
    images = torch.from_numpy(images).permute(0, 3, 1, 2).float()/255.
    grid = make_grid(images, nrow=nrow)
    return grid