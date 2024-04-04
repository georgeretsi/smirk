import os
import torch
import numpy as np


def preprocess_batch(batch, K=1, device='cuda'):
    # this function preprocesses the batch before input to the model
    # it is used to handle the K-tuple sampling
    
    kmask = batch['K_useful']

    for key in batch.keys():
        if key == 'K_useful' or key == 'subject' or key == 'filename':
            continue
        if key == 'dataset_name':
            # list of strings
            if sum(kmask) > 0:
                list1 = [t for t in np.asarray(batch[key])[kmask.numpy()] for _ in range(K)]
            else:
                list1 = []
            if sum(~kmask) > 0:
                list2 = [t for t in np.asarray(batch[key])[~kmask.numpy()]]
            else:
                list2 = []
            batch[key] = list1 + list2
            continue
            
        # per batch flag
        if key == 'flag_landmarks_fan':
            batch[key] = torch.cat([
                batch[key][kmask].view(-1, 1).repeat(1, K).view(-1),
                batch[key][~kmask].view(-1)
            ], dim=0)
        else:
            tsize = batch[key].shape[2:]
            batch[key] = torch.cat([
                batch[key][kmask].view(-1, *tsize),
                batch[key][~kmask][:, 0].view(-1, *tsize)   
            ], dim=0)
        
        batch[key] = batch[key].to(device)  

    # number of K-tuples in the batch
    batch['NK'] = torch.sum(kmask).item() * K
    
    return batch
    


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

