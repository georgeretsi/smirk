
import os
import torch
import numpy as np
import torch.nn.functional as F
import cv2
from src.renderer.util import vertex_normals, face_vertices 
from src.FLAME.lbs import vertices2landmarks


def load_probabilities_per_FLAME_triangle():
    """
    FLAME_masks_triangles.npy contains for each face area the indices of the triangles that belong to that area.
    Using that, we can assign a probability to each triangle based on the area it belongs to, and then sample for masking.
    """
    flame_masks_triangles = np.load('assets/FLAME_masks/FLAME_masks_triangles.npy', allow_pickle=True).item()

    area_weights = {
        'neck': 0.0,
        'right_eyeball': 0.0,
        'right_ear': 0.0,
        'lips': 0.00,
        'nose': 0.0,
        'left_ear': 0.0,
        'eye_region': 1.0,
        'forehead':1.0, 
        'left_eye_region': 1.0, 
        'right_eye_region': 1.0, 
        'face_clean': 1.0,
        'cleaner_lips': 1.0
    }

    face_probabilities = torch.zeros(9976)

    for area in area_weights.keys():
        face_probabilities[flame_masks_triangles[area]] = area_weights[area]

    return face_probabilities

def mask_generator(batch_size, input_size, mask_prob, block_size=None):
    
    if block_size is not None:
        # create a random binary mask with size block_size x block_size using torch.bernoulli
        mask = torch.bernoulli(mask_prob * torch.ones((batch_size, block_size, block_size))).cuda()
        
        # rescale the mask to size input_size x input_size
        mask = torch.nn.functional.interpolate(mask.unsqueeze(1), size=input_size, mode='nearest')
    else:
        mask = torch.bernoulli(mask_prob * torch.ones((batch_size, input_size, input_size))).unsqueeze(1).cuda()
    
    return mask


def triangle_area(vertices):
    # Using the Shoelace formula to calculate the area of triangles in the xy plane
    # vertices is expected to be of shape (..., 3, 2) where the last dimension holds x and y coordinates.
    x1, y1 = vertices[..., 0, 0], vertices[..., 0, 1]
    x2, y2 = vertices[..., 1, 0], vertices[..., 1, 1]
    x3, y3 = vertices[..., 2, 0], vertices[..., 2, 1]

    # Shoelace formula for the area of a triangle given by coordinates (x1, y1), (x2, y2), (x3, y3)
    area = 0.5 * torch.abs(x1 * y2 + x2 * y3 + x3 * y1 - x2 * y1 - x3 * y2 - x1 * y3)
    return area



def random_barycentric(num=1):
    # Generate two random numbers for each set
    u, v = torch.rand(num), torch.rand(num)
    
    # Adjust the random numbers if they are outside the triangle
    outside_triangle = u + v > 1
    u[outside_triangle], v[outside_triangle] = 1 - u[outside_triangle], 1 - v[outside_triangle]
    
    # Calculate the barycentric coordinates
    alpha = 1 - (u + v)
    beta = u
    gamma = v
    
    # Combine the coordinates into a single tensor
    return torch.stack((alpha, beta, gamma), dim=1)


def masking(img, mask, extra_points, wr=15, rendered_mask=None, extra_noise=True, random_mask=0.01):
    # img: B x C x H x W
    # mask: B x 1 x H x W
    
    B, C, H, W = img.size()
    
    mask = 1-F.max_pool2d(1-mask, 2 * wr + 1, stride=1, padding=wr)
    
    #tmp_img = mask[0].permute(1, 2, 0).cpu().numpy()
    #tmp_img = cv2.cvtColor(255 * tmp_img, cv2.COLOR_RGB2BGR)
    #cv2.imwrite('img_mask.png', tmp_img)
    
    
    if rendered_mask is not None:
        mask = mask * (1 - rendered_mask) 
        
        

        
    # save img in img1.png
    #tmp_img = img[0].permute(1, 2, 0).cpu().numpy()
    #tmp_img = cv2.cvtColor(255 * tmp_img, cv2.COLOR_RGB2BGR)
    #cv2.imwrite('img_init.png', tmp_img)

    masked_img = img * mask
    # add extra points 
    if extra_noise:
        # normal around 1 with std 0.1
        noise_mult = torch.randn(extra_points.shape).to(img.device) * 0.05 + 1
        extra_points = extra_points * noise_mult

    # select 1% pixels as centers to crop out patches
    if random_mask > 0:
        random_mask = torch.bernoulli(torch.ones((B, 1, H, W)) * random_mask).to(img.device)
        # dilate the mask to have 11x11 patches
        random_mask = 1 - F.max_pool2d(random_mask, 11, stride=1, padding=5)

        extra_points = extra_points * random_mask
        
    #extra_points_mask = extra_points > 0
    #extra_points_mask = F.max_pool2d(extra_points_mask.float(), 3, stride=1, padding=1)
    #extra_points = extra_points_mask * img
    #extra_points = F.max_pool2d(extra_points, 5, stride=1, padding=2)
    #tmp_img = extra_points[0].permute(1, 2, 0).cpu().numpy()
    #tmp_img = cv2.cvtColor(255 * tmp_img, cv2.COLOR_RGB2BGR)
    #cv2.imwrite('img_points.png', tmp_img)

    masked_img[extra_points > 0] = extra_points[extra_points > 0]

    masked_img = masked_img.detach()
    
    #tmp_img = masked_img[0].permute(1, 2, 0).cpu().numpy()
    #tmp_img = cv2.cvtColor(255 * tmp_img, cv2.COLOR_RGB2BGR)
    #cv2.imwrite('img_m.png', tmp_img)
    
    #raise ValueError('Check the masked image')

    return masked_img


def mouth_mask(landmarks, image_size=(224, 224)):
    # landmarks: B x 68 x 2
    device = landmarks.device

    landmarks = landmarks.cpu().numpy()

    # normalize landmarks from [-1, 1] to image size
    landmarks = (landmarks + 1) * image_size[1] / 2

    B = landmarks.shape[0]
    # compute th convex hull of the mouth
    mouth_masks = np.zeros((B, image_size[0], image_size[1]))
    for i in range(B):
        mask = np.zeros(image_size)
        cv2.fillConvexPoly(mask, np.int32(landmarks[i, 60:68, :].round()), 1)
        mouth_masks[i] = mask

    return torch.from_numpy(mouth_masks).unsqueeze(1).float().to(device)

def point2ind(npoints, H):
    
    npoints = npoints * (H // 2) + H // 2
    npoints = npoints.long()
    npoints[...,1] = torch.clamp(npoints[..., 1], 0, H-1)
    npoints[...,0] = torch.clamp(npoints[..., 0], 0, H-1)
    
    return npoints

'''
def transfer_pixels(img, flame_out1, flame_out2, inds):
    
    
    B, C, H, W = img.size()
    retained_pixels = torch.zeros_like(img).cuda()
    
    # trans verts of flame_output to image space
    npoints1 = flame_out1[torch.arange(B).unsqueeze(-1), inds]
    npoints2 = flame_out2[torch.arange(B).unsqueeze(-1), inds]
    
    npoints1 = point2ind(npoints1, H)
    npoints2 = point2ind(npoints2, H)

    for bi in range(B):
        retained_pixels[bi, :, npoints2[bi, :, 1], npoints2[bi, :, 0]] = img[bi, :, npoints1[bi, :, 1], npoints1[bi, :, 0]] 
        
    return retained_pixels
'''

def transfer_pixels(img, points1, points2, rbound=None):

    B, C, H, W = img.size()
    retained_pixels = torch.zeros_like(img).to(img.device)

    if rbound is not None:
        for bi in range(B):
            retained_pixels[bi, :, points2[bi, :rbound[bi], 1], points2[bi, :rbound[bi], 0]] = \
            img[bi, :, points1[bi, :rbound[bi], 1], points1[bi, :rbound[bi], 0]]
    else:
        retained_pixels[torch.arange(B).unsqueeze(-1), :, points2[..., 1], points2[..., 0]] = \
        img[torch.arange(B).unsqueeze(-1), :, points1[..., 1], points1[..., 0]]

    return retained_pixels

"""
def mesh_based_mask(self, flame_output, mask_ratio=.2, inds=None):

    mask = torch.zeros((flame_output['trans_verts'].size(0), 1, self.config.image_size, self.config.image_size)).to(self.config.device)
    
    # trans verts of flame_output to image space
    npoints = flame_output['trans_verts'][:, self.renderer.final_mask, :2]
    
    # randomly select some points according to mask_ratio
    if inds is None:
        # random select inds of size img.size(0) x int(mask_ratio * npoints.size(1))
        inds = torch.argsort(torch.rand((npoints.size(0), npoints.size(1))), dim=-1)[..., :int(mask_ratio * npoints.size(1))]

    npoints = npoints[torch.arange(npoints.shape[0]).unsqueeze(-1), inds]
    
    npoints = npoints * (self.config.image_size // 2) + self.config.image_size // 2
    npoints = npoints.long()
    npoints[...,1] = torch.clamp(npoints[..., 1], 0, self.config.image_size-1)
    npoints[...,0] = torch.clamp(npoints[..., 0], 0, self.config.image_size-1)
    

    for bi in range(mask.size(0)):
        mask[bi, :, npoints[bi, :, 1], npoints[bi, :, 0]] = 1  
        
    return mask
"""
"""
def mesh_based_mask_uniform(self, flame_output, mask_ratio=.2, inds=None):

    import pickle
    flame_masks = pickle.load(
        open('assets/FLAME_masks/FLAME_masks.pkl', 'rb'),
        encoding='latin1')

    face_mask = flame_masks['face']

    face_mask_clean = []
    for index in face_mask:
        found = False
        for key in flame_masks.keys():
            if key == 'face':
                continue
            if index in flame_masks[key]:
                found = True
                break
        if not found:
            face_mask_clean.append(index)

    face_mask = np.asarray(face_mask_clean)
    flame_masks['face_clean'] = face_mask

    # flame_masks['inner_lips'] = [1657, 1658, 1693, 1694, 1695, 1696, 1716, 1717, 1735, 1749, 1773, 1774, 1775, 1776, 1794, 1795, 1802, 1803, 1850, 1865, 2774, 2775, 2810, 2811, 2812, 2813, 2833, 2834, 2850, 2864, 2880, 2881, 2882, 2883, 2897, 2898, 2905, 2906, 2939, 2948, 3503, 3506, 3541, 3543]
    flame_masks['even_inner_lips'] = [1657, 1694, 1696, 1716, 1735, 1775, 1776, 1794, 1803, 1850, 2774, 2811, 2813, 2833, 2850, 2882, 2883, 2897, 2906, 2939, 3506, 3543]


    area_ratios = {
        # 'eye_region': 0.0,
        'neck': 0.0,
        # 'left_eyeball': 0.0,
        'right_eyeball': 0.0,
        'right_ear': 0.0,
        'right_eye_region': 0.0,
        'forehead': mask_ratio,
        'lips': 0.0,
        'nose': .1 * mask_ratio,
        # 'scalp': 0.0,
        # 'boundary': 0.0,
        # 'face': 0.0,
        'left_ear': 0.0,
        'left_eye_region': 0.0,
        'face_clean': mask_ratio,
        'even_inner_lips': mask_ratio
    }

    mask = torch.zeros((flame_output['trans_verts'].size(0), 1, self.config.image_size, self.config.image_size)).to(self.config.device)

    all_points = []

    for area in area_ratios.keys():
        area_points = flame_output['trans_verts'][:, flame_masks[area], :2]

        # sample randomly from the area
        inds = torch.argsort(torch.rand((area_points.size(0), area_points.size(1))), dim=-1)[..., :int(area_ratios[area] * area_points.size(1))]

        npoints = area_points[torch.arange(area_points.shape[0]).unsqueeze(-1), inds]
        all_points.append(npoints)


    npoints = torch.cat(all_points, dim=1)
    
    npoints = npoints * (self.config.image_size // 2) + self.config.image_size // 2
    npoints = npoints.long()
    npoints[...,1] = torch.clamp(npoints[..., 1], 0, self.config.image_size-1)
    npoints[...,0] = torch.clamp(npoints[..., 0], 0, self.config.image_size-1)
    

    for bi in range(mask.size(0)):
        mask[bi, :, npoints[bi, :, 1], npoints[bi, :, 0]] = 1  
        
    return mask, torch.cat(all_points, dim=1)
"""

def mesh_based_mask_uniform_faces(flame_trans_verts, flame_faces, face_probabilities, mask_ratio=0.1, coords=None, IMAGE_SIZE=224):
    """
    This function samples points from the FLAME mesh based on the face probabilities and the mask ratio.
    """
    batch_size = flame_trans_verts.size(0)
    DEVICE = flame_trans_verts.device

    # if mask_ratio is single value, then use it as a ratio of the image size
    num_points_to_sample = int(mask_ratio * IMAGE_SIZE * IMAGE_SIZE)

    flame_faces_expanded = flame_faces.expand(batch_size, -1, -1)

    if coords is None:
        # calculate face normals
        transformed_normals = vertex_normals(flame_trans_verts, flame_faces_expanded) 
        transformed_face_normals = face_vertices(transformed_normals, flame_faces_expanded)
        transformed_face_normals = transformed_face_normals[:,:,:,2].mean(dim=-1)
        face_probabilities = face_probabilities.repeat(batch_size,1).to(flame_trans_verts.device)

        # # where the face normals are negative, set probability to 0
        face_probabilities = torch.where(transformed_face_normals < 0.05, face_probabilities, torch.zeros_like(transformed_face_normals).to(DEVICE))
        # face_probabilities = torch.where(transformed_face_normals > 0, torch.ones_like(transformed_face_normals).to(flame_trans_verts.device), face_probabilities)

        # calculate xy area of faces and scale the probabilities by it
        fv = face_vertices(flame_trans_verts, flame_faces_expanded)
        xy_area = triangle_area(fv)

        face_probabilities = face_probabilities * xy_area


        sampled_faces_indices = torch.multinomial(face_probabilities, num_points_to_sample, replacement=True).to(DEVICE)

        barycentric_coords = random_barycentric(num=batch_size*num_points_to_sample).to(DEVICE)
        barycentric_coords = barycentric_coords.view(batch_size, num_points_to_sample, 3)
    else:
        sampled_faces_indices = coords['sampled_faces_indices']
        barycentric_coords = coords['barycentric_coords']

    npoints = vertices2landmarks(flame_trans_verts, flame_faces, sampled_faces_indices, barycentric_coords)

    npoints = .5 * (1 + npoints) * IMAGE_SIZE
    npoints = npoints.long()
    npoints[...,1] = torch.clamp(npoints[..., 1], 0, IMAGE_SIZE-1)
    npoints[...,0] = torch.clamp(npoints[..., 0], 0, IMAGE_SIZE-1)

    #mask = torch.zeros((flame_output['trans_verts'].size(0), 1, self.config.image_size, self.config.image_size)).to(flame_output['trans_verts'].device)

    #mask[torch.arange(batch_size).unsqueeze(-1), :, npoints[..., 1], npoints[..., 0]] = 1        

    return npoints, {'sampled_faces_indices':sampled_faces_indices, 'barycentric_coords':barycentric_coords}
