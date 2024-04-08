
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
        'lips': 0.5,
        'nose': 0.5,
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
    
    # dilate face mask, drawn from convex hull of face landmarks
    mask = 1-F.max_pool2d(1-mask, 2 * wr + 1, stride=1, padding=wr)
    
    # optionally remove the rendered mask 
    if rendered_mask is not None:
        mask = mask * (1 - rendered_mask) 

    masked_img = img * mask
    # add noise to extra in-face points
    if extra_noise:
        # normal around 1 with std 0.1
        noise_mult = torch.randn(extra_points.shape).to(img.device) * 0.05 + 1
        extra_points = extra_points * noise_mult

    # select random_mask percentage of pixels as centers to crop out 11x11 patches 
    if random_mask > 0:
        random_mask = torch.bernoulli(torch.ones((B, 1, H, W)) * random_mask).to(img.device)
        # dilate the mask to have 11x11 patches
        random_mask = 1 - F.max_pool2d(random_mask, 11, stride=1, padding=5)

        extra_points = extra_points * random_mask

    masked_img[extra_points > 0] = extra_points[extra_points > 0]

    masked_img = masked_img.detach()
    return masked_img



def point2ind(npoints, H):
    
    npoints = npoints * (H // 2) + H // 2
    npoints = npoints.long()
    npoints[...,1] = torch.clamp(npoints[..., 1], 0, H-1)
    npoints[...,0] = torch.clamp(npoints[..., 0], 0, H-1)
    
    return npoints


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
