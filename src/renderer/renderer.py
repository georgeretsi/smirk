import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.renderer.mesh import rasterize_meshes
from src.renderer.util import face_vertices, vertex_normals, batch_orth_proj
import pickle

def keep_vertices_and_update_faces(faces, vertices_to_keep):
    """
    Keep specified vertices in the mesh and update the faces.

    Parameters:
    faces (torch.Tensor): Tensor of shape (F, 3) representing faces, with each value being a vertex index.
    vertices_to_keep (list or torch.Tensor): List or tensor of vertex indices to keep.

    Returns:
    Tuple[torch.Tensor, torch.Tensor]: Updated vertices and faces tensors.
    """
    # Convert vertices_to_keep to a tensor if it's a list or numpy array
    if isinstance(vertices_to_keep, list) or isinstance(vertices_to_keep, np.ndarray):
        vertices_to_keep = torch.tensor(vertices_to_keep, dtype=torch.long)

    # Ensure vertices_to_keep is unique and sorted
    vertices_to_keep = torch.unique(vertices_to_keep)

    max_vertex_index = faces.max().long().item() + 1

    # Create a mask for vertices to keep
    mask = torch.zeros(max_vertex_index, dtype=torch.bool)
    mask[vertices_to_keep] = True


    # Create a mapping from old vertex indices to new ones
    new_vertex_indices = torch.full((max_vertex_index,), -1, dtype=torch.long)
    new_vertex_indices[mask] = torch.arange(len(vertices_to_keep))

    # Remove faces that reference removed vertices (where mapping is -1)
    valid_faces_mask = (new_vertex_indices[faces] != -1).all(dim=1)
    filtered_faces = faces[valid_faces_mask]

    # Update face indices to new vertex indices
    updated_faces = new_vertex_indices[filtered_faces]

    return updated_faces

class Renderer(nn.Module):
    def __init__(self, render_full_head=False, obj_filename='assets/FLAME2020/head_template.obj'):
        super(Renderer, self).__init__()
        self.image_size = 224

        verts, faces, aux = load_obj(obj_filename)
        uvcoords = aux.verts_uvs[None, ...]      # (N, V, 2)
        uvfaces = faces.textures_idx[None, ...] # (N, F, 3)
        faces = faces.verts_idx[None,...]


        self.render_full_head = render_full_head

        # shape colors, for rendering shape overlay
        colors = torch.tensor([180, 180, 180])[None, None, :].repeat(1, faces.max()+1, 1).float()/255.

        flame_masks = pickle.load(
            open('assets/FLAME_masks/FLAME_masks.pkl', 'rb'),
            encoding='latin1')
        self.flame_masks = flame_masks

        if not render_full_head:
            self.final_mask = flame_masks['face'].tolist()

            # keep only faces that include vertices in face_mask
            faces = keep_vertices_and_update_faces(faces[0], self.final_mask).unsqueeze(0)

            colors = colors[:, self.final_mask, :]

        self.register_buffer('faces', faces)

        face_colors = face_vertices(colors, faces)
        self.register_buffer('face_colors', face_colors)
        
        self.register_buffer('raw_uvcoords', uvcoords)

        # uv coords
        uvcoords = torch.cat([uvcoords, uvcoords[:,:,0:1]*0.+1.], -1) #[bz, ntv, 3]
        uvcoords = uvcoords*2 - 1; uvcoords[...,1] = -uvcoords[...,1]
        face_uvcoords = face_vertices(uvcoords, uvfaces)
        self.register_buffer('uvcoords', uvcoords)
        self.register_buffer('uvfaces', uvfaces)
        self.register_buffer('face_uvcoords', face_uvcoords)

        ## SH factors for lighting
        pi = np.pi
        constant_factor = torch.tensor([1/np.sqrt(4*pi), ((2*pi)/3)*(np.sqrt(3/(4*pi))), ((2*pi)/3)*(np.sqrt(3/(4*pi))),\
                           ((2*pi)/3)*(np.sqrt(3/(4*pi))), (pi/4)*(3)*(np.sqrt(5/(12*pi))), (pi/4)*(3)*(np.sqrt(5/(12*pi))),\
                           (pi/4)*(3)*(np.sqrt(5/(12*pi))), (pi/4)*(3/2)*(np.sqrt(5/(12*pi))), (pi/4)*(1/2)*(np.sqrt(5/(4*pi)))]).float()
        self.register_buffer('constant_factor', constant_factor)

    def forward(self, vertices, cam_params, **landmarks):
        transformed_vertices = batch_orth_proj(vertices, cam_params)
        transformed_vertices[:, :, 1:] = -transformed_vertices[:, :, 1:]

        transformed_landmarks = {}
        for key in landmarks.keys():
            transformed_landmarks[key] = batch_orth_proj(landmarks[key], cam_params)
            transformed_landmarks[key][:, :, 1:] = - transformed_landmarks[key][:, :, 1:]
            transformed_landmarks[key] = transformed_landmarks[key][...,:2]

        rendered_img = self.render(vertices, transformed_vertices)

        outputs = {
            'rendered_img': rendered_img,
            'transformed_vertices': transformed_vertices
        }
        outputs.update(transformed_landmarks)

        return outputs
        

    def render(self, vertices, transformed_vertices):
        """ Render the mesh with the given vertices. Transformed vertices includes vertices in NDC space.
        Note that due to this custom implementation of the renderer, the NDC space does not follow the PyTorch3D convention of axes.
        """
        batch_size = vertices.shape[0]

        light_positions = torch.tensor(
            [
            [-1,1,1],
            [1,1,1],
            [-1,-1,1],
            [1,-1,1],
            [0,0,1]
            ]
        )[None,:,:].expand(batch_size, -1, -1).float()
        light_intensities = torch.ones_like(light_positions).float()*1.7
        lights = torch.cat((light_positions, light_intensities), 2).to(vertices.device)
        
        if not self.render_full_head:
            transformed_vertices = transformed_vertices[:,self.final_mask,:]
            vertices = vertices[:,self.final_mask,:]

        ## rasterizer near 0 far 100. move mesh so minz larger than 0
        transformed_vertices[:,:,2] = transformed_vertices[:,:,2] + 10
        
        # attributes
        normals = vertex_normals(vertices, self.faces.expand(batch_size, -1, -1)) 
        face_normals = face_vertices(normals, self.faces.expand(batch_size, -1, -1))
        
        colors = self.face_colors.expand(batch_size, -1, -1, -1)

        attributes = torch.cat([colors,
                                face_normals], 
                                -1)
        # rasterize
        rendering = self.rasterize(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)
        
        albedo_images = rendering[:, :3, :, :]

        # shading

        normal_images = rendering[:, 3:6, :, :]

        shading = self.add_directionlight(normal_images.permute(0,2,3,1).reshape([batch_size, -1, 3]), lights)
        shading_images = shading.reshape([batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0,3,1,2).contiguous()        
        shaded_images = albedo_images*shading_images
        
        return shaded_images


    def rasterize(self, vertices, faces, attributes=None, h=None, w=None):
        fixed_vertices = vertices.clone()
        fixed_vertices[...,:2] = -fixed_vertices[...,:2]

        if h is None and w is None:
            image_size = self.image_size
        else:
            image_size = [h, w]
            if h>w:
                fixed_vertices[..., 1] = fixed_vertices[..., 1]*h/w
            else:
                fixed_vertices[..., 0] = fixed_vertices[..., 0]*w/h
            
        meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,
            max_faces_per_bin=None,
            perspective_correct=False,
        )
        vismask = (pix_to_face > -1).float()
        D = attributes.shape[-1]
        attributes = attributes.clone(); attributes = attributes.view(attributes.shape[0]*attributes.shape[1], 3, attributes.shape[-1])
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0  # Replace masked values in output.
        pixel_vals = pixel_vals[:,:,:,0].permute(0,3,1,2)
        pixel_vals = torch.cat([pixel_vals, vismask[:,:,:,0][:,None,:,:]], dim=1)
        return pixel_vals

    def add_SHlight(self, normal_images, sh_coeff):
        '''
            sh_coeff: [bz, 9, 3]
        '''
        N = normal_images
        sh = torch.stack([
                N[:,0]*0.+1., N[:,0], N[:,1], \
                N[:,2], N[:,0]*N[:,1], N[:,0]*N[:,2], 
                N[:,1]*N[:,2], N[:,0]**2 - N[:,1]**2, 3*(N[:,2]**2) - 1
                ], 
                1) # [bz, 9, h, w]
        sh = sh*self.constant_factor[None,:,None,None]
        shading = torch.sum(sh_coeff[:,:,:,None,None]*sh[:,:,None,:,:], 1) # [bz, 9, 3, h, w]  
        return shading

    def add_pointlight(self, vertices, normals, lights):
        '''
            vertices: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nv, 3]
        '''
        light_positions = lights[:,:,:3]; light_intensities = lights[:,:,3:]
        directions_to_lights = F.normalize(light_positions[:,:,None,:] - vertices[:,None,:,:], dim=3)
        # normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        normals_dot_lights = (normals[:,None,:,:]*directions_to_lights).sum(dim=3)
        shading = normals_dot_lights[:,:,:,None]*light_intensities[:,:,None,:]
        return shading.mean(1)
    

    def add_directionlight(self, normals, lights):
        '''
            normals: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nv, 3]
        '''
        light_direction = lights[:,:,:3]; light_intensities = lights[:,:,3:]
        directions_to_lights = F.normalize(light_direction[:,:,None,:].expand(-1,-1,normals.shape[1],-1), dim=3)
        normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        shading = normals_dot_lights[:,:,:,None]*light_intensities[:,:,None,:]
        return shading.mean(1)



    def render_multiface(self, vertices, transformed_vertices, faces):
        
        batch_size = vertices.shape[0]

        # ----- different light positions ! ----- #
        light_positions = torch.tensor(
            [
            [-1,-1,-1],
            [1,-1,-1],
            [-1,+1,-1],
            [1,+1,-1],
            [0,0,-1]
            ]
        )[None,:,:].expand(batch_size, -1, -1).float()

        light_intensities = torch.ones_like(light_positions).float()*1.7
        lights = torch.cat((light_positions, light_intensities), 2).to(vertices.device)
        
        ## rasterizer near 0 far 100. move mesh so minz larger than 0
        transformed_vertices[:,:,2] = transformed_vertices[:,:,2] + 10
        normals = vertex_normals(vertices, faces) 
        face_normals = face_vertices(normals, faces)
        
        colors = torch.tensor([180, 180, 180])[None, None, :].repeat(1, transformed_vertices.shape[1]+1, 1).float()/255.
        colors = colors.cuda()

        face_colors = face_vertices(colors, faces[0].unsqueeze(0))
        
        colors = face_colors.expand(batch_size, -1, -1, -1)

        attributes = torch.cat([colors,
                                face_normals], 
                                -1)
        # rasterize
        rendering = self.rasterize(transformed_vertices, faces, attributes)
        
        albedo_images = rendering[:, :3, :, :]

        # shading

        normal_images = rendering[:, 3:6, :, :]

        shading = self.add_directionlight(normal_images.permute(0,2,3,1).reshape([batch_size, -1, 3]), lights)
        shading_images = shading.reshape([batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0,3,1,2).contiguous()        
        shaded_images = albedo_images*shading_images
        
        return shaded_images
