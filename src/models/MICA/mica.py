import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.MICA.arcface import Arcface


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')


class MappingNetwork(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, map_output_dim, hidden=2):
        super().__init__()

        if hidden > 5:
            self.skips = [int(hidden / 2)]
        else:
            self.skips = []

        self.network = nn.ModuleList(
            [nn.Linear(z_dim, map_hidden_dim)] +
            [nn.Linear(map_hidden_dim, map_hidden_dim) if i not in self.skips else
             nn.Linear(map_hidden_dim + z_dim, map_hidden_dim) for i in range(hidden)]
        )

        self.output = nn.Linear(map_hidden_dim, map_output_dim)
        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.output.weight *= 0.25

    def forward(self, z):
        h = z
        for i, l in enumerate(self.network):
            h = self.network[i](h)
            h = F.leaky_relu(h, negative_slope=0.2)
            if i in self.skips:
                h = torch.cat([z, h], 1)

        output = self.output(h)
        return output
    



class MICA(nn.Module):
    def __init__(self):
        super(MICA, self).__init__()

        self.arcface = Arcface()

        self.regressor = MappingNetwork(512, 300, 300, hidden=3)

        checkpoint = torch.load("assets/mica.tar")

        self.arcface.load_state_dict(checkpoint['arcface'], strict=True)

        mapping_network_keys = {}
        for key in checkpoint['flameModel'].keys():
            if 'network' in key or 'output' in key:
                mapping_network_keys[key.replace("regressor.","")] = checkpoint['flameModel'][key]

        self.regressor.load_state_dict(mapping_network_keys, strict=True)


    def forward(self, images):

        transformed_images_for_mica = images.sub(0.5).div(0.5)
        transformed_images_for_mica = transformed_images_for_mica[:, [2, 1, 0], :, :]
        # transformed_images_for_mica = F.interpolate(transformed_images_for_mica, size=(112, 112))

        arcface_features = F.normalize(self.arcface(transformed_images_for_mica))
        
        shape_params = self.regressor(arcface_features)

        return {'shape_params': shape_params}

    def calculate_mica_shape_loss(self, shape_params, img):
        """ Runs MICA on the input image and calculates the L2 loss between the input shape_params and the shape_params calculated by MICA. """
        B, D = shape_params.size()

        with torch.no_grad():
            mica_output = self.forward(img.reshape(-1, 3, 112, 112))
            mica_shape = mica_output['shape_params'].detach()

        if D > mica_shape.size(-1):
            mica_shape = torch.cat([mica_shape, torch.zeros(B, D - mica_shape.size(-1)).to(self.config.device)], dim=-1)

        loss = F.mse_loss(shape_params, mica_shape)
        return loss