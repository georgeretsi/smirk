import torch
import torch.nn.functional as F
from torch import nn
import timm


def create_backbone(backbone_name, pretrained=True):
    backbone = timm.create_model(backbone_name, 
                        pretrained=pretrained,
                        features_only=True)
    feature_dim = backbone.feature_info[-1]['num_chs']
    return backbone, feature_dim

class PoseEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
              
        self.encoder, feature_dim = create_backbone('tf_mobilenetv3_small_minimal_100')
        
        self.pose_cam_layers = nn.Sequential(
            nn.Linear(feature_dim, 6)
        )

        self.init_weights()

    def init_weights(self):
        self.pose_cam_layers[-1].weight.data *= 0.001
        self.pose_cam_layers[-1].bias.data *= 0.001

        self.pose_cam_layers[-1].weight.data[3] = 0
        self.pose_cam_layers[-1].bias.data[3] = 7


    def forward(self, img):
        features = self.encoder(img)[-1]
            
        features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)

        outputs = {}

        pose_cam = self.pose_cam_layers(features).reshape(img.size(0), -1)
        outputs['pose_params'] = pose_cam[...,:3]
        outputs['cam'] = pose_cam[...,3:]

        return outputs


class ShapeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder, feature_dim = create_backbone('tf_mobilenetv3_large_minimal_100')

        self.shape_layers = nn.Sequential(
            nn.Linear(feature_dim, 300)
        )

        self.init_weights()


    def init_weights(self):
        self.shape_layers[-1].weight.data *= 0
        self.shape_layers[-1].bias.data *= 0


    def forward(self, img):
        features = self.encoder(img)[-1]
            
        features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)

        parameters = self.shape_layers(features).reshape(img.size(0), -1)

        return {'shape_params': parameters}


class ExpressionEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder, feature_dim = create_backbone('tf_mobilenetv3_large_minimal_100')
        
        self.expression_layers = nn.Sequential( 
            nn.Linear(feature_dim, 55) # num expressions + jaw + eyelid
        )

        self.init_weights()


    def init_weights(self):
        self.expression_layers[-1].weight.data *= 0.1
        self.expression_layers[-1].bias.data *= 0.1


    def forward(self, img):
        features = self.encoder(img)[-1]
            
        features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)


        parameters = self.expression_layers(features).reshape(img.size(0), -1)

        outputs = {}

        outputs['expression_params'] = parameters[...,:50]
        outputs['eyelid_params'] = torch.clamp(parameters[...,50:52], 0, 1)
        outputs['jaw_params'] = torch.cat([F.relu(parameters[...,52].unsqueeze(-1)), 
                                           torch.clamp(parameters[...,53:55], -.2, .2)], dim=-1)

        return outputs


class SmirkEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.pose_encoder = PoseEncoder()

        self.shape_encoder = ShapeEncoder()

        self.expression_encoder = ExpressionEncoder() 

    def forward(self, img):
        pose_outputs = self.pose_encoder(img)
        shape_outputs = self.shape_encoder(img)
        expression_outputs = self.expression_encoder(img)

        outputs = {}
        outputs.update(pose_outputs)
        outputs.update(shape_outputs)
        outputs.update(expression_outputs)

        return outputs
