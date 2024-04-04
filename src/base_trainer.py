import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torchvision.utils import make_grid
import cv2
import os
import random
import copy

class BaseTrainer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


    def logging(self, batch_idx, losses, phase):

        # ---------------- logging ---------------- #
        if self.config.train.log_losses_every > 0 and batch_idx % self.config.train.log_losses_every == 0:
            # print losses in one line
            loss_str = ''
            for k, v in losses.items():
                loss_str += f'{k}: {v:.6f} '
                if self.logger is not None:
                    self.logger.log({f'{phase}/{k}': v})
            print(loss_str)

    def configure_optimizers(self, n_steps):
        
        self.n_steps = n_steps

        # start decaying at max_epochs // 2
        # at the end of training, the lr will be 0.1 * lr
        # lambda_func = lambda epoch: 1.0 - max(0, .1 + epoch - max_epochs//2) / float(max_epochs//2)

        encoder_scale = .25

        # check if self.encoder_optimizer exists
        if hasattr(self, 'encoder_optimizer'):
            for g in self.encoder_optimizer.param_groups:
                g['lr'] = encoder_scale * self.config.train.lr
        else:
            params = []
            if self.config.train.optimize_expression:
                params += list(self.smirk_encoder.expression_encoder.parameters()) 
            if self.config.train.optimize_shape:
                params += list(self.smirk_encoder.shape_encoder.parameters())
            if self.config.train.optimize_pose:
                params += list(self.smirk_encoder.pose_encoder.parameters())

            self.encoder_optimizer = torch.optim.Adam(params, lr= encoder_scale * self.config.train.lr)
                
        # cosine schedulers for both optimizers - per iterations
        self.encoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.encoder_optimizer, T_max=n_steps,
                                                                                  eta_min=0.01 * encoder_scale * self.config.train.lr)

        if self.config.arch.enable_fuse_generator:
            if hasattr(self, 'fuse_generator_optimizer'):
                for g in self.smirk_generator_optimizer.param_groups:
                    g['lr'] = self.config.train.lr
            else:
                self.smirk_generator_optimizer = torch.optim.Adam(self.smirk_generator.parameters(), lr= self.config.train.lr, betas=(0.5, 0.999))

            
            self.smirk_generator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.smirk_generator_optimizer, T_max=n_steps,
                                                                                    eta_min=0.01 * self.config.train.lr)

        
    def load_random_template(self, num_expressions=50):
        random_key = random.choice(list(self.templates.keys()))
        templates = self.templates[random_key]
        random_index = random.randint(0, templates.shape[0]-1)

        return templates[random_index][:num_expressions]
        
    def setup_logger(self, wandb_run):
        self.logger = wandb_run

    def setup_losses(self):
        from src.losses.VGGPerceptualLoss import VGGPerceptualLoss
        self.vgg_loss = VGGPerceptualLoss()
        self.vgg_loss.eval()
        for param in self.vgg_loss.parameters():
            param.requires_grad_(False)
        
            
        if self.config.train.loss_weights['emotion_loss'] > 0:
            from src.losses.ExpressionLoss import ExpressionLoss
            self.emotion_loss = ExpressionLoss()
            # freeze the emotion model
            self.emotion_loss.eval()
            for param in self.emotion_loss.parameters():
                param.requires_grad_(False)

        if self.config.train.loss_weights['mica_loss'] > 0:
            from src.models.MICA.mica import MICA

            self.mica = MICA()
            self.mica.eval()

            for param in self.mica.parameters():
                param.requires_grad_(False)

            
    def scheduler_step(self):
        self.encoder_scheduler.step()
        if self.config.arch.enable_fuse_generator:
            self.smirk_generator_scheduler.step()

    def train(self):
        self.smirk_encoder.train()
        if self.config.arch.enable_fuse_generator:
            self.smirk_generator.train()
    
    def eval(self):
        self.smirk_encoder.eval()
        if self.config.arch.enable_fuse_generator:
            self.smirk_generator.eval()

    def optimizers_zero_grad(self):
        self.encoder_optimizer.zero_grad()
        if self.config.arch.enable_fuse_generator:
            self.smirk_generator_optimizer.zero_grad()

    def optimizers_step(self, step_encoder=True, step_fuse_generator=True):
        if step_encoder:
            self.encoder_optimizer.step()
        if step_fuse_generator and self.config.arch.enable_fuse_generator:
            self.smirk_generator_optimizer.step()


    def save_visualizations(self, outputs, save_path, show_landmarks=True):
        nrow = 1
        
        if 'img' in outputs and 'rendered_img' in outputs and 'masked_1st_path' in outputs:
            outputs['overlap_image'] = outputs['img'] * 0.7 + outputs['rendered_img'] * 0.3
            outputs['overlap_image_pixels'] = outputs['img'] * 0.7 +  0.3 * outputs['masked_1st_path']
        
        original_img_with_landmarks = outputs['img']
        original_grid = make_grid(original_img_with_landmarks, nrow=nrow)

        image_keys = ['img_mica', 'rendered_img_base', 'rendered_img', 
                      'overlap_image', 'overlap_image_pixels',
                    'rendered_img_mica_zero', 'rendered_img_zero', 
                      'masked_1st_path', 'reconstructed_img', 'loss_img', 
                      '2nd_path']
        
        nrows = [1 if '2nd_path' not in key else 4 * self.config.train.Ke for key in image_keys]
        
        grid = torch.cat([original_grid] + [make_grid(outputs[key].detach().cpu(), nrow=nr) for key, nr in zip(image_keys, nrows) if key in outputs.keys()], dim=2)
            
        grid = grid.permute(1,2,0).cpu().numpy()*255.0
        grid = np.clip(grid, 0, 255)
        grid = grid.astype(np.uint8)
        grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)

        cv2.imwrite(save_path, grid)


    def create_visualizations(self, batch, outputs):
        zero_pose_cam = torch.tensor([7,0,0]).unsqueeze(0).repeat(batch['img'].shape[0], 1).float().to(self.config.device)

        # batch keys are already in device, so no need to move them
        # outputs are in cpu, so we need to move them to device if we want to use them in the renderer

        visualizations = {}

        visualizations['img'] = batch['img']
        visualizations['rendered_img'] = outputs['rendered_img']
        

        # 2. Base model
        base_output = self.base_encoder(batch['img'])
        flame_output_base = self.flame.forward(base_output)
        rendered_img_base = self.renderer.forward(flame_output_base['vertices'], base_output['cam'])['rendered_img']
        visualizations['rendered_img_base'] = rendered_img_base
    
        flame_output_zero = self.flame.forward(outputs['encoder_output'], zero_expression=True, zero_pose=True)
        rendered_img_zero = self.renderer.forward(flame_output_zero['vertices'].to(self.config.device), zero_pose_cam)['rendered_img']
        visualizations['rendered_img_zero'] = rendered_img_zero
    
    
        if self.config.arch.enable_fuse_generator:
            visualizations['reconstructed_img'] = outputs['reconstructed_img']
            visualizations['masked_1st_path'] = outputs['masked_1st_path']
            visualizations['loss_img'] = outputs['loss_img']

        for key in visualizations.keys():
            visualizations[key] = visualizations[key].detach().cpu()

        # 3. MICA
        if self.config.train.loss_weights['mica_loss'] > 0:
            mica_output_shape = self.mica(batch['img_mica'])
            mica_output = copy.deepcopy(base_output) # just to get the keys and structure
            mica_output['shape_params'] = mica_output_shape['shape_params']
            flame_output_mica = self.flame.forward(mica_output, zero_expression=True, zero_pose=True)
            rendered_img_mica_zero = self.renderer.forward(flame_output_mica['vertices'], zero_pose_cam)['rendered_img']
            visualizations['rendered_img_mica_zero'] = rendered_img_mica_zero

            visualizations['img_mica'] = batch['img_mica'].reshape(-1, 3, 112, 112)
            visualizations['img_mica'] = F.interpolate(visualizations['img_mica'], self.config.image_size).detach().cpu()


        if self.config.train.loss_weights['cycle_loss'] > 0:
            if '2nd_path' in outputs:
                visualizations['2nd_path'] = outputs['2nd_path']

        return visualizations

    def save_model(self, state_dict, save_path):
        # remove everything that is not smirk_encoder or smirk_generator
        new_state_dict = {}
        for key in list(state_dict.keys()):
            if key.startswith('smirk_encoder') or key.startswith('smirk_generator'):
                new_state_dict[key] = state_dict[key]

        torch.save(new_state_dict, save_path)


    def create_base_encoder(self):
        self.base_encoder = copy.deepcopy(self.smirk_encoder)
        self.base_encoder.eval()



    def load_model(self, resume, load_fuse_generator=True, load_encoder=True, device='cuda'):
        loaded_state_dict = torch.load(resume, map_location=device)

        print(f'Loading checkpoint from {resume}, load_encoder={load_encoder}, load_fuse_generator={load_fuse_generator}')

        filtered_state_dict = {}
        for key in loaded_state_dict.keys():
            # new
            if (load_encoder and key.startswith('smirk_encoder')) or (load_fuse_generator and key.startswith('smirk_generator')):
                filtered_state_dict[key] = loaded_state_dict[key]
            

        self.load_state_dict(filtered_state_dict, strict=False) # set it false because it asks for mica and other models apart from smirk_encoder and smirk_generator



    def set_freeze_status(self, config, batch_idx, epoch_idx):
        self.config.train.freeze_encoder_in_first_path = False
        self.config.train.freeze_generator_in_first_path = False
        self.config.train.freeze_encoder_in_second_path = False
        self.config.train.freeze_generator_in_second_path = False

        decision_idx = batch_idx if config.train.freeze_schedule.per_iteration else epoch_idx

        if config.train.freeze_schedule.generator_first:
            decision_idx = decision_idx + 1

        decision_idx_first_path = decision_idx
        decision_idx_second_path = decision_idx + 1 if config.train.freeze_schedule.alternate_first_second_path else decision_idx_first_path

        if config.train.freeze_schedule.apply_on_first_path:
            self.config.train.freeze_encoder_in_first_path = decision_idx_first_path % 2 == 0
            self.config.train.freeze_generator_in_first_path = (decision_idx_first_path + 1) % 2 == 0
        
        if config.train.freeze_schedule.apply_on_second_path:
            self.config.train.freeze_encoder_in_second_path = decision_idx_second_path % 2 == 0
            self.config.train.freeze_generator_in_second_path = (decision_idx_second_path + 1) % 2 == 0


        #print(f'freeze_encoder_in_first_path: {self.config.train.freeze_encoder_in_first_path}, freeze_generator_in_first_path: {self.config.train.freeze_generator_in_first_path}, freeze_encoder_in_second_path: {self.config.train.freeze_encoder_in_second_path}, freeze_generator_in_second_path: {self.config.train.freeze_generator_in_second_path}')