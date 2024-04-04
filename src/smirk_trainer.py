import torch.utils.data
import torch.nn.functional as F
from src.FLAME.FLAME import FLAME
from src.renderer.renderer import Renderer
from src.smirk_encoder import SmirkEncoder
from src.smirk_generator import SmirkGenerator
from src.base_trainer import BaseTrainer 
import numpy as np
import src.utils.utils as utils
import src.utils.masking as masking_utils

class SmirkTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        if self.config.arch.enable_fuse_generator:
            self.smirk_generator = SmirkGenerator(in_channels=6, out_channels=3, init_features=32, res_blocks=5)
            
        self.smirk_encoder = SmirkEncoder()
        
        self.flame = FLAME()

        self.renderer = Renderer(config)
        self.setup_losses()

        self.logger = None

        self.templates = utils.load_templates()
            
        # --------- setup flame masks for sampling --------- #
        self.face_probabilities = masking_utils.load_probabilities_per_FLAME_triangle()  


    def step1(self, batch):
        B, C, H, W = batch['img'].shape

        encoder_output = self.smirk_encoder(batch['img'])
        
        if self.config.train.freeze_encoder_in_first_path:
            encoder_output = {key: value.detach() for key, value in encoder_output.items()}
        
        with torch.no_grad():
            base_output = self.base_encoder(batch['img'])

        flame_output = self.flame.forward(encoder_output)
        
                
        renderer_output = self.renderer.forward(flame_output['vertices'], encoder_output['cam'],
                                                landmarks_fan=flame_output['landmarks_fan'], landmarks_mp=flame_output['landmarks_mp'])
        rendered_img = renderer_output['rendered_img']
        flame_output.update(renderer_output)
 
        # ---------------- losses ---------------- #
        losses = {}

        img = batch['img']

        # ---------------- landmark based losses ---------------- #
        valid_landmarks = batch['flag_landmarks_fan']
        losses['landmark_loss_fan'] = 0 if torch.sum(valid_landmarks) == 0 else F.mse_loss(flame_output['landmarks_fan'][valid_landmarks,:17], batch['landmarks_fan'][valid_landmarks,:17])

        losses['landmark_loss_mp'] = F.mse_loss(flame_output['landmarks_mp'], batch['landmarks_mp'])


        #  ---------------- regularization losses ---------------- # 
        if self.config.train.use_base_model_for_regularization:
            with torch.no_grad():
                base_output = self.base_encoder(batch['img'])
        else:
            base_output = {key[0]: torch.zeros(B, key[1]).to(self.config.device) for key in zip(['expression_params', 'shape_params', 'jaw_params'], [self.config.arch.num_expression, self.config.arch.num_shape, 3])}

        losses['expression_regularization'] = torch.mean((encoder_output['expression_params'] - base_output['expression_params'])**2)
        losses['shape_regularization'] = torch.mean((encoder_output['shape_params'] - base_output['shape_params'])**2)
        losses['jaw_regularization'] = torch.mean((encoder_output['jaw_params'] - base_output['jaw_params'])**2)


        if self.config.arch.enable_fuse_generator:
            masks = batch['mask']

            # mask out face and add random points inside the face
            rendered_mask = 1 - (rendered_img == 0).all(dim=1, keepdim=True).float()
            tmask_ratio = self.config.train.mask_ratio #* self.config.train.mask_ratio_mul # upper bound on the number of points to sample
            
            npoints, _ = masking_utils.mesh_based_mask_uniform_faces(flame_output['transformed_vertices'], 
                                                                     flame_faces=self.flame.faces_tensor,
                                                                     face_probabilities=self.face_probabilities,
                                                                     mask_ratio=tmask_ratio)
            
            extra_points = masking_utils.transfer_pixels(img, npoints, npoints)
            masked_img = masking_utils.masking(img, masks, extra_points, self.config.train.mask_dilation_radius, rendered_mask=rendered_mask)

            reconstructed_img = self.smirk_generator(torch.cat([rendered_img, masked_img], dim=1))

            # reconstruction loss
            img_mean, img_std = (torch.mean(img, dim=[2,3], keepdim=True).detach(), torch.std(img, dim=[2,3], keepdim=True).detach() + 1e-5) if self.config.train.norm_l1 else (0, 1)

            reconstruction_loss = F.l1_loss((reconstructed_img - img_mean)/img_std, (img - img_mean)/img_std, reduction='none')

            # for visualization
            loss_img = reconstruction_loss.mean(dim=1, keepdim=True)
            losses['reconstruction_loss'] = reconstruction_loss.mean()

            # perceptual loss
            losses['perceptual_vgg_loss'] = self.vgg_loss(reconstructed_img, img)

            # perceptual losses
            # perceptual_losses = 0
            if self.config.train.loss_weights['emotion_loss'] > 0:
                # do not let this gradient flow through the generator
                for param in self.smirk_generator.parameters():
                    param.requires_grad_(False)
                self.smirk_generator.eval()
                reconstructed_img_p = self.smirk_generator(torch.cat([rendered_img, masked_img], dim=1))
                for param in self.smirk_generator.parameters():
                    param.requires_grad_(True)
                self.smirk_generator.train()

                mead_mask = torch.Tensor(['mead' in tname.lower() for tname in batch['dataset_name']]).bool()
               
                valid_mask = mead_mask.to(img.device)
                if torch.sum(valid_mask) == 0:
                    losses['emotion_loss'] = 0
                else:
                    losses['emotion_loss'] = self.emotion_loss(reconstructed_img_p[valid_mask], img[valid_mask], metric='l2', use_mean=False)
                    #losses['emotion_loss'] = F.relu(losses['emotion_loss'] - 0.1).mean()
                    losses['emotion_loss'] = losses['emotion_loss'].mean()
                # perceptual_losses += losses['emotion_loss'] * self.config.train.loss_weights['emotion_loss']
            else:
                losses['emotion_loss'] = 0
        else:
            losses['reconstruction_loss'] = 0
            losses['perceptual_vgg_loss'] = 0
            losses['emotion_loss'] = 0

            
        if self.config.train.loss_weights['mica_loss'] > 0:
            losses['mica_loss'] = self.mica.calculate_mica_shape_loss(encoder_output['shape_params'], batch['img_mica'])
        else:
            losses['mica_loss'] = 0


        shape_losses = losses['shape_regularization'] * self.config.train.loss_weights['shape_regularization'] + \
                                    losses['mica_loss'] * self.config.train.loss_weights['mica_loss']

        expression_losses = losses['expression_regularization'] * self.config.train.loss_weights['expression_regularization'] + \
                            losses['jaw_regularization'] * self.config.train.loss_weights['jaw_regularization']
        
        landmark_losses = losses['landmark_loss_fan'] * self.config.train.loss_weights['landmark_loss'] + \
                            losses['landmark_loss_mp'] * self.config.train.loss_weights['landmark_loss'] 


        fuse_generator_losses = losses['perceptual_vgg_loss'] * self.config.train.loss_weights['perceptual_vgg_loss'] + \
                                losses['reconstruction_loss'] * self.config.train.loss_weights['reconstruction_loss'] + \
                                losses['emotion_loss'] * self.config.train.loss_weights['emotion_loss']
               
   
        loss_first_path = (
            (shape_losses if self.config.train.optimize_shape else 0) +
            (expression_losses if self.config.train.optimize_expression else 0) +
            (landmark_losses) +
            (fuse_generator_losses if self.config.arch.enable_fuse_generator else 0)
        )

        for key, value in losses.items():
            losses[key] = value.item() if isinstance(value, torch.Tensor) else value

        # ---------------- create a dictionary of outputs to visualize ---------------- #
        outputs = {}
        outputs['rendered_img'] = rendered_img
        outputs['vertices'] = flame_output['vertices']
        outputs['img'] = img
        if self.config.arch.enable_fuse_generator:
            outputs['loss_img'] = loss_img
            outputs['reconstructed_img'] = reconstructed_img
            outputs['masked_1st_path'] = masked_img

        for key in outputs.keys():
            outputs[key] = outputs[key].detach().cpu()

        outputs['encoder_output'] = encoder_output

        return outputs, losses, loss_first_path, encoder_output


        
    # ---------------- second path ---------------- #
    def step2(self, encoder_output, batch, batch_idx, phase='train'):
        B, C, H, W = batch['img'].shape        
        img = batch['img']
        masks = batch['mask']
        
        # number of multiple versions for the second path
        Ke = min(4, self.config.train.Ke)
        
        # start from the same encoder output and add noise to expression params
        # hard clone flame_feats
        flame_feats = {}
        for k, v in encoder_output.items():
            tmp = v.clone().detach()
            flame_feats[k] = torch.cat(Ke * [tmp], dim=0)

        # split Ke * B into 4 random groups
        gids = torch.randperm(Ke * B).to(self.config.device) 
        # 4 groups 

        # use augm percentages to split gids
        wperc = np.asarray([self.config.train.augm_rand_percent, 
                           self.config.train.augm_perm_percent, 
                           self.config.train.augm_inj_percent, 
                           self.config.train.augm_zero_percent])
        # wperc = [1., 1., 1., 1.]
        wperc = wperc / np.sum(wperc)
        wperc = np.cumsum(wperc) * Ke * B

        gids = [gids[:int(wperc[0])], gids[int(wperc[0]): int(wperc[1])], gids[int(wperc[1]): int(wperc[2])], gids[int(wperc[2]):]]

        feats_dim = flame_feats['expression_params'].size(1)        

        #''
        # ---------------- random expression ---------------- #
        # 1 of 4 Ke - random expressions!        
        if self.config.train.augm_rand_percent > 0:    
            param_mask = torch.bernoulli(torch.ones((len(gids[0]), feats_dim)) * 0.5).to(self.config.device)
            new_expressions = (torch.randn((len(gids[0]), feats_dim)).to(self.config.device)) * (1 + 2 * torch.rand((len(gids[0]), 1)).cuda()) * param_mask + flame_feats['expression_params'][gids[0]]
            flame_feats['expression_params'][gids[0]] = torch.clamp(new_expressions, -4.0, 4.0) +  (0 + 0.2 * torch.rand((len(gids[0]), 1)).cuda()) * torch.randn((len(gids[0]), feats_dim)).to(self.config.device)
            
            #expression_magn = (new_expressions ** 2).mean(dim=1) 
            #expresion_scale = (torch.clamp(expression_magn, 0, 2.0) / expression_magn).sqrt()
            #flame_feats['expression_params'][gids[0]] = expresion_scale.view(-1, 1) * new_expressions
        
        # ---------------- permutation of expression ---------------- #
        # 2 of 4 Ke - permutation!     
        if self.config.train.augm_perm_percent > 0:   
            flame_feats['expression_params'][gids[1]] = (0.25 + 1.5 * torch.rand((len(gids[1]), 1)).cuda()) * flame_feats['expression_params'][gids[1]][torch.randperm(len(gids[1]))] + (0 + 0.2 * torch.rand((len(gids[1]), 1)).cuda()) *  torch.randn((len(gids[1]), feats_dim)).to(self.config.device)
        
        # ---------------- template injection ---------------- #
        # 3 of 4 Ke - template injection!   
        if self.config.train.augm_inj_percent > 0:
            for i in range(len(gids[2])):
                expression = self.load_random_template(num_expressions=self.config.arch.num_expression)
                flame_feats['expression_params'][gids[2][i],:self.config.arch.num_expression] = (0.25 + 1.5 * torch.rand((1, 1)).cuda()) * torch.Tensor(expression).to(self.config.device)
            flame_feats['expression_params'][gids[2]] += (0 + 0.2 * torch.rand((len(gids[2]), 1)).cuda()) * torch.randn((len(gids[2]), feats_dim)).to(self.config.device)
        #'''
        

        # ---------------- tweak jaw for all paths ---------------- #
        scale_mask = torch.Tensor([1, .1, .1]).to(self.config.device).view(1, 3) * torch.bernoulli(torch.ones(Ke * B) * 0.5).to(self.config.device).view(-1, 1)
        flame_feats['jaw_params'] = flame_feats['jaw_params']  + torch.randn(flame_feats['jaw_params'].size()).to(self.config.device) * 0.2 * scale_mask
        flame_feats['jaw_params'][..., 0] = torch.clamp(flame_feats['jaw_params'][..., 0] , 0.0, 0.5)
        
        # ---------------- tweak eyelids for all paths ---------------- #
        if self.config.arch.use_eyelids:
            flame_feats['eyelid_params'] += (-1 + 2 * torch.rand(size=flame_feats['eyelid_params'].size()).to(self.config.device)) * 0.5
            # flame_feats['eyelid_params'] = torch.clamp(flame_feats['eyelid_params'], -0.5, 1.0)
            flame_feats['eyelid_params'] = torch.clamp(flame_feats['eyelid_params'], 0.0, 1.0)

        # ---------------- zero expression ---------------- #
        # 4 of 4 Ke - zero expression!     
        if self.config.train.augm_zero_percent > 0:   
            # use zero expression as one of the paths if Ke > 1 - let the eyelids to move a lot
            flame_feats['expression_params'][gids[3]] *= 0.0
            flame_feats['expression_params'][gids[3]] += (0 + 0.2 * torch.rand((len(gids[3]), 1)).cuda()) * torch.randn((len(gids[3]), flame_feats['expression_params'].size(1))).to(self.config.device)
            
            flame_feats['jaw_params'][gids[3]] *= 0.0
            flame_feats['eyelid_params'][gids[3]] = torch.rand(size=flame_feats['eyelid_params'][gids[3]].size()).to(self.config.device)        

        # small tweak on pose params
        #flame_feats['pose_params'] += torch.randn(flame_feats['pose_params'].size()).to(self.config.device) * 0.01 * torch.pi

        flame_feats['expression_params'] = flame_feats['expression_params'].detach()
        flame_feats['pose_params'] = flame_feats['pose_params'].detach()
        flame_feats['shape_params'] = flame_feats['shape_params'].detach()
        flame_feats['jaw_params'] = flame_feats['jaw_params'].detach()
        flame_feats['eyelid_params'] = flame_feats['eyelid_params'].detach()


        # after defining param augmentation, we can render the new faces
        with torch.no_grad():
            flame_output = self.flame.forward(encoder_output)
            rendered_output = self.renderer.forward(flame_output['vertices'], encoder_output['cam'])
            flame_output.update(rendered_output)
     
            # render the tweaked face
            flame_output_2nd_path = self.flame.forward(flame_feats)
            renderer_output_2nd_path = self.renderer.forward(flame_output_2nd_path['vertices'], encoder_output['cam'])
            rendered_img_2nd_path = renderer_output_2nd_path['rendered_img'].detach()

            
            # sample points for the image reconstruction
            # use the initial flame estimation to sample points from the initial image
            tmask_ratio = self.config.train.mask_ratio * 3.0 #* self.config.train.mask_ratio_mul 
            points1, sampled_coords = masking_utils.mesh_based_mask_uniform_faces(flame_output['transformed_vertices'], 
                                                                     flame_faces=self.flame.faces_tensor,
                                                                     face_probabilities=self.face_probabilities,
                                                                     mask_ratio=tmask_ratio)
            
           
            # apply repeat on sampled_coords elements
            sampled_coords['sampled_faces_indices'] = sampled_coords['sampled_faces_indices'].repeat(Ke, 1)
            sampled_coords['barycentric_coords'] = sampled_coords['barycentric_coords'].repeat(Ke, 1, 1)
            
            # get the sampled points that correspond to the face deformations
            points2, sampled_coords = masking_utils.mesh_based_mask_uniform_faces(renderer_output_2nd_path['transformed_vertices'], 
                                                                     flame_faces=self.flame.faces_tensor,
                                                                     face_probabilities=self.face_probabilities,
                                                                     mask_ratio=tmask_ratio,
                                                                     coords=sampled_coords)
            

            # transfer pixels from initial image to the new image
            extra_points = masking_utils.transfer_pixels(img.repeat(Ke, 1, 1, 1), points1.repeat(Ke, 1, 1), points2)
        
            
            rendered_mask = (rendered_img_2nd_path > 0).all(dim=1, keepdim=True).float()
                
        masked_img_2nd_path = masking_utils.masking(img.repeat(Ke, 1, 1, 1), masks.repeat(Ke, 1, 1, 1), extra_points, self.config.train.mask_dilation_radius, 
                                      rendered_mask=rendered_mask, extra_noise=True, random_mask=0.005)

        
        ww = 1
        
        reconstructed_img_2nd_path = self.smirk_generator(torch.cat([rendered_img_2nd_path, masked_img_2nd_path], dim=1).detach())
        if self.config.train.freeze_generator_in_second_path:
            reconstructed_img_2nd_path = reconstructed_img_2nd_path.detach()
            #ww = 10.0 
        
        
        #if self.config.train.freeze_generator_in_second_path:
        #    ww = 10
        #else:
        #    ww = 1
         
        '''
        if self.config.train.freeze_generator_in_second_path:
            with torch.no_grad():
                reconstructed_img_2nd_path = self.smirk_generator(torch.cat([rendered_img_2nd_path, masked_img_2nd_path], dim=1).detach())
            #print('detached')
            reconstructed_img_2nd_path = reconstructed_img_2nd_path.detach()
            #ww = 100
        else:
            reconstructed_img_2nd_path = self.smirk_generator(torch.cat([rendered_img_2nd_path, masked_img_2nd_path], dim=1).detach())
        '''
        
        recon_feats = self.smirk_encoder(reconstructed_img_2nd_path.view(Ke * B, C, H, W)) 
        #if self.config.train.freeze_encoder_in_second_path:
        #    ww = .1 
            
        #if self.config.train.freeze_encoder_in_second_path:
        #    for key in recon_feats.keys():
        #        recon_feats[key].requires_grad_(True)

        #with torch.no_grad():
        flame_output_2nd_path_2 = self.flame.forward(recon_feats)
        rendered_img_2nd_path_2 = self.renderer.forward(flame_output_2nd_path_2['vertices'], recon_feats['cam'])['rendered_img']

        losses = {}
        
        cycle_loss = 1.0 * F.mse_loss(recon_feats['expression_params'], flame_feats['expression_params']) + \
                     10.0 * F.mse_loss(recon_feats['jaw_params'], flame_feats['jaw_params'])

        
        if self.config.arch.use_eyelids:
            cycle_loss += 10.0 * F.mse_loss(recon_feats['eyelid_params'], flame_feats['eyelid_params'])

        
        if not self.config.train.freeze_generator_in_second_path:                
            
            cycle_loss += 1.0 * F.mse_loss(recon_feats['shape_params'], flame_feats['shape_params']) 

            #cycle_loss *= 0.1 # scale down generator loss



        
        losses['cycle_loss']  = cycle_loss
        loss_second_path = ww * losses['cycle_loss'] * self.config.train.loss_weights.cycle_loss

        '''
        if self.config.train.loss_weights['identity_loss'] > 0:
            # freeze the generator
            identity_loss = self.identity_loss(reconstructed_img_2nd_path, img.repeat(Ke, 1, 1, 1), use_mean=False)
            losses['identity_loss'] = F.relu(identity_loss - 0.3).mean()
            
            loss_second_path +=  losses['identity_loss'] * self.config.train.loss_weights['identity_loss'] 
        '''

        for key, value in losses.items():
            losses[key] = value.item() if isinstance(value, torch.Tensor) else value


        # ---------------- visualization struct ---------------- #
        
        outputs = {}
        if batch_idx % self.config.train.visualize_every == 0:
            outputs['2nd_path'] = torch.stack([rendered_img_2nd_path.detach().cpu().view(Ke, B,  C, H, W).permute(1, 0 , 2, 3, 4).reshape(-1, C, H, W), 
                                             masked_img_2nd_path.detach().cpu().view(Ke, B, C, H, W).permute(1, 0 , 2, 3, 4).reshape(-1, C, H, W),
                                             reconstructed_img_2nd_path.detach().cpu().view(Ke, B,  C, H, W).permute(1, 0 , 2, 3, 4).reshape(-1, C, H, W), 
                                             rendered_img_2nd_path_2.detach().cpu().view(Ke, B, C, H, W).permute(1, 0 , 2, 3, 4).reshape(-1, C, H, W)], dim=1).reshape(-1, C, H, W)

            
            # create images of the histograms for expression params
            '''
            import matplotlib.pyplot as plt
            
            def histogram_to_image(data): 
                
                data = data.flatten()
                
                #hist, bin_edges = np.histogram(data, bins=bins)
                
                # plt bar 
                
                #fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
                #ax.bar(bin_edges[:-1], hist)
                #ax.bar(np.arange(data.shape[0]), data)
                
                fig = plt.Figure(figsize=(7,5), dpi=300)
                # canvas = fig.canvas
                indices = np.arange(data.shape[0])
                plt.bar(indices, data)
                # y axis range from -4 to 4
                plt.ylim(-4, 4)
                plt.savefig('temp.png')
                plt.close()
                
                image = cv2.imread('temp.png')
                
                
                # resize to config.train.image_size
                image = cv2.resize(image, (self.config.image_size, self.config.image_size)) /255.0
                
                return image
            
            
            hist1 = np.stack([histogram_to_image(tmp) for tmp in flame_feats['expression_params'].detach().cpu().numpy()])
            hist2 = np.stack([histogram_to_image(tmp) for tmp in recon_feats['expression_params'].detach().cpu().numpy()])
            
            outputs['hist1'] = torch.from_numpy(hist1).permute(0, 3, 1, 2)
            outputs['hist2'] = torch.from_numpy(hist2).permute(0, 3, 1, 2)
            '''
            
            
        return outputs, losses, loss_second_path

    def freeze_encoder(self):
        utils.freeze_module(self.smirk_encoder.pose_encoder, 'pose encoder')
        utils.freeze_module(self.smirk_encoder.shape_encoder, 'shape encoder')
        utils.freeze_module(self.smirk_encoder.expression_encoder, 'expression encoder')
        
    def unfreeze_encoder(self):
        if self.config.train.optimize_pose:
            utils.unfreeze_module(self.smirk_encoder.pose_encoder, 'pose encoder')
        
        if self.config.train.optimize_shape:
            utils.unfreeze_module(self.smirk_encoder.shape_encoder, 'shape encoder')
            
        if self.config.train.optimize_expression:
            utils.unfreeze_module(self.smirk_encoder.expression_encoder, 'expression encoder')

    def step(self, batch, batch_idx, phase='train'):
        # ------- set the model to train or eval mode ------- #
        if phase == 'train':
            self.train()
            torch.set_grad_enabled(True)
        else:
            self.eval()
            torch.set_grad_enabled(False)


        '''
        if phase == 'train':
            # ------- freeze the parts of the encoder that we don't optimize for ------- #
            utils.freeze_module(self.smirk_encoder.pose_encoder, 'pose encoder') if not self.config.train.optimize_pose else None
            utils.freeze_module(self.smirk_encoder.shape_encoder, 'shape encoder') if not self.config.train.optimize_shape else None
            utils.freeze_module(self.smirk_encoder.expression_encoder, 'expression encoder') if not self.config.train.optimize_expression else None
    
            if self.config.train.freeze_encoder_in_first_path:
                utils.freeze_module(self.smirk_encoder, 'encoder')
            if self.config.train.freeze_generator_in_first_path:
                utils.freeze_module(self.smirk_generator, 'fuse generator')
        '''
                
        outputs1, losses1, loss_first_path, encoder_output = self.step1(batch)

        if phase == 'train':
            
            message = 'First path '
            if self.config.train.freeze_encoder_in_first_path:
                self.freeze_encoder()


            #    message += 'freeze encoder '
            #else:
            #    self.unfreeze_encoder()
            #    message += 'unfreeze encoder '
                
            #utils.unfreeze_module(self.smirk_generator, 'fuse generator')
            if self.config.train.freeze_generator_in_first_path:
                utils.freeze_module(self.smirk_generator, 'fuse generator')
                
            #    message += 'freeze generator'
            #else:
            #    utils.unfreeze_module(self.smirk_generator, 'fuse generator')
            #    message += 'unfreeze generator'
                
            #print(message)
            
            
            self.optimizers_zero_grad()
            loss_first_path.backward()
            self.optimizers_step(step_encoder=not self.config.train.freeze_encoder_in_first_path, 
                                step_fuse_generator=not self.config.train.freeze_generator_in_first_path)

             
            if self.config.train.freeze_encoder_in_first_path:
                self.unfreeze_encoder()
            if self.config.train.freeze_generator_in_first_path:
                utils.unfreeze_module(self.smirk_generator, 'fuse generator')
                
        if (self.config.train.loss_weights['cycle_loss'] > 0) and (phase == 'train'):
           
            #self.config.train.freeze_encoder_in_second_path = False
            #self.config.train.freeze_generator_in_second_path = True
           
           
            if self.config.train.freeze_encoder_in_second_path:
                self.freeze_encoder()
                #self.smirk_encoder.eval() # cant freeze the encoder here, so just set it to eval mode
            if self.config.train.freeze_generator_in_second_path:
                utils.freeze_module(self.smirk_generator, 'fuse generator')
             
            '''
            message = 'Second path '
            if self.config.train.freeze_encoder_in_second_path:
                self.freeze_encoder()
                #self.unfreeze_encoder()
                #self.smirk_encoder.eval()
                message += 'freeze encoder '
            else:
                self.unfreeze_encoder()
                message += 'unfreeze encoder '
            
            if self.config.train.freeze_generator_in_second_path:
                utils.freeze_module(self.smirk_generator, 'fuse generator')
                message += 'freeze generator'
            else:
                utils.unfreeze_module(self.smirk_generator, 'fuse generator') 
                message += 'unfreeze generator'
                
            #print(message)
            '''
                    
            outputs2, losses2, loss_second_path = self.step2(encoder_output, batch, batch_idx, phase)
            
            self.optimizers_zero_grad()
            loss_second_path.backward()
            
            # clip !
            #if not self.config.train.freeze_encoder_in_second_path:
            #    torch.nn.utils.clip_grad_norm_(self.smirk_encoder.parameters(), 1e-2)

            #if not self.config.train.freeze_generator_in_second_path:
            #    torch.nn.utils.clip_grad_norm_(self.smirk_generator.parameters(), 1e-2)

            self.optimizers_step(step_encoder=not self.config.train.freeze_encoder_in_second_path, 
                                 step_fuse_generator=not self.config.train.freeze_generator_in_second_path)

            losses1.update(losses2)
            outputs1.update(outputs2)

            if self.config.train.freeze_encoder_in_second_path:
                self.unfreeze_encoder()
            
            if self.config.train.freeze_generator_in_second_path:
                utils.unfreeze_module(self.smirk_generator, 'fuse generator')
        
        losses = losses1
        self.logging(batch_idx, losses, phase)
        
        if phase == 'train':
            self.scheduler_step()

        return outputs1

