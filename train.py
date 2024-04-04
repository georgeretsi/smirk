import sys
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
from src.smirk_trainer import SmirkTrainer
import os
from datasets.data_utils import load_dataloaders
import src.utils.utils as utils


def parse_args():
    conf = OmegaConf.load(sys.argv[1])

    OmegaConf.set_struct(conf, True)

    sys.argv = [sys.argv[0]] + sys.argv[2:] # Remove the configuration file name from sys.argv

    conf.merge_with_cli()
    return conf


if __name__ == '__main__':
    # ----------------------- initialize configuration ----------------------- #
    config = parse_args()

    # ----------------------- initialize log directories ----------------------- #
    os.makedirs(config.train.log_path, exist_ok=True)
    train_images_save_path = os.path.join(config.train.log_path, 'train_images')
    os.makedirs(train_images_save_path, exist_ok=True)
    val_images_save_path = os.path.join(config.train.log_path, 'val_images')
    os.makedirs(val_images_save_path, exist_ok=True)
    OmegaConf.save(config, os.path.join(config.train.log_path, 'config.yaml'))

    train_loader, val_loader = load_dataloaders(config)

    trainer = SmirkTrainer(config)
    trainer = trainer.to(config.device)

    if config.resume:
        trainer.load_model(config.resume, load_fuse_generator=config.load_fuse_generator, load_encoder=config.load_encoder, device=config.device)

    # after loading, copy the base encoder 
    # this is used for regularization w.r.t. the base model as well as to compare the results    
    trainer.create_base_encoder()

    for epoch in range(config.train.resume_epoch, config.train.num_epochs):
        
        # restart everything at each epoch!
        trainer.configure_optimizers(len(train_loader))

        for phase in ['train', 'val']:
            loader = train_loader if phase == 'train' else val_loader
            
            for batch_idx, batch in tqdm(enumerate(loader), total=len(loader)):
                if batch is None:
                    continue

                trainer.set_freeze_status(config, batch_idx, epoch)

                batch = utils.preprocess_batch(batch, K=config.K, device=config.device)

                outputs = trainer.step(batch, batch_idx, phase=phase)

                if batch_idx % config.train.visualize_every == 0:
                    with torch.no_grad():
                        visualizations = trainer.create_visualizations(batch, outputs)
                        trainer.save_visualizations(visualizations, f"{config.train.log_path}/{phase}_images/{epoch}_{batch_idx}.jpg", show_landmarks=False)
                                    


        if epoch % config.train.save_every == 0:
            trainer.save_model(trainer.state_dict(), os.path.join(config.train.log_path, 'model_{}.pt'.format(epoch)))
