import torch
import os
from datalib import load_data, build_data_loader
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
import torchvision
import logging
from utils.logger import log_function
from utils.utils import create_directory, set_random_seed, save_model, save_config
from torch.utils.tensorboard import SummaryWriter



class baseTrainer:
    
    '''intended to be the base trainer for both Autoencoder and predictor modules'''
    
    def __init__(self, config) -> None:
        
        self.cfg = config 
        root = config['training']['root']
        self.exp_name = os.path.join('experiments',config['training']['model_name']) 
        self.exp_path = os.path.join(root, self.exp_name)
        self.tboard_logs_path = os.path.join(self.exp_path,"tboard_logs")
        create_directory(self.exp_path)
        create_directory(self.tboard_logs_path)
        
        self.use_predictor = self.cfg['vit_cfg']['use_predictor']
        
        self.training_losses = []
        self.validation_losses = []
        self.writer = SummaryWriter(log_dir=self.tboard_logs_path)
        
        self.data_loader()
        # self.setup_model()
        return
    
    
    def data_loader(self): 
        
        """
        Loading dataset and data-loaders
        """
        path = self.cfg['data']['dataset_path']
        train_dataset = load_data(path, split='train')
        val_dataset = load_data(path, split='validation')
        
        self.train_loader = build_data_loader(train_dataset, split='train')
        self.eval_loader = build_data_loader(val_dataset, split='validation')
        
        return
    
    
    def setup_model(self, model):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        
        set_random_seed()
        
        self.model = model.to(self.device)
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg['training']['lr'])
        
        # # Setup scheduler
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer, 
        #     T_max=self.cfg['training']['num_epochs']
        # )
        
        self.num_epochs = self.cfg['training']['num_epochs']
        
        return

    
    def train_epoch(self, epoch):
        
        """Training an Autoencoder model for one epoch """
        
        self.model.train()
        epoch_losses = []
        
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        
        for batch_idx, (images, masks, flows, coords) in progress_bar:

            images = images.to(self.device)
            # Normalize images to [0, 1] range
            images = images / 255.0
            
            # Prepare multi-modal data
            masks_data = {'masks': masks['masks'].to(self.device)} if masks is not None else None
            bboxes_data = coords['bbox'].to(self.device) if coords is not None else None
            
            # Clear gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            recons, loss, all_masks, all_ids_restore = self.model(images, masks_data, bboxes_data)
            
            # Debug: Check for NaN or inf loss
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"NaN or Inf loss detected at epoch {epoch+1}, batch {batch_idx}")
                logging.warning(f"Loss value: {loss.item()}")
                logging.warning(f"Images range: [{images.min().item():.4f}, {images.max().item():.4f}]")
                logging.warning(f"Recons range: [{recons.min().item():.4f}, {recons.max().item():.4f}]")
                continue
            
            # update progress bar
            if batch_idx % 10 == 0:
                progress_bar.set_description(f"Epoch {epoch+1} iter {batch_idx}: train loss {loss.item():.5f}.")
                  
            epoch_losses.append(loss.item())
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Debug: Check gradient norms
            if batch_idx % 50 == 0:
                logging.info(f"Gradient norm: {grad_norm:.6f}")
                logging.info(f"Loss: {loss.item():.6f}")
            
            # Update parameters
            self.optimizer.step()   
        
        mean_loss = np.mean(epoch_losses)
        
        self.training_losses.append(mean_loss)
        
        return 
    
    @torch.no_grad()
    def eval_model(self, epoch):
        
        """Evaluating the Autoencoder model"""
        
        self.model.eval()
        epoch_losses = []
        
        progress_bar = tqdm(enumerate(self.eval_loader), total=len(self.eval_loader))
        
        for batch_idx, (images, masks, flows, coords) in progress_bar:
            # Move data to device
            images = images.to(self.device)
            # Normalize images to [0, 1] range
            images = images / 255.0
            
            # Prepare multi-modal data
            masks_data = {'masks': masks['masks'].to(self.device)} if masks is not None else None
            bboxes_data = coords['bbox'].to(self.device) if coords is not None else None
            
            # Forward pass
            recons, loss, all_masks, all_ids_restore = self.model(images, masks_data, bboxes_data)
                                
            if epoch % 5 == 0 and batch_idx == 0:  # Only save first batch to avoid too many images
                # Reshape recons from [B, T, C, H, W] to [B*T, C, H, W] for visualization
                recons_vis = recons.view(-1, recons.shape[2], recons.shape[3], recons.shape[4])
                # Take only first 8 images for grid
                recons_vis = recons_vis[:8]
                
                # Create directory if it doesn't exist
                img_dir = os.path.join(self.tboard_logs_path, "imgs")
                os.makedirs(img_dir, exist_ok=True)
                
                grid = make_grid(recons_vis, nrow=4)
                self.writer.add_image('reconstructed_images', grid, global_step=epoch)
                save_image(grid, os.path.join(img_dir, f"recons_{epoch}.png"))

                
            if batch_idx % 10 == 0:
                progress_bar.set_description(f"Epoch {epoch+1} iter {batch_idx}: valid loss {loss.item():.5f}. ")

            epoch_losses.append(loss.item())
        
        mean_loss = np.mean(epoch_losses)
        
        self.validation_losses.append(mean_loss)
        
        return 
    
    
    def train_model(self, start_epoch=0):
        """ Training a model for a given number of epochs"""
            
        logging.info("Starting Autoencoder training...")

        
        for epoch in range(self.num_epochs):
            logging.info(f"Epoch {epoch+1}/{self.num_epochs}...")        

            # Validation epoch
            logging.info("  --> Running validation epoch")
            self.eval_model(epoch)
            
            self.writer.add_scalar(f'Loss/Valid', self.validation_losses[-1], global_step=epoch+start_epoch)
            
            # Training epoch
            logging.info("  --> Running train epoch")
            self.train_epoch(epoch)
            
            self.writer.add_scalar(f'Loss/Train', self.training_losses[-1], global_step=epoch+start_epoch)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar(f'LR',current_lr, global_step= epoch+start_epoch)
            
            # Step scheduler
            # self.scheduler.step()
            
            logging.info(f"Train loss: {round(self.training_losses[-1], 5)}")
            logging.info(f"Valid loss: {round(self.validation_losses[-1], 5)}")
            logging.info("\n")
        
        logging.info(f"Training completed")
        logging.info(f"Saving the config ...")
        
        save_config(self.cfg,self.exp_path)
        logging.info(f"Config Saved !")       
        logging.info("Saving final checkpoint")
        
        stats = {
            "train_loss": self.training_losses[-1],
            "valid_loss": self.validation_losses[-1],
                }

        save_model(self.model, self.optimizer, epoch=self.cfg['training']['num_epochs'], stats=stats, experiment_name=self.exp_name)
        logging.info("Finished Saving the model!")
        
        return self.training_losses, self.validation_losses