import torch
import os
from datalib import load_data, build_data_loader
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
import torchvision
import logging
from utils.logger import log_function
from utils.utils import create_directory, set_random_seed, save_model, save_config, plot_sequence_comparison
from model.model_utils import get_scheduler
# from torch.utils.tensorboard import SummaryWriter
from utils.utils import TensorboardWriter



class baseTrainer:
    
    '''intended to be the base trainer for both Autoencoder and predictor modules'''
    
    def __init__(self, config) -> None:
        
        self.cfg = config 
        root = config['training']['root']
        self.exp_name = config['training']['model_name']
        self.exp_path = os.path.join(root, 'experiments',self.exp_name) 
        self.tboard_logs_path = os.path.join(self.exp_path,"tboard_logs", "imgs")
        self.config_path = os.path.join(self.exp_path,"config")
        self.checkpoints_path = os.path.join(self.exp_path,"checkpoints")
        create_directory(self.exp_path)
        create_directory(self.tboard_logs_path)
        create_directory(self.config_path)
        create_directory(self.checkpoints_path)
        
        self.training_losses = []
        self.validation_losses = []
        self.writer = TensorboardWriter(logdir=self.tboard_logs_path)
        
        self.data_loader()
        # self.setup_model()
        return
    
    
    def data_loader(self): 
        
        """
        Loading dataset and data-loaders
        """
        path = self.cfg['data']['dataset_path']
        train_dataset = load_data(path, split='train', use_transforms=True)
        val_dataset = load_data(path, split='validation', use_transforms=True)
        
        self.train_loader = build_data_loader(train_dataset, split='train')
        self.eval_loader = build_data_loader(val_dataset, split='validation')
        
        return
    
    
    def setup_model(self, model, mode):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        
        set_random_seed()
        
        self.training_mode = mode # "Autoencoder" or "Predictor"
        
        self.model = model.to(self.device)
        self.num_epochs = self.cfg['training']['num_epochs']
        
        # Setup optimizer with better memory efficiency
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg['training']['lr'])
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.cfg['training']['lr'],
            weight_decay=1e-4,  # Add weight decay for better generalization
            eps=1e-8,
            betas=(0.9, 0.95)  # Slightly more stable than default (0.9, 0.999)
        )
        
        # Setup scheduler
        # warmup epochs + cosine annealing
        warmup_epochs = self.cfg['training']['warmup_epochs']
        self.use_scheduler = self.cfg['training']['use_scheduler']
        if self.use_scheduler:
            self.scheduler = get_scheduler(self.optimizer, num_epochs=self.num_epochs, warmup_epochs= warmup_epochs) 
        
        self.use_early_stopping = self.cfg['training']['use_early_stopping']
        
        return

    
    def train_epoch(self, epoch):
        
        """Training an Autoencoder model for one epoch """
        
        self.model.train()
        epoch_losses = []
        
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        
        for batch_idx, (images, masks, flows, coords) in progress_bar:

            # Non-blocking GPU transfer for maximum overlap
            images = images.to(self.device, non_blocking=True)

            # Prepare multi-modal data with non-blocking transfer
            masks_data = {'masks': masks['masks'].to(self.device, non_blocking=True)} if masks is not None else None
            bboxes_data = coords['bbox'].to(self.device, non_blocking=True) if coords is not None else None
            
            # Clear gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            recons, loss = self.model(images, masks_data, bboxes_data)
            
            # update progress bar
            progress_bar.set_description(f"Epoch {epoch+1} batch {batch_idx}: train loss {loss.item():.5f}.")
                  
            epoch_losses.append(loss.item())
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
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
            # Move data to device with non-blocking transfer, reduce CPU ↔ GPU data transfer bottleneck, when pin_memory is True
            images = images.to(self.device, non_blocking=True)
            
            # Prepare multi-modal data with non-blocking transfer
            masks_data = {'masks': masks['masks'].to(self.device, non_blocking=True)} if masks is not None else None
            bboxes_data = coords['bbox'].to(self.device, non_blocking=True) if coords is not None else None
            
            # Forward pass
            recons, loss = self.model(images, masks_data, bboxes_data)
            
            # Reconstruction image saving for visualization - optimized for tensorboard
            if epoch % 5 == 0 and batch_idx == 0:  # Use first batch for consistency
                
                    # Reshape recons from [B, T, C, H, W] to [B*T, C, H, W] for visualization
                    B, T, C, H, W = recons.shape
                    recons_vis = recons.view(B * T, C, H, W)[:8]
                    images_vis = images.view(B * T, C, H, W)[:8]
                    
                    self.writer.plot_reconstruction_images(images_vis, recons_vis, epoch, self.tboard_logs_path, self.writer)
                 
            progress_bar.set_description(f"Epoch {epoch+1} batch {batch_idx}: valid loss {loss.item():.5f}. ")

            epoch_losses.append(loss.item())
        
        mean_loss = np.mean(epoch_losses)
        
        self.validation_losses.append(mean_loss)
        
        return 
    
    
    def train_model(self, start_epoch=0):
        """ Training a model for a given number of epochs"""
        
        logging.info(f"Saving the config ...")
        save_config(self.cfg, self.config_path , self.exp_name)
        logging.info(f"Config Saved !")   
        
        logging.info(f"Starting {self.training_mode} Training ...")
        
        # Initialize best validation loss for tracking
        best_val_loss = float('inf')
        save_frequency = self.cfg['training']['save_frequency']
        
        # Early stopping parameters
        early_stopping_patience = self.cfg['training']['early_stopping_patience']
        epochs_without_improvement = 0
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch + 1  
            logging.info(f"Epoch {self.current_epoch}/{self.num_epochs}...")        

            # Validation epoch
            logging.info("  --> Running validation epoch")
            self.eval_model(epoch)
            current_val_loss = self.validation_losses[-1]
            
            # Training epoch
            logging.info("  --> Running train epoch")
            self.train_epoch(epoch)
            current_train_loss = self.training_losses[-1]
            
            # Log metrics using original tensorboard writer methods
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log individual metrics
            self.writer.add_scalar('Loss/Train', current_train_loss, step=epoch+start_epoch)
            self.writer.add_scalar('Loss/Valid', current_val_loss, step=epoch+start_epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, step=epoch+start_epoch)
            
            # Create joint loss comparison plot
            self.writer.add_scalars('Loss/Comparison', ['Train', 'Validation'], [current_train_loss, current_val_loss], epoch+start_epoch)
            
            # Step scheduler
            if self.use_scheduler:
                self.scheduler.step()
            
            
            logging.info(f"Train loss: {round(current_train_loss, 5)}")
            logging.info(f"Valid loss: {round(current_val_loss, 5)}")
            
            stats = {
            "train_losses": self.training_losses,
            "valid_losses": self.validation_losses,
            "train_loss": current_train_loss if self.training_losses else 0,
            "valid_loss": current_val_loss if self.validation_losses else 0,
            "training_mode": self.training_mode,
            }

            # Check if this is the best model so far
            is_best = current_val_loss < best_val_loss
            if is_best:
                best_val_loss = current_val_loss
                epochs_without_improvement = 0  # Reset early stopping counter
                save_type = "best"
                logging.info(f"🎉 New best validation loss: {round(best_val_loss, 5)}")
            else:
                epochs_without_improvement += 1
                logging.warning(f"No improvement for {epochs_without_improvement} epochs!!")
                save_type = "checkpoint"
                
            # Early stopping check
            if self.use_early_stopping and epochs_without_improvement >= early_stopping_patience:
                logging.info(f"🛑 Early stopping triggered! No improvement for {early_stopping_patience} epochs.")
                break
                
            # Periodic checkpoint saving
            if (self.current_epoch % save_frequency == 0) or is_best:           
                save_model(self.model, 
                          self.optimizer, 
                          epoch=self.current_epoch, 
                          stats=stats, 
                          experiment_name=self.exp_name, 
                          path=self.checkpoints_path,
                          save_type=save_type,
                          training_mode=self.training_mode)

        logging.info(f"🏁 Training completed!")
            
        logging.info("Saving final checkpoint ...")
        # Save final checkpoint
        save_model(self.model, 
                   self.optimizer, 
                   epoch=self.current_epoch if hasattr(self, 'current_epoch') else self.cfg['training']['num_epochs'], 
                   stats=stats, 
                   experiment_name=self.exp_name, 
                   path=self.checkpoints_path,
                   save_type="final",
                   training_mode=self.training_mode)
        
        return 