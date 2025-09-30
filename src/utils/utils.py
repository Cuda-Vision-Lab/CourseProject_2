import os
import numpy as np
import torch.nn as nn
import torch
import random
import logging
import json
from matplotlib import pyplot as plt
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter

    
def set_random_seed(random_seed=None):
    """
    Using random seed for numpy and torch
    """
    if(random_seed is None):
        random_seed = 42
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    return

def init_weights(m):
    # initialize nn.Linear and nn.LayerNorm
    if isinstance(m, nn.Linear):
        # we use xavier_uniform following official JAX ViT:
        torch.nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
            


def smooth(f, K=5):
    """ Smoothing a function using a low-pass filter (mean) of size K """
    kernel = np.ones(K) / K
    f = np.concatenate([f[:int(K//2)], f, f[int(-K//2):]])  # to account for boundaries
    smooth_f = np.convolve(f, kernel, mode="same")
    smooth_f = smooth_f[K//2: -K//2]  # removing boundary-fixes
    return smooth_f


def save_model(model, optimizer, epoch, stats, experiment_name, path, save_type="checkpoint", training_mode="Autoencoder"):
    """ 
    model checkpoint saving 
    """
    import os
    
    # Ensure directory exists
    os.makedirs(path, exist_ok=True)
    
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'stats': stats,
        'save_type': save_type,
        'training_mode': training_mode,
        'experiment_name': experiment_name
    }
    
    # Add state dicts based on training mode
    if training_mode == "Autoencoder" and hasattr(model, 'encoder') and hasattr(model, 'decoder'):
        checkpoint_data['encoder_state_dict'] = model.encoder.state_dict()
        checkpoint_data['decoder_state_dict'] = model.decoder.state_dict()
    elif training_mode == "Predictor" and hasattr(model, 'predictor'):
        checkpoint_data['predictor_state_dict'] = model.predictor.state_dict()
    
    # Add lr
    if hasattr(optimizer, 'param_groups'):
        checkpoint_data['learning_rate'] = optimizer.param_groups[0]['lr']
    
    try:
        if save_type == "final":
            savepath = f"{path}/final_{experiment_name}.pth"
        elif save_type == "best":
            savepath = f"{path}/best_{experiment_name}.pth"
        else:
            # Periodic checkpoint 
            savepath = f"{path}/checkpoint_{experiment_name}_epoch_{epoch:04d}.pth"
        
        torch.save(checkpoint_data, savepath)
                     
        logging.info(f"Saved {save_type} checkpoint: {savepath}\n")
        return savepath
        
    except Exception as e:
        logging.error(f"Failed to save checkpoint: {e}")
        return None

def save_config(config, path, experiment_name):
    """
    Saves the config (dict or config.py module) as a JSON file in the given path.

    Args:
        config: Either a config dictionary or a config.py module (with attributes).
        path: The file path where the JSON should be saved.
    """
    config_name = f"{experiment_name}.json"
    path = os.path.join(path, config_name)
    with open(path, "w") as f:
        json.dump(config, f, indent=4)
    return

def load_model(model, mode, path_AE, path_predictor=None):
    """ Loading pretrained checkpoint """
    
    checkpoint = torch.load(path_AE, map_location="cpu", weights_only=False)
    
    if mode == "predictor_training":
        model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        
        # Freeze encoder and decoder for training the predictor
        model.encoder.requires_grad_(False)
        model.decoder.requires_grad_(False)
        
    elif mode == "AE_inference":
        model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        
    elif mode == "inference":
        model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        checkpoint_predictor = torch.load(path_predictor, map_location="cpu", weights_only=False)    
        model.predictor.load_state_dict(checkpoint_predictor['predictor_state_dict'])
 
    return model


def count_model_params(model):
    """ Counting the number of learnable parameters in a nn.Module """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params

def create_directory(exp_path):
    """
    Creating a folder in given path.
    """
    # if(not os.path.exists(exp_path)):
    os.makedirs(exp_path,exist_ok=True)
    # if(dir_name is not None):
    #     dir_path = os.path.join(dir_path, dir_name)
    return

class TensorboardWriter:
    """
    Class for handling the tensorboard logger

    Args:
    -----
    logdir: string
        path where the tensorboard logs will be stored
        
    This code is adapted from https://github.com/AIS-Bonn/OCVP-object-centric-video-prediction/tree/master
    """

    def __init__(self, logdir):
        """ Initializing tensorboard writer """
        self.logdir = logdir
        self.writer = SummaryWriter(logdir)
        return

    def add_scalar(self, name, val, step):
        """ Adding a scalar for plot """
        self.writer.add_scalar(name, val, step)
        return

    def add_scalars(self, plot_name, val_names, vals, step):
        """ Adding several values in one plot """
        val_dict = {val_name: val for (val_name, val) in zip(val_names, vals)}
        self.writer.add_scalars(plot_name, val_dict, step)
        return

    def add_image(self, fig_name, img_grid, step):
        """ Adding a new step image to a figure """
        try:
            # Ensure image tensor is properly formatted for tensorboard
            if img_grid.dim() == 4:
                # If batch dimension exists, take first image
                img_grid = img_grid[0]
            
            # Ensure tensor is on CPU and detached
            if hasattr(img_grid, 'detach'):
                img_grid = img_grid.detach().cpu()
            
            # Add image to tensorboard
            self.writer.add_image(fig_name, img_grid, global_step=step)
            
            # Flush to ensure immediate writing
            self.writer.flush()
            
        except Exception as e:
            print(f"Error adding image {fig_name} to tensorboard: {str(e)}")
            
        return

    def add_figure(self, tag, figure, step):
        """ Adding a whole new figure to the tensorboard """
        self.writer.add_figure(tag=tag, figure=figure, global_step=step)
        return
    
    def add_graph(self, model, input):
        """ Logging model graph to tensorboard """
        self.writer.add_graph(model, input_to_model=input)
        return

    def log_full_dictionary(self, dict, step, plot_name="Losses", dir=None):
        """
        Logging a bunch of losses into the Tensorboard. Logging each of them into
        its independent plot and into a joined plot
        """
        if dir is not None:
            dict = {f"{dir}/{key}": val for key, val in dict.items()}
        else:
            dict = {key: val for key, val in dict.items()}

        for key, val in dict.items():
            self.add_scalar(name=key, val=val, step=step)

        plot_name = f"{dir}/{plot_name}" if dir is not None else key
        self.add_scalars(plot_name=plot_name, val_names=dict.keys(), vals=dict.values(), step=step)
        return
    

    def plot_reconstruction_images(self, images_vis, recons_vis, epoch, path, writer):
    
        # Ensure data is on CPU and properly normalized for tensorboard
        recons_vis = recons_vis.detach().cpu()
        images_vis = images_vis.detach().cpu()
        
        # Robust normalization function that handles varying tensor ranges consistently
        def normalize_for_tensorboard_robust(tensor, reference_tensor=None):
            """
            Normalize tensor to [0, 1] range using consistent approach.
            
            Args:
                tensor: Input tensor to normalize
                reference_tensor: Optional reference tensor to use for consistent scaling
            """
            # Use reference tensor statistics if provided (for consistent scaling)
            if reference_tensor is not None:
                ref_min = reference_tensor.min()
                ref_max = reference_tensor.max()
                
                # Determine the likely data range from reference
                if ref_max <= 1.1 and ref_min >= -0.1:
                    # Likely [0, 1] or [-1, 1] range
                    if ref_min >= -0.1:
                        # [0, 1] range
                        return torch.clamp(tensor, 0, 1)
                    else:
                        # [-1, 1] range, convert to [0, 1]
                        return torch.clamp((tensor + 1.0) / 2.0, 0, 1)
                else:
                    # Likely [0, 255] or other range, use reference min/max
                    return torch.clamp((tensor - ref_min) / (ref_max - ref_min + 1e-8), 0, 1)
            
            # Fallback: use tensor's own statistics
            tensor_min = tensor.min()
            tensor_max = tensor.max()
            
            # More robust range detection with tolerance
            if tensor_max <= 1.1 and tensor_min >= -0.1:
                if tensor_min >= -0.1:
                    # [0, 1] range
                    return torch.clamp(tensor, 0, 1)
                else:
                    # [-1, 1] range, convert to [0, 1]
                    return torch.clamp((tensor + 1.0) / 2.0, 0, 1)
            else:
                # Other range, normalize using min-max
                return torch.clamp((tensor - tensor_min) / (tensor_max - tensor_min + 1e-8), 0, 1)
        
        # Normalize images first (as reference)
        images_vis = normalize_for_tensorboard_robust(images_vis)
        
        # Normalize reconstructions using the same approach as images for consistency
        recons_vis = normalize_for_tensorboard_robust(recons_vis, reference_tensor=images_vis)
        
        # Additional safety: ensure both are in [0, 1] range
        images_vis = torch.clamp(images_vis, 0, 1)
        recons_vis = torch.clamp(recons_vis, 0, 1)
        
        # Final check: if reconstructions still look wrong, apply percentile-based normalization
        if recons_vis.max() > 0.99 and recons_vis.min() < 0.01:
            # Use percentile-based normalization for extreme cases
            recons_p1 = torch.quantile(recons_vis, 0.01)
            recons_p99 = torch.quantile(recons_vis, 0.99)
            if recons_p99 > recons_p1:
                recons_vis = torch.clamp((recons_vis - recons_p1) / (recons_p99 - recons_p1), 0, 1)
        
        # Create grids with proper settings for tensorboard
        nrow = 4
        
        # Use make_grid with normalize=False since we already normalized
        input_grid = make_grid(images_vis, nrow=nrow, normalize=False, pad_value=1.0)
        output_grid = make_grid(recons_vis, nrow=nrow, normalize=False, pad_value=1.0)
        
        # Add images to tensorboard with descriptive tags
        if epoch == 0:
            writer.add_image('Visualization/Imgs', input_grid, step=epoch)
        writer.add_image('Visualization/Recons', output_grid, step=epoch)
        
        # Create side-by-side comparison for better visualization
        comparison_grid = torch.cat([input_grid, output_grid], dim=2)  # Horizontal concatenation
        writer.add_image('Visualization/Imgs_vs_Recons', comparison_grid, step=epoch)
        
        # Also save to disk for backup
        save_image(input_grid, os.path.join(path, f"input_epoch_{epoch}.png"))
        save_image(output_grid, os.path.join(path, f"recons_epoch_{epoch}.png"))
        save_image(comparison_grid, os.path.join(path, f"comparison_epoch_{epoch}.png"))
        
        # Log normalization info for debugging
        logging.info(f"✅ Saved visualization images to tensorboard for epoch {epoch}")

