from .model_utils import PositionalEncoding, Patchifier
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from base.baseTransformer import baseTransformer
from CONFIG import config

class VitDecoder(baseTransformer):
    """
    Vision Transformer Decoder for multi-modal reconstruction task.
    Reconstructs images, masks, and bounding boxes from encoded features.
    """
        
    def __init__(self , mode):

        
        super().__init__(config=config)
        
        self.mode = mode

        self.decoder_projection = self.get_projection('decoder')
        self.decoder_pos_embed = self.get_positional_encoder(self.decoder_embed_dim)
        self.decoder_blocks = self.get_transformer_blocks(self.decoder_embed_dim, self.decoder_depth)
        self.decoder_norm = self.get_ln(self.decoder_embed_dim)
        
        self.decoder_pred_image = nn.Linear(self.decoder_embed_dim, self.patch_size**2 * self.out_chans, bias=True)
        
        # Initialize weights
        self.initialize_weights()
        
        return
    
    def unpatchify(self, x):
        """
        Convert predicted image patches back to full images.
        
        Args:
            x: [B, T, N, patch_size**2 * out_chans] - predicted patches
        
        Returns:
            reconstructed images as [B, T, C, H, W]
        """
        B, T, N, D = x.shape
        
        # Calculate dimensions
        grid_size = int(N**0.5)
        C = D // (self.patch_size * self.patch_size)
        H = W = grid_size * self.patch_size
        
        # [B, T, N, D] -> [B, T, grid_size, grid_size, C, patch_size, patch_size]
        x = x.reshape(B, T, grid_size, grid_size, C, self.patch_size, self.patch_size)
        
        # Reverse of patchify permute)
        # Patchify does: (0, 1, 3, 5, 2, 4, 6) -> (B, T, num_patch_H, num_patch_W, C, patch_size, patch_size)
        # So reverse should be: (0, 1, 4, 2, 5, 3, 6) -> (B, T, C, num_patch_H, patch_size, num_patch_W, patch_size)
        x = x.permute(0, 1, 4, 2, 5, 3, 6)
        
        # Reshape back to full image
        x = x.reshape(B, T, C, H, W)
        
        return x
    
    def patchify(self, imgs):
        """
        Convert images to patches (same as encoder).
        
        Args:
            imgs: [B, T, C, H, W]
        
        Returns:
            patches: [B, T, N, patch_dim]
        """
        return self.patchifier(imgs)
    
    def forward_loss(self, target_patches, pred_patches):
        """
        Compute reconstruction loss for different modalities.
        
        Args:
            targets: [B, T, N, patch_dim] 
            pred: [B, T, N, patch_dim] - predicted patches
        """
        loss_fn = nn.MSELoss()
        loss = loss_fn(target_patches, pred_patches)
        
        return loss
    
 
    def forward(self, encoded_features, target=None):
        """
        Forward pass through decoder.     
        Masked token vectors for the missing patches
    Encoder output vectors for the known patches
        
        Args:
            encoded_features: [B, T, N, embed_dim] - encoded features from encoder
            target: target tokens for loss computation
        
        Returns:
            predictions: Reconstructed images
            loss: Loss value
        """
        B, T, N, D = encoded_features.shape
        

        # Project to decoder dimension
        x = self.decoder_projection(encoded_features)  # [B, T, N_keep, decoder_embed_dim
                
        # Add positional encoding
        x = self.decoder_pos_embed(x)
        
        # Apply decoder blocks
        for block in self.decoder_blocks:
            x = block(x)
        
        # Final layer norm 
        x = self.decoder_norm(x)
        
        # Project predictions 
        pred_patches = self.decoder_pred_image(x)  # [B, T, N, ps*ps*C]
        
        # Compute loss/ only in training mode
        loss = None
        if self.mode == 'training':
            if target is not None:   
                target_patches = self.patchifier(target)
                loss = self.forward_loss(target_patches, pred_patches)
        
        # Reconstruct full images from predicted patches
        recons = self.unpatchify(pred_patches)
            
        return recons, loss