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

        # self.norm_pix_loss = norm_pix_loss
        # self.patch_size = patch_size
        # self.out_chans = out_chans
        # self.decoder_embed_dim = decoder_embed_dim
        
        # Project encoded features to decoder dimension
        # self.decoder_embed = nn.Linear(self.predictor_embed_dim, self.decoder_embed_dim, bias=True)
        
        # Mask token for reconstruction
        '''
        The learnable mask token serves as a shared, 
        trainable embedding that fills those masked positions in the decoder's input sequence, providing an initial, 
        representation for the missing content. This allows the decoder's Transformer layers to attend to
        surrounding visible tokens and progressively refine predictions for the masked regions
        '''
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))
        
        decoder_input_dim = self.encoder_embed_dim if self.mode == 'training' else self.predictor_embed_dim
        
        self.decoder_projection = self.get_projection('decoder', in_dim= decoder_input_dim)
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
        
        # Reshape to patches
        x = x.reshape(B, T, N, self.patch_size, self.patch_size, -1)
        
        # Calculate image dimensions
        H = W = int(N**0.5) * self.patch_size # number of patches (in h and w) * patch_size
        
        # Reconstruct full image
        # x = x.permute(0, 1, 3, 2, 4, 5)  # [B, T, patch_size, H//patch_size, patch_size, W//patch_size, C]
        # x = x.reshape(B, T, H, W, -1)
        
        # # Permute to [B, T, C, H, W] format

        # x = x.permute(0, 1, 4, 2, 3)
        # Reshape to [B, T, sqrt(N), sqrt(N), patch_size, patch_size, C]
        x = x.reshape(B, T, int(N**0.5), int(N**0.5), self.patch_size, self.patch_size, -1)
        # Permute to [B, T, C, sqrt(N), patch_size, sqrt(N), patch_size]
        x = x.permute(0, 1, 6, 2, 4, 3, 5)
        # Reshape to [B, T, C, H, W]
        x = x.reshape(B, T, -1, H, W)
        
        return x
    
    def patchify(self, imgs):
        """
        Convert images to patches (same as encoder).
        
        Args:
            imgs: [B, T, C, H, W]
        
        Returns:
            patches: [B, T, N, patch_dim]
        """
        # patchifier = Patchifier(self.patch_size)
        return self.patchifier(imgs)
    
    def forward_loss(self, targets, pred, mask):
        """
        Compute reconstruction loss for different modalities.
        
        Args:
            targets: [B, T, C, H, W] or [B, T, N, 4] for bboxes
            pred: [B, T, N, patch_dim] - predicted patches
            mask: [B, T, N] - mask indicating which patches to reconstruct
            modality: 'image', 'mask', or 'bbox'
        """
        target_patches = self.patchifier(targets)
            
        # Normalize if specified
        if self.norm_pix_loss:
            mean = target_patches.mean(dim=-1, keepdim=True)
            var = target_patches.var(dim=-1, keepdim=True)
            target_patches = (target_patches - mean) / (var + 1.e-6)**.5
        
        # Compute loss
        loss = (pred - target_patches) ** 2
        loss = loss.mean(dim=-1)  # [B, T, N]
        
        # Avoid division by zero
        mask_sum = mask.sum()
        if mask_sum > 0:
            loss = (loss * mask).sum() / mask_sum  # mean loss on masked patches
        else:
            loss = loss.mean()  # fallback to mean loss if no masked patches
        
        
        return loss
    
 
    def forward(self, encoded_features, mask, ids_restore, target=None):
        """
        Forward pass through decoder.     
        Masked token vectors for the missing patches
    Encoder output vectors for the known patches
        
        Args:
            encoded_features: [B, T, N, embed_dim] - encoded features from encoder
            masks: dict with modality keys and mask tensors [B, T, N]
            ids_restore: dict with modality keys and restore indices [B, T, N]
            targets: dict with modality keys and target data for loss computation
        
        Returns:
            predictions: dict with reconstructed data for each modality
            losses: dict with losses for each modality (if targets provided)
        """
        B, T, N, D = encoded_features.shape
        

        # Project to decoder dimension
        x = self.decoder_projection(encoded_features)  # [B, T, N, decoder_embed_dim]
        
        if self.mode == 'training' and mask is not None:
            # Add mask tokens for masked positions
            
            # Extract image mask and ids_restore from dictionaries
            image_mask = mask['image'] 
            image_ids_restore = ids_restore['image']
                   
            N_total = image_mask.shape[2]
            N_keep = x.shape[2]
            num_masked = N_total - N_keep
            if num_masked > 0:
                mask_tokens_needed = self.mask_token.repeat(B, T, num_masked, 1)
                x = torch.cat([x, mask_tokens_needed], dim=2)  # [B, T, N_keep + num_masked, D] == [B, T, N_total, D]
            
            # Restore original order
            ids_restore_expanded = image_ids_restore.unsqueeze(-1).expand(-1, -1, -1, self.decoder_embed_dim)
            x = torch.gather(x, dim=2, index=ids_restore_expanded)
        
        elif self.mode == 'inference' or self.mode == 'predictor':
            x = encoded_features
                
        # Add positional encoding
        x = self.decoder_pos_embed(x)
        
        # Apply decoder blocks
        for block in self.decoder_blocks:
            x = block(x)
        
        # Final layer norm 
        x = self.decoder_norm(x)
        
        # Generate predictions 
        pred_patches = self.decoder_pred_image(x)  # [B, T, N, ps*ps*C]
        
        # Compute loss 
        loss = None
        if target is not None:        
            # Extract image mask from the dictionary
            image_mask = mask['image'] 
            loss = self.forward_loss(target, pred_patches, image_mask)
        
        # Reconstruct full images from predicted patches
        recons = self.unpatchify(pred_patches)
            
        return recons, loss