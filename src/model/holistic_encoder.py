import shutil
import os
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import random
import re
# from transformations import *
from .model_utils import MaskEncoder, BBoxEncoder
from base.baseTransformer import baseTransformer
from CONFIG import config
from torch.utils.tensorboard import SummaryWriter
import math


class HolisticEncoder(baseTransformer):
    
    """ 
    Vision Transformer for image reconstruction task
    """
    def __init__(self , mode, mask_ratio):
                #  patch_size, 
                #  embed_dim, 
                #  attn_dim, 
                #  num_heads, 
                #  mlp_size, 
                #  encoder_depth, 
                #  in_chans = 3, 
                #  max_len  = 64,
                # #  mask_ratio = 0.75,
                #  max_objects=11,
                #  use_masks=False,
                #  use_bboxes=False,
                #  norm_pix_loss=False):
        
        self.mode = mode
        self.mask_ratio = mask_ratio

        super().__init__(config=config)
        
        # encoder_embed_dim = config['vit_cfg']['encoder_embed_dim'] # encoder output
        # encoder_depth = config['vit_cfg']['encoder_depth']
        # self.embed_dim = embed_dim
        # self.use_masks = use_masks
        # self.use_bboxes = use_bboxes
        # self.mask_ratio = mask_ratio
        # self.norm_pix_loss = norm_pix_loss
        
        # Projection to transformer token dimension
        self.patch_projection = self.get_projection('encoder', in_dim=None)
        
        # Positional encoding
        self.encoder_pos_embed = self.get_positional_encoder(self.encoder_embed_dim)
        
        # Transformer blocks
        self.encoder_blocks = self.get_transformer_blocks(self.encoder_embed_dim, self.encoder_depth)
        
        # Layer normalization
        self.encoder_norm = self.get_ln(self.encoder_embed_dim)
        
        # Initialize weights
        self.initialize_weights()

        return


    def random_masking(self, x, mask_ratio):
        """
        Perform random masking on patchified sequences.
        
        Args:
            x: [B, T, N, D] 
                - B = batch
                - T = sequence length
                - N = number of patches
                - D = patch embedding dimension (flattened patch)
            mask_ratio: float, fraction of patches to mask

        Returns:
            x_masked: [B, T, N_keep, D]   (only visible patches)
            mask:     [B, T, N]           (0 = keep, 1 = masked)
            ids_restore: [B, T, N]        (indices to restore order)
        
        Code initiallly adapted from https://github.com/facebookresearch/mae/blob/main/models_mae.py and adjusted to our task
        """
        
        B, T, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio)) # How many tokens/patches to keep?

        # Generate random noise per sequence element
        noise = torch.rand(B, T, N, device=x.device)

        # Sort patches by noise
        ids_shuffle = torch.argsort(noise, dim=-1)          # [B, T, N]
        ids_restore = torch.argsort(ids_shuffle, dim=-1)    # [B, T, N]

        # Keep the first len_keep patches
        ids_keep = ids_shuffle[:, :, :len_keep]             # [B, T, N_keep]
        ids_keep_expanded = ids_keep.unsqueeze(-1).expand(-1, -1, -1, D)

        # Select the kept patches
        x_masked = torch.gather(x, dim=2, index=ids_keep_expanded)  # [B, T, N_keep, D]

        # Build mask: 0 = keep, 1 = masked
        mask = torch.ones([B, T, N], device=x.device)
        mask[:, :, :len_keep] = 0
        mask = torch.gather(mask, dim=2, index=ids_restore)  # reorder to original
        '''
        x_masked: only the unmasked (kept) tokens.
        mask: binary mask for the original sequence (0=keep, 1=mask). id's for which tokens to keep and mask
        ids_restore: the permutation needed to restore original order.
        '''
        return x_masked, mask, ids_restore

    
    def forward(self, images, masks=None, bboxes=None): # full Transformer encoder block forward pass
        """ 
        Forward pass
        """
        B, T = images.shape[:2]  
        all_masks = {}
        all_ids_restore = {}
        # Breaking image into patches
        image_patches = self.patchifier(images)
        
        # Projection to transformer token dimension
        image_tokens = self.patch_projection(image_patches)
        
        # Adding positional encoding
        image_tokens = self.encoder_pos_embed(image_tokens)

        if self.mode == 'training':
            # mask the image tokens in training mode
            image_tokens, image_mask, image_ids_restore = self.random_masking(image_tokens, mask_ratio= self.mask_ratio)
            
            encoded_features = self.encoder_blocks(image_tokens)
            encoded_features = self.encoder_norm(encoded_features)
            
            # Return dictionaries for consistency with expected interface
            all_masks = {'image': image_mask}
            all_ids_restore = {'image': image_ids_restore}
            return encoded_features, all_masks, all_ids_restore
            
        else: 
            # no masking in inference and predictor training mode
            encoded_features = self.encoder_blocks(image_tokens)
            encoded_features = self.encoder_norm(encoded_features)
            
            # Return dictionaries with None values for consistency
            all_masks = {'image': None}
            all_ids_restore = {'image': None}
            return encoded_features, all_masks, all_ids_restore
 