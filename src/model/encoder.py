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


class MultiModalVitEncoder(baseTransformer):
    
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
        
        encoder_embed_dim = config['vit_cfg']['encoder_embed_dim'] # encoder output
        encoder_depth = config['vit_cfg']['encoder_depth']
        # self.embed_dim = embed_dim
        # self.use_masks = use_masks
        # self.use_bboxes = use_bboxes
        # self.mask_ratio = mask_ratio
        # self.norm_pix_loss = norm_pix_loss
        
        # Projection to transformer token dimension
        self.patch_projection = self.get_projection('encoder')
        
        # Positional encoding
        self.encoder_pos_embed = self.get_positional_encoder(encoder_embed_dim)
        
        # Transformer blocks
        self.encoder_blocks = self.get_transformer_blocks(encoder_embed_dim, encoder_depth)
        
        # Layer normalization
        self.encoder_norm = self.get_ln(encoder_embed_dim)
        
        # Mask processing
        if self.use_masks:
            self.mask_encoder = MaskEncoder(self.patch_size, self.encoder_embed_dim, in_chans=1) # Mask has only 1 channel
            
        # Bounding box processing
        if self.use_bboxes:
            self.bbox_encoder = BBoxEncoder(self.encoder_embed_dim, self.max_objects)
            
        
        # Multi-modal embeddings
        if self.use_bboxes or self.use_masks:
            self.modality_embeddings = nn.Embedding(3, self.encoder_embed_dim) # “tag” the input stream ---> 0: image, 1: mask, 2: bbox
        
        # Initialize weights
        self.initialize_weights()

        return


    def random_masking(self, x):
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
        len_keep = int(N * (1 - self.mask_ratio)) # How many tokens/patches to keep?

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
        all_tokens = []
        all_masks = {}
        all_ids_restore = {}

        # Breaking image into patches
        image_patches = self.patchifier(images)
        
        # Projection to transformer token dimension
        image_tokens = self.patch_projection(image_patches)
        
        # Adding positional encoding
        image_tokens = self.encoder_pos_embed(image_tokens)


        # If using multiple modalities, add modality embedding for masks and bboxes
        if self.use_masks or self.use_bboxes:
            modality_emb = self.modality_embeddings(torch.zeros(B, T, 1, device=images.device, dtype=torch.long))
            image_tokens = image_tokens + modality_emb
        
        # Apply masking to images
        img_masked, img_mask, img_ids_restore = self.random_masking(image_tokens)
        all_tokens.append(img_masked)
        all_masks["image"] = img_mask
        all_ids_restore["image"] = img_ids_restore
        
        '''Single-modal input, only images''' 
        if not self.use_masks and not self.use_bboxes:
            encoded_features = self.encoder_blocks(img_masked)
            encoded_features = self.encoder_norm(encoded_features)
            return encoded_features, all_masks, all_ids_restore
        
        '''Multi-modal processing'''
        # Process masks if provided
        if self.use_masks and masks is not None:
            mask_tokens = self.mask_encoder(masks['masks'])
            mask_tokens = self.encoder_pos_embed(mask_tokens)
            
            # Add modality embedding for masks
            modality_emb = self.modality_embeddings(torch.ones(B, T, 1, device=masks['masks'].device, dtype=torch.long))
            mask_tokens = mask_tokens + modality_emb
            
            # Apply masking to masks
            mask_masked, mask_mask, mask_ids_restore = self.random_masking(mask_tokens)
            all_tokens.append(mask_masked)
            all_masks["mask"] = mask_mask
            all_ids_restore["mask"] = mask_ids_restore
        
        # Process bounding boxes if provided
        if self.use_bboxes and bboxes is not None:
            bbox_tokens = self.bbox_encoder(bboxes)
            bbox_tokens = self.encoder_pos_embed(bbox_tokens)
            
            # Add modality embedding for bboxes
            modality_emb = self.modality_embeddings(torch.full((B, T, 1), 2, device=bboxes.device, dtype=torch.long))
            bbox_tokens = bbox_tokens + modality_emb
            
            # Apply masking to bboxes
            bbox_masked, bbox_mask, bbox_ids_restore = self.random_masking(bbox_tokens)
            all_tokens.append(bbox_masked)
            all_masks["bbox"] = bbox_mask
            all_ids_restore["bbox"] = bbox_ids_restore
        
        # Concatenate all tokens
        if len(all_tokens) > 1:
            # Pad shorter sequences to match the longest
            max_length = max(token.shape[2] for token in all_tokens)
            padded_tokens = []
            
            for token in all_tokens:
                if token.shape[2] < max_length:
                    # Pad with zeros
                    padding = torch.zeros(B, T, max_length - token.shape[2], self.encoder_embed_dim, 
                                        device=token.device, dtype=token.dtype)
                    token = torch.cat([token, padding], dim=2)
                padded_tokens.append(token)
            
            combined_tokens = torch.cat(padded_tokens, dim=2)
        else:
            combined_tokens = all_tokens[0]
        
           
        # apply Transformer blocks
        encoded_features = self.encoder_blocks(combined_tokens)
        # print(f"Out tf_block tokens shape: {x.shape}")

        encoded_features = self.encoder_norm(encoded_features)

        return encoded_features, all_masks, all_ids_restore


    def get_attn_mask(self):
        """
        Fetching the last attention maps from all TF Blocks
        """
        attn_masks = [tf.get_attention_masks() for tf in self.encoder_blocks]
        return attn_masks


