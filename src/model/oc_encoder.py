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


class ObjectCentricEncoder(baseTransformer):
    
    """ 
    Vision Transformer for image reconstruction task
    """
    def __init__(self , mode):

        self.mode = mode
        # self.mask_ratio = mask_ratio

        super().__init__(config=config)

        # Projection to transformer token dimension
        self.patch_projection = self.get_projection('oc_encoder')
        
        # Positional encoding
        self.encoder_pos_embed = self.get_positional_encoder(self.encoder_embed_dim)
        
        # Transformer blocks
        self.encoder_blocks = self.get_transformer_blocks(self.encoder_embed_dim, self.encoder_depth)
        
        # Layer normalization
        self.encoder_norm = self.get_ln(self.encoder_embed_dim)
        
        # Initialize weights
        self.initialize_weights()

        return

    def extract_object_frames_from_masks(self, images, masks):
        """
        images: Tensor of shape [B, T, C, H, W]
        masks: Tensor of shape [B, T, H, W] with int values from 0 to num_objects-1
        num_objects: int, number of unique objects (including background if needed)

        Returns:
            object_frames: Tensor of shape [B, T, num_objects, C, H, W]
        """
        B, T, C, H, W = images.shape
        device = images.device
        num_objects = 11 # Max number of objects in movi-c dataset (including background)
        
        # Expand images for each object
        object_frames = torch.zeros(B, T, num_objects, C, H, W, device=device, dtype=images.dtype)

        for obj_id in range(num_objects):
            # Create mask for this object: shape [B, T, 1, H, W]
            obj_mask = (masks == obj_id).unsqueeze(2)  # [B, T, 1, H, W]
            # Broadcast mask to all channels
            obj_mask = obj_mask.expand(-1, -1, C, -1, -1)  # [B, T, C, H, W]
            # Apply mask
            object_frames[:, :, obj_id] = images * obj_mask

        return object_frames    # shape: [B, T, 11, C, H, W]  e.g., torch.Size([32, 24, 11, 3, 128, 128])

    def extract_object_frames_from_bboxes(self, images, bboxes):
        """
        images: Tensor of shape [B, T, C, H, W]
        bboxes: Tensor of shape [B, T, num_objects, 4] (x1, y1, x2, y2) in pixel coordinates

        Returns:
            object_frames: Tensor of shape [B, T, num_objects, C, H, W]
        """
        B, T, C, H, W = images.shape
        device = images.device
        num_objects = bboxes.shape[2]

        # Prepare output tensor
        object_frames = torch.zeros(B, T, num_objects, C, H, W, device=device, dtype=images.dtype)

        for obj_id in range(num_objects):
            for b in range(B):
                for t in range(T):
                    x1, y1, x2, y2 = bboxes[b, t, obj_id]
                    # Clamp coordinates to image bounds and convert to int
                    x1 = int(torch.clamp(x1, 0, W-1).item())
                    y1 = int(torch.clamp(y1, 0, H-1).item())
                    x2 = int(torch.clamp(x2, 0, W-1).item())
                    y2 = int(torch.clamp(y2, 0, H-1).item())
                    # Ensure valid bbox
                    if x2 > x1 and y2 > y1:
                        # Copy the region from the image to the corresponding location in object_frames
                        object_frames[b, t, obj_id, :, y1:y2, x1:x2] = images[b, t, :, y1:y2, x1:x2]
                    
        return object_frames  # torch.Size([32, 24, 11, 3, 128, 128])

    def forward(self, images, masks=None, bboxes=None): # full Transformer encoder block forward pass
        """ 
        Forward pass
        """
        # B, T = images.shape[:2]  

        # Process images based on mode, with or without masking
        
        if self.use_masks and self.use_bboxes:
            raise ValueError("Either masks or bboxes must be provided! Not both!")

        # if self.mode == 'training':
            # Process images from masks
        if self.use_masks:
            if not masks:
                raise ValueError("Masks must be provided!")
            object_frames = self.extract_object_frames_from_masks(images, masks['masks'])
        
        # Process images from bboxes
        elif self.use_bboxes:
            if not bboxes:
                raise ValueError("Bboxes must be provided!")
            object_frames = self.extract_object_frames_from_bboxes(images, bboxes) # [B, T, num_objects, C, H, W]
        else:
            raise ValueError("Either masks or bboxes must be provided in object centric scene representation")

        # No patchifying here, each object frame is a token now
        
        B, T, num_frames, C, H, W = object_frames.shape
        patch_size = C*H*W
        # print(f'object frames shape: {object_frames.shape}')
        
        object_patches = object_frames.view(B,T,num_frames,patch_size)
        # print(f'object patches shape: {object_patches.shape}')
        # object_patches = self.patchifier(object_frames) # [B, T, num_objects, num_patches, patch_dim]
        
        # # Reshape object frames for patchifier: [B, T, num_objects, C, H, W] -> [B*T*num_objects, C, H, W]
        # B, T, num_objects, C, H, W = object_frames.shape
        # object_frames_reshaped = object_frames.view(B * T * num_objects, C, H, W)
        
        # # Add a dummy sequence dimension for patchifier: [B*T*num_objects, 1, C, H, W]
        # object_frames_reshaped = object_frames_reshaped.unsqueeze(1)
        
        # # Patchify: [B*T*num_objects, 1, num_patches, patch_dim]
        # object_patches = self.patchifier(object_frames_reshaped)
        
        # # Remove dummy sequence dimension and reshape back: [B*T*num_objects, num_patches, patch_dim] -> [B, T, num_objects, num_patches, patch_dim]
        # object_patches = object_patches.squeeze(1).view(B, T, num_objects, -1, object_patches.shape[-1])
        
        # # Reshape for transformer: [B, T, num_objects*num_patches, patch_dim]
        # num_patches = object_patches.shape[3]
        # object_patches = object_patches.view(B, T, num_objects * num_patches, -1)
        
        object_tokens = self.patch_projection(object_patches)
        # print(f'object tokens shape after projection: {object_tokens.shape}')
        
        object_tokens = self.encoder_pos_embed(object_tokens)
        # print(f'object tokens shape after pos embed: {object_tokens.shape}')
        object_tokens = self.encoder_blocks(object_tokens)
        # print(f'object tokens shape after blocks: {object_tokens.shape}')
        object_tokens = self.encoder_norm(object_tokens)
        # print(f'object tokens shape after norm, FINAL: {object_tokens.shape}')

        return object_tokens # [B, T, num_objects, embed_dim]