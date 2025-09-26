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
    def __init__(self , mode):

        
        self.mode = mode
        # self.mask_ratio = mask_ratio

        super().__init__(config=config)
        
        # Projection to transformer token dimension
        self.patch_projection = self.get_projection('encoder')
        
        # Positional encoding
        self.encoder_pos_embed = self.get_positional_encoder(self.encoder_embed_dim)
        
        # Transformer blocks
        self.encoder_blocks = self.get_transformer_blocks(self.encoder_embed_dim, self.encoder_depth)
        
        # Layer normalization
        self.encoder_norm = self.get_ln(self.encoder_embed_dim)
        
        # Initialize weights
        self.initialize_weights()

        return

    def forward(self, images, masks=None, bboxes=None): # full Transformer encoder block forward pass
        """ 
        Forward pass
        """
        B, T = images.shape[:2]  
        
        # Breaking image into patches
        image_patches = self.patchifier(images)
        
        # Projection to transformer token dimension
        image_tokens = self.patch_projection(image_patches)
        
        # Adding positional encoding
        image_tokens = self.encoder_pos_embed(image_tokens)

        # if self.mode == 'training':

        encoded_features = self.encoder_blocks(image_tokens)
        encoded_features = self.encoder_norm(encoded_features)
        
        return encoded_features