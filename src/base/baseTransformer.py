import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from model.model_utils import TransformerBlock, Patchifier, PositionalEncoding
from CONFIG import config
from abc import ABC, abstractmethod 
from utils.logger import log_function

class baseTransformer(nn.Module, ABC):
    
    '''
    The base class to encapsulate the mutual functionalities of a transformer-based architecture
    '''
    
    def __init__(self, config) -> None:
        super().__init__()
        # self.cfg = config
        self.patch_size = config['data']['patch_size']
        self.max_objects =  config['data']['max_objects']
        self.encoder_embed_dim = config['vit_cfg']['encoder_embed_dim'] # encoder output
        # self.predictor_embed_dim = self.encoder_embed_dim # or five times this?! --> no , the input to predictor is five times this. This is the predictor output
        self.decoder_embed_dim = config['vit_cfg']['decoder_embed_dim'] # decoder input
        self.max_len = config['vit_cfg']['max_len']
        # self.norm_pix_loss = config['vit_cfg']['norm_pix_loss']
        self.out_chans = self.in_chans =  config['vit_cfg']['in_out_channels']
        self.use_masks =  config['vit_cfg']['use_masks']
        self.use_bboxes = config['vit_cfg']['use_bboxes']
        # self.mask_ratio = config['vit_cfg']['mask_ratio']
        self.attn_dim = config['vit_cfg']['attn_dim']
        self.num_heads = config['vit_cfg']['num_heads']
        self.mlp_size = config['vit_cfg']['mlp_size']
        self.num_preds = config['vit_cfg']['num_preds']
        self.predictor_window_size = config['vit_cfg']['predictor_window_size']
        self.predictor_embed_dim = config['vit_cfg']['predictor_embed_dim']
        self.residual = config['vit_cfg']['residual']
        
        self.image_height = config['data']['image_height']
        self.image_width = config['data']['image_width']
        
        self.encoder_depth = config['vit_cfg']['encoder_depth']
        self.decoder_depth = config['vit_cfg']['decoder_depth']
        self.predictor_depth = config['vit_cfg']['predictor_depth']
        
        self.patchifier = Patchifier(patch_size = self.patch_size) 
        
        return
    
    def get_projection(self, module_name):
        
        '''Prediction heads for different modalities'''
        
        if module_name == 'holistic_encoder':
            return nn.Sequential(   
                                 nn.LayerNorm(self.patch_size * self.patch_size * self.in_chans),
                                 nn.Linear(self.patch_size * self.patch_size * self.in_chans, self.encoder_embed_dim) # embed_dim = token embedding
                                )
        elif module_name == 'oc_encoder':
            input_dim = self.image_height * self.image_width * self.in_chans
            return nn.Sequential( nn.Linear(input_dim, input_dim // 8),  # 49,152 → 6,144
                                 nn.GELU(),
                                 nn.Dropout(0.1),
                                 nn.Linear(input_dim // 8, input_dim // 16),  # 6,144 → 3,072  
                                 nn.GELU(),
                                 nn.Dropout(0.1),
                                 nn.Linear(input_dim // 16, self.encoder_embed_dim) ) # 3,072 → 5 )
            # return nn.Sequential(     
            #     nn.LayerNorm(input_dim),
            #     nn.Linear(input_dim, self.encoder_embed_dim)
            #             )
        
        elif module_name == 'decoder':
            return nn.Linear(self.encoder_embed_dim, self.decoder_embed_dim, bias=True) # decoder input always encoder embedding size, predictor should also handle this
        
        elif module_name == 'predictor': # input and output projection
            return nn.Linear(self.encoder_embed_dim, self.predictor_embed_dim, bias=True),\
                   nn.Linear(self.predictor_embed_dim, self.encoder_embed_dim, bias=True)  

        else:
            raise ModuleNotFoundError('The given module does not exist! or the configs are not correct')
    
    def get_positional_encoder (self, embed_dim):
        '''
        Spatial and temporal Positional encoding for the transformer blocks
        '''
        return PositionalEncoding(d_model = embed_dim)
        

    def get_transformer_blocks (self, embed_dim, depth):
        
        transformer_blocks = [
            TransformerBlock(    # cascade of transformer blocks
                    embed_dim = embed_dim ,
                    attn_dim  = self.attn_dim,
                    num_heads = self.num_heads,
                    mlp_size  = self.mlp_size
                )
            for _ in range(depth)
        ]
        return nn.Sequential(*transformer_blocks)

    
    def get_ln(self, embed_dim):
        '''
        Layer normalization for the transformer blocks
        '''
        return nn.LayerNorm(embed_dim)


    def initialize_weights(self):
        """Initialize module parameters with transformer-friendly defaults.

        Rules:
        - Linear weights: Xavier uniform, biases to zero
        - LayerNorm: weights to 1, biases to zero
        - Embedding: normal with std=0.02
        - Mask Token: normal with std=0.2
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Initialize mask token with normal distribution
        if hasattr(self, "mask_token"):
            nn.init.normal_(self.mask_token, std=0.02)
    
    @abstractmethod
    def forward (self):
        pass
