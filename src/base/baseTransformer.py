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
        self.predictor_embed_dim = self.encoder_embed_dim # or five times this?! --> no , the input to predictor is five times this. This is the predictor output
        self.decoder_embed_dim = config['vit_cfg']['decoder_embed_dim'] # decoder input
        self.max_len = config['vit_cfg']['max_len']
        self.norm_pix_loss = config['vit_cfg']['norm_pix_loss']
        self.out_chans = self.in_chans =  config['vit_cfg']['in_out_channels']
        self.use_masks =  config['vit_cfg']['use_masks']
        self.use_bboxes = config['vit_cfg']['use_bboxes']
        # self.mask_ratio = config['vit_cfg']['mask_ratio']
        self.attn_dim = config['vit_cfg']['attn_dim']
        self.num_heads = config['vit_cfg']['num_heads']
        self.mlp_size = config['vit_cfg']['mlp_size']
        
        self.encoder_depth = config['vit_cfg']['encoder_depth']
        self.decoder_depth = config['vit_cfg']['decoder_depth']
        self.predictor_depth = config['vit_cfg']['predictor_depth']
        # self.use_predictor = config['vit_cfg']['use_predictor']
        
        
        self.patchifier = Patchifier(patch_size = self.patch_size)
        
        # #VitEncoder
        # module_name = 'encoder'
        
        # functions = [self.get_positional_encoder, self.get_projection, self.get_transformer_blocks, self.get_ln]
        
        # self.encoder_pos_embed, self.patch_projection, self.encoder_blocks, self.encoder_norm = list(map(lambda f : f(module_name),functions))
        
        ''' Image processing. Creating the embedding for each image patch/token'''
        
        #VitDecoder
        # module_name = 'decoder'
        
        # self.decoder_pos_embed, self.decoder_projection, self.decoder_blocks, self.decoder_norm = list(map(lambda f : f(module_name),functions))
        
        # self.decoder_pred_image = nn.Linear(self.decoder_embed_dim, self.patch_size**2 * self.out_chans, bias=True)
        
        # #Predictor
        # module_name = 'predictor'
          

        return
    
    def get_projection(self, module_name, in_dim : None):
        '''Prediction heads for different modalities'''
        
        if module_name == 'encoder':
            return nn.Sequential(   
                                 nn.LayerNorm(self.patch_size * self.patch_size * self.in_chans),
                                 nn.Linear(self.patch_size * self.patch_size * self.in_chans, self.encoder_embed_dim) # embed_dim = token embedding
                                )

        elif module_name == 'decoder':
            return nn.Linear(in_dim, self.decoder_embed_dim, bias=True)
        
        elif module_name == 'predictor':
            return nn.Linear(self.encoder_embed_dim, 5 * self.predictor_embed_dim, bias=True)  #TODO: CHECK!!

        else:
            raise ModuleNotFoundError('The given module does not exist! or the configs are not correct')
    
    def get_positional_encoder (self, embed_dim):
        return PositionalEncoding(d_model = embed_dim ,max_len = self.max_len)
        
        # if module_name == 'encoder': 
        #     return  PositionalEncoding(d_model = self.encoder_embed_dim ,max_len = self.max_len)
        
        # elif module_name == 'decoder':
        #     return  PositionalEncoding(d_model = self.decoder_embed_dim ,max_len = self.max_len)
        
        # elif module_name == 'predictor':
        #     return PositionalEncoding(d_model = 5 * self.predictor_embed_dim ,max_len = self.max_len) #TODO: CHECK!!
        
        # else:
        #     raise ModuleNotFoundError('The given module does not exist! or the configs are not correct')

    def get_transformer_blocks (self, embed_dim, depth):
        
        # if module_name == 'encoder':
        #     depth = self.encoder_depth
        #     embed_dim = self.encoder_embed_dim
        # elif module_name == 'decoder':
        #     depth = self.decoder_depth
        #     embed_dim = self.decoder_embed_dim
        # else:
        #     depth = self.predictor_depth       
        #     embed_dim = self.predictor_embed_dim
        
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
        return nn.LayerNorm(embed_dim)
        # if module_name == 'encoder':
        #     return nn.LayerNorm(self.encoder_embed_dim)

        # # elif module_name == 'decoder' and not self.use_predictor:
        # elif module_name == 'decoder':
        #     return nn.LayerNorm(self.decoder_embed_dim)
        
        # elif module_name == 'predictor':
        #     return nn.LayerNorm(self.predictor_embed_dim)
        
        # else:
        #     raise ModuleNotFoundError('The given module does not exist! or the configs are not correct')

    def initialize_weights(self):
        """Initialize module parameters with transformer-friendly defaults.

        Rules:
        - Linear weights: Xavier uniform, biases to zero
        - LayerNorm: weights to 1, biases to zero
        - Embedding: normal with std=0.02
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

        # If BBoxEncoder has learnable positional encodings, ensure reasonable init
        if hasattr(self, "bbox_encoder") and hasattr(self.bbox_encoder, "bbox_pos_encoding"):
            nn.init.normal_(self.bbox_encoder.bbox_pos_encoding, mean=0.0, std=0.02)
    
    @abstractmethod
    def forward (self):
        pass
