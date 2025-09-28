"""
Base Transformer class for all transformer-based architectures. All other transformer-based architectures should inherit from this class.
"""

import torch.nn as nn
from model.model_utils import TransformerBlock, Patchifier, PositionalEncoding
from abc import ABC, abstractmethod 
from utils.logger import log_function

class baseTransformer(nn.Module, ABC):
    
    '''
    The base class to encapsulate the mutual functionalities of a transformer-based architecture
    '''
    
    def __init__(self, config) -> None:
        """
        Initialize the base transformer class. Global configurations are loaded here.
        """
        super().__init__()
        
        self.patch_size = config['data']['patch_size']
        self.max_objects =  config['data']['max_objects']
        self.encoder_embed_dim = config['vit_cfg']['encoder_embed_dim'] # encoder output
        self.decoder_embed_dim = config['vit_cfg']['decoder_embed_dim'] # decoder input
        self.max_len = config['vit_cfg']['max_len']
        self.out_chans = self.in_chans =  config['vit_cfg']['in_out_channels']
        self.use_masks =  config['vit_cfg']['use_masks']
        self.use_bboxes = config['vit_cfg']['use_bboxes']
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
        
        """
        Projection heads for different modalities
        
        Args:
            module_name: str
                The name of the module to get the projection for
                
        Returns:
            nn.Sequential: The projection for the given module
        """
        
        if module_name == 'holistic_encoder':
            mlp_in = nn.Sequential(   
                                 nn.LayerNorm(self.patch_size * self.patch_size * self.in_chans),
                                 nn.Linear(self.patch_size * self.patch_size * self.in_chans, self.encoder_embed_dim) # embed_dim = token embedding
                                )
            return mlp_in
        
        elif module_name == 'oc_encoder':
            # Sequential projection for object centric encoder to reduce the dimension of the input images
            
            input_dim = self.image_height * self.image_width * self.in_chans  # 49,152

            intermediate_dim1 = input_dim // 4   # 49,152 → 12,288
            intermediate_dim2 = input_dim // 8   # 49,152 → 6,144
            intermediate_dim3 = input_dim // 16  # 49,152 → 3,072
            
            mlp_in = nn.Sequential(
                # First compression stage
                nn.Linear(input_dim, intermediate_dim1),  # 49,152 → 12,288
                nn.LayerNorm(intermediate_dim1),
                nn.GELU(),
                nn.Dropout(0.1),
                
                # Second compression stage  
                nn.Linear(intermediate_dim1, intermediate_dim2),  # 12,288 → 6,144
                nn.LayerNorm(intermediate_dim2),
                nn.GELU(),
                nn.Dropout(0.1),
                
                # Third compression stage
                nn.Linear(intermediate_dim2, intermediate_dim3),  # 6,144 → 3,072
                nn.LayerNorm(intermediate_dim3),
                nn.GELU(),
                nn.Dropout(0.1),
                
                # Final projection to embedding dimension
                nn.Linear(intermediate_dim3, self.encoder_embed_dim),  # 3,072 → 512
                nn.LayerNorm(self.encoder_embed_dim),
            )
            return mlp_in
                
        
        elif module_name == 'holistic_decoder':
            
            # Input and output projection for decoder. Decoder input always encoder embedding size, predictor should also handle this
            
            mlp_in = nn.Linear(self.encoder_embed_dim, self.decoder_embed_dim, bias=True)
            mlp_out = nn.Linear(self.decoder_embed_dim, self.patch_size**2 * self.out_chans, bias=True) 
            return mlp_in, mlp_out   
        
        elif module_name == 'oc_decoder':
            
            # MLP for decoder input and output reconstruction head with upsampling
            
            mlp_in = nn.Linear(self.encoder_embed_dim, self.decoder_embed_dim, bias=True)
            mlp_out = nn.Sequential(
                                    nn.Linear(self.decoder_embed_dim, self.decoder_embed_dim * 2),
                                    nn.LayerNorm(self.decoder_embed_dim * 2),
                                    nn.GELU(),
                                    nn.Dropout(0.1),
                                    
                                    nn.Linear(self.decoder_embed_dim * 2, self.decoder_embed_dim * 4),
                                    nn.LayerNorm(self.decoder_embed_dim * 4),
                                    nn.GELU(),
                                    nn.Dropout(0.1),
                                    
                                    nn.Linear(self.decoder_embed_dim * 4, self.image_height * self.image_width * self.out_chans),
                                )
            return mlp_in, mlp_out
        
        elif module_name == 'predictor': 
            
            # input and output projection for predictor
            
            mlp_in = nn.Linear(self.encoder_embed_dim, self.predictor_embed_dim, bias=True)
            mlp_out = nn.Linear(self.predictor_embed_dim, self.encoder_embed_dim, bias=True)
            return mlp_in, mlp_out
        
        else:
            raise ModuleNotFoundError('The given module does not exist! or the configs are not correct')
    
    def get_positional_encoder (self, embed_dim):
        '''
        Positional encoding for the transformer blocks
        '''
        return PositionalEncoding(d_model = embed_dim)
        

    def get_transformer_blocks (self, embed_dim, depth):
        
        """
        Get the transformer blocks
        """
        
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
