"""
Base Transformer class for all transformer-based architectures. All other transformer-based architectures should inherit from this class.
"""

import torch
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
            # Efficient CNN-based projection for object centric encoder
            # Much more efficient than MLP: 128x128x3 → 8x8x512 → 512
            
            # mlp_in = nn.Sequential(
            #                         # Efficient downsampling with CNN layers
            #                         nn.Conv2d(self.in_chans, 32, kernel_size=4, stride=4, padding=0),  # 128x128x3 → 32x32x32
            #                         nn.BatchNorm2d(32),
            #                         nn.GELU(),
            #                         nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0),  # 32x32x32 → 16x16x64
            #                         nn.BatchNorm2d(64),
            #                         nn.GELU(),
            #                         nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=0),  # 16x16x64 → 8x8x128
            #                         nn.BatchNorm2d(128),
            #                         nn.GELU(),
            #                         nn.Conv2d(128, self.encoder_embed_dim, kernel_size=1, stride=1, padding=0),  # 8x8x128 → 8x8x512
            #                         nn.BatchNorm2d(self.encoder_embed_dim),
            #                         nn.GELU(),
            #                         # Global average pooling to get 512-dim vector
            #                         nn.AdaptiveAvgPool2d(1),  # 8x8x512 → 1x1x512
            #                         nn.Flatten(),  # 1x1x512 → 512
            #                         nn.LayerNorm(self.encoder_embed_dim),
            #                     )
            

            # in_dim = self.image_height * self.image_width * self.in_chans  # 64*64*3 = 12288
            
            # mlp_in = nn.Linear(in_dim, self.encoder_embed_dim, bias=True) - Tried here. Works but a bit waek recons. 05_OC_AE_XL_64_Linear
            # Current simple approach (comment out to use advanced strategies)
            # mlp_in = nn.Sequential( 
            #                         nn.Linear(in_dim, in_dim//2, bias=True),
            #                         nn.GELU(),
            #                         nn.Linear(in_dim//2, self.encoder_embed_dim, bias=True),
            #                         nn.LayerNorm(self.encoder_embed_dim),
            #                     )
            
            # Use progressive compression with multiple stages - Tried on cuda4. Not much better than simple approach.
            # hidden_dim1 = in_dim // 4      # 3072
            # hidden_dim2 = in_dim // 8      # 1536
            # hidden_dim3 = in_dim // 16     # 768
            
            # mlp_in = nn.Sequential(
            #     # Stage 1: Initial compression with residual-like structure
            #     nn.Linear(in_dim, hidden_dim1, bias=True),
            #     nn.LayerNorm(hidden_dim1),
            #     nn.GELU(),
            #     nn.Dropout(0.1),
                
            #     # Stage 2: Further compression
            #     nn.Linear(hidden_dim1, hidden_dim2, bias=True),
            #     nn.LayerNorm(hidden_dim2),
            #     nn.GELU(),
            #     nn.Dropout(0.1),
                
            #     # Stage 3: Intermediate representation
            #     nn.Linear(hidden_dim2, hidden_dim3, bias=True),
            #     nn.LayerNorm(hidden_dim3),
            #     nn.GELU(),
            #     nn.Dropout(0.1),
                
            #     # Stage 4: Final projection to embedding dimension
            #     nn.Linear(hidden_dim3, self.encoder_embed_dim, bias=True),
            #     nn.LayerNorm(self.encoder_embed_dim),
            # )
            
            
            in_dim = self.image_height * self.image_width * self.in_chans

            # Advanced Strategy 1: Residual MLP with Skip Connections
            # This creates multiple compression pathways with residual connections
            class ResidualMLP(nn.Module):
                def __init__(self, in_dim, out_dim):
                    super().__init__()
                    # Multiple parallel pathways with different compression ratios
                    self.pathway1_dims = [in_dim, in_dim//2, in_dim//4, out_dim//2]  # Aggressive path
                    self.pathway2_dims = [in_dim, in_dim//3, out_dim//2]             # Moderate path
                    
                    # Pathway 1: Aggressive compression
                    self.path1 = nn.ModuleList([
                        nn.Sequential(
                            nn.Linear(self.pathway1_dims[i], self.pathway1_dims[i+1]),
                            nn.LayerNorm(self.pathway1_dims[i+1]),
                            nn.GELU(),
                            nn.Dropout(0.1)
                        ) for i in range(len(self.pathway1_dims)-1)
                    ])
                    
                    # Pathway 2: Moderate compression
                    self.path2 = nn.ModuleList([
                        nn.Sequential(
                            nn.Linear(self.pathway2_dims[i], self.pathway2_dims[i+1]),
                            nn.LayerNorm(self.pathway2_dims[i+1]),
                            nn.GELU(),
                            nn.Dropout(0.1)
                        ) for i in range(len(self.pathway2_dims)-1)
                    ])
                    
                    # Fusion layer
                    self.fusion = nn.Sequential(
                        nn.Linear(out_dim, out_dim),
                        nn.LayerNorm(out_dim),
                        nn.GELU()
                    )
                    
                def forward(self, x):
                    # Pathway 1
                    x1 = x
                    for layer in self.path1:
                        x1 = layer(x1)
                    
                    # Pathway 2  
                    x2 = x
                    for layer in self.path2:
                        x2 = layer(x2)
                    
                    # Combine pathways
                    combined = torch.cat([x1, x2], dim=-1)
                    return self.fusion(combined)
            
            # Choose between strategies:
            # Strategy 1: Residual MLP (current)
            mlp_in = ResidualMLP(in_dim, self.encoder_embed_dim)
            
            # Strategy 3: Information-Preserving Bottleneck (alternative)
            # Uncomment below and comment above to use this strategy
            """
            class InfoPreservingBottleneck(nn.Module):
                def __init__(self, in_dim, out_dim):
                    super().__init__()
                    
                    # Create multiple compression ratios
                    compress_ratios = [2, 4, 8, 16]  # Different compression levels
                    self.compressors = nn.ModuleList()
                    
                    for ratio in compress_ratios:
                        intermediate_dim = max(in_dim // ratio, out_dim)
                        compressor = nn.Sequential(
                            nn.Linear(in_dim, intermediate_dim),
                            nn.LayerNorm(intermediate_dim),
                            nn.GELU(),
                            nn.Dropout(0.05),
                            nn.Linear(intermediate_dim, out_dim // len(compress_ratios))
                        )
                        self.compressors.append(compressor)
                    
                    # Information fusion with learnable weights
                    self.fusion_weights = nn.Parameter(torch.ones(len(compress_ratios)) / len(compress_ratios))
                    self.final_norm = nn.LayerNorm(out_dim)
                    
                def forward(self, x):
                    compressed_features = []
                    for compressor in self.compressors:
                        compressed_features.append(compressor(x))
                    
                    # Weighted combination
                    stacked = torch.stack(compressed_features, dim=0)  # [num_ratios, B, T, N, out_dim//num_ratios]
                    weights = torch.softmax(self.fusion_weights, dim=0).view(-1, 1, 1, 1, 1)
                    weighted = stacked * weights
                    combined = torch.sum(weighted, dim=0)  # [B, T, N, out_dim//num_ratios * num_ratios]
                    
                    # Concatenate all compressed features
                    final_features = torch.cat(compressed_features, dim=-1)
                    return self.final_norm(final_features)
            
            # mlp_in = InfoPreservingBottleneck(in_dim, self.encoder_embed_dim)
            """

            return mlp_in
                
        
        elif module_name == 'holistic_decoder':
            
            # Input and output projection for decoder. Decoder input always encoder embedding size, predictor should also handle this
            
            mlp_in = nn.Linear(self.encoder_embed_dim, self.decoder_embed_dim, bias=True)
            mlp_out = nn.Linear(self.decoder_embed_dim, self.patch_size**2 * self.out_chans, bias=True) 
            return mlp_in, mlp_out   
        
        elif module_name == 'oc_decoder':
            
            # MLP for decoder input and output reconstruction head with upsampling
            
            mlp_in = nn.Linear(self.encoder_embed_dim, self.decoder_embed_dim, bias=True)
            # mlp_out = nn.Linear(self.decoder_embed_dim, self.image_height * self.image_width * self.out_chans, bias=True)
            
            # Current simple approach (comment out to use advanced strategies) - Tried here. Works but a bit waek recons. 05_OC_AE_XL_64_Linear
            # mlp_out= nn.Sequential(
            #             nn.Linear(self.decoder_embed_dim, self.decoder_embed_dim * 2),
            #             nn.ReLU(),
            #             nn.Linear(self.decoder_embed_dim * 2, self.image_height * self.image_width * self.out_chans)
            #             # nn.Sigmoid()
            #         )
            
            # # Progressive expansion - mirror the encoder compression - Tried on cuda4. Not much better than simple approach.
            # hidden_dim1 = out_dim // 16    # 768  - start from small
            # hidden_dim2 = out_dim // 8     # 1536
            # hidden_dim3 = out_dim // 4     # 3072
                        # mlp_out = nn.Sequential(
            #     # Stage 1: Initial expansion from decoder embedding
            #     nn.Linear(self.decoder_embed_dim, hidden_dim1, bias=True),
            #     nn.LayerNorm(hidden_dim1),
            #     nn.GELU(),
            #     nn.Dropout(0.1),
                
            #     # Stage 2: Progressive expansion
            #     nn.Linear(hidden_dim1, hidden_dim2, bias=True),
            #     nn.LayerNorm(hidden_dim2),
            #     nn.GELU(),
            #     nn.Dropout(0.1),
                
            #     # Stage 3: Further expansion
            #     nn.Linear(hidden_dim2, hidden_dim3, bias=True),
            #     nn.LayerNorm(hidden_dim3),
            #     nn.GELU(),
            #     nn.Dropout(0.1),
                
            #     # Stage 4: Final projection to output dimension
            #     nn.Linear(hidden_dim3, out_dim, bias=True),
            #     # Use Tanh for bounded output in [-1, 1] range
            #     nn.Tanh()
            # )
            
            
            out_dim = self.image_height * self.image_width * self.out_chans  # 64*64*3 = 12288
            
            
            # Advanced Strategy 2: Multi-Scale Decoder with Attention Pooling
            class MultiScaleDecoder(nn.Module):
                def __init__(self, in_dim, out_dim):
                    super().__init__()
                    
                    # Multi-scale expansion pathways
                    mid_dim1 = in_dim * 2      # 768
                    mid_dim2 = in_dim * 4      # 1536  
                    mid_dim3 = in_dim * 8      # 3072
                    
                    # Pathway 1: Fine-grained details (slow expansion)
                    self.detail_path = nn.Sequential(
                        nn.Linear(in_dim, mid_dim1),
                        nn.LayerNorm(mid_dim1),
                        nn.GELU(),
                        nn.Dropout(0.1),
                        nn.Linear(mid_dim1, mid_dim2),
                        nn.LayerNorm(mid_dim2),
                        nn.GELU(),
                        nn.Dropout(0.1),
                        nn.Linear(mid_dim2, out_dim//2)
                    )
                    
                    # Pathway 2: Coarse structure (fast expansion)  
                    self.structure_path = nn.Sequential(
                        nn.Linear(in_dim, mid_dim3),
                        nn.LayerNorm(mid_dim3),
                        nn.GELU(),
                        nn.Dropout(0.1),
                        nn.Linear(mid_dim3, out_dim//2)
                    )
                    
                    # Attention-based fusion
                    self.attention = nn.MultiheadAttention(
                        embed_dim=out_dim//2, 
                        num_heads=8, 
                        batch_first=True
                    )
                    
                    # Final reconstruction layer
                    self.final_layer = nn.Sequential(
                        nn.Linear(out_dim, out_dim),
                        nn.LayerNorm(out_dim),
                        nn.Tanh()  # Bounded output
                    )
                    
                def forward(self, x):
                    # Multi-pathway processing
                    detail_features = self.detail_path(x)      # Fine details
                    structure_features = self.structure_path(x) # Coarse structure
                    
                    # Attention-based fusion
                    # Reshape for attention: [B*T*N, 1, D//2]
                    detail_reshaped = detail_features.unsqueeze(-2)
                    structure_reshaped = structure_features.unsqueeze(-2)
                    
                    # Cross-attention between detail and structure
                    fused_detail, _ = self.attention(detail_reshaped, structure_reshaped, structure_reshaped)
                    fused_structure, _ = self.attention(structure_reshaped, detail_reshaped, detail_reshaped)
                    
                    # Combine and squeeze back
                    fused_detail = fused_detail.squeeze(-2)
                    fused_structure = fused_structure.squeeze(-2)
                    
                    # Concatenate pathways
                    combined = torch.cat([fused_detail, fused_structure], dim=-1)
                    
                    # Final reconstruction
                    return self.final_layer(combined)
            
            # Choose between strategies:
            # Strategy 2: Multi-Scale Decoder (current)
            mlp_out = MultiScaleDecoder(self.decoder_embed_dim, out_dim)
            
            # Strategy 4: Hierarchical Reconstruction (alternative)
            # Uncomment below and comment above to use this strategy
            """
            class HierarchicalReconstructor(nn.Module):
                def __init__(self, in_dim, out_dim):
                    super().__init__()
                    
                    # Multi-resolution reconstruction
                    # Low-res: 8x8, Mid-res: 16x16, High-res: 32x32, Full-res: 64x64
                    self.low_res_head = nn.Sequential(
                        nn.Linear(in_dim, 8*8*3),
                        nn.LayerNorm(8*8*3),
                        nn.GELU()
                    )
                    
                    self.mid_res_head = nn.Sequential(
                        nn.Linear(in_dim + 8*8*3, 16*16*3),  # Include low-res info
                        nn.LayerNorm(16*16*3),
                        nn.GELU()
                    )
                    
                    self.high_res_head = nn.Sequential(
                        nn.Linear(in_dim + 16*16*3, 32*32*3),  # Include mid-res info
                        nn.LayerNorm(32*32*3),
                        nn.GELU()
                    )
                    
                    self.full_res_head = nn.Sequential(
                        nn.Linear(in_dim + 32*32*3, out_dim),  # Include high-res info
                        nn.Tanh()
                    )
                    
                def forward(self, x):
                    # Progressive reconstruction from coarse to fine
                    low_res = self.low_res_head(x)
                    
                    # Concatenate previous resolution with input
                    mid_input = torch.cat([x, low_res], dim=-1)
                    mid_res = self.mid_res_head(mid_input)
                    
                    high_input = torch.cat([x, mid_res], dim=-1)
                    high_res = self.high_res_head(high_input)
                    
                    full_input = torch.cat([x, high_res], dim=-1)
                    full_res = self.full_res_head(full_input)
                    
                    return full_res
            
            # mlp_out = HierarchicalReconstructor(self.decoder_embed_dim, out_dim)
            """
            # CNN-based decoder for efficient image reconstruction
            # Start from 1x1 feature maps and upsample to full image
            # mlp_out = nn.Sequential(
            #                         # Reshape to spatial dimensions for CNN processing
            #                         nn.Linear(self.decoder_embed_dim, 8 * 8 * 32),  # e.g., 384 → 8x8x32
            #                         nn.Unflatten(1, (32, 8, 8)),  # Reshape to [B, 32, 8, 8]
            #                         # Upsampling path: 8x8 → 16x16 → 32x32 → 64x64 → 128x128
            #                         nn.Upsample(scale_factor=2, mode="nearest"),  # 8x8 → 16x16
            #                         nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            #                         nn.ReLU(),
            #                         nn.Upsample(scale_factor=2, mode="nearest"),  # 16x16 → 32x32
            #                         nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            #                         nn.ReLU(),
            #                         nn.Upsample(scale_factor=2, mode="nearest"),  # 32x32 → 64x64
            #                         nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1),
            #                         nn.ReLU(),
            #                         nn.Upsample(scale_factor=2, mode="nearest"),  # 64x64 → 128x128
            #                         # Final projection to RGB channels
            #                         nn.Conv2d(4, self.out_chans, kernel_size=3, stride=1, padding=1),  # 128x128x4 → 128x128x3
            #                         nn.Tanh()  # Output in [-1, 1] range
            #                     )
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
