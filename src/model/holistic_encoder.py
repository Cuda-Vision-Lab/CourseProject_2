"""
Holistic scene Representation Encoder
"""

from base.baseTransformer import baseTransformer
from CONFIG import config

class HolisticEncoder(baseTransformer):
    
    """ 
    Vision Transformer Encoder for holistic scene representation task

    Args:
        images (torch.Tensor): Input images of shape [B, T, C, H, W].
        masks (torch.Tensor, optional): Optional masks for input images.
        bboxes (torch.Tensor, optional): Optional bounding boxes for input images.

    Returns:
        torch.Tensor: Encoded features of the input images. Shape is [B, T, N, embed_dim].
    """
    def __init__(self):

        # self.mask_ratio = mask_ratio

        super().__init__(config=config)
        
        # Projection to transformer token dimension
        self.patch_projection = self.get_projection('holistic_encoder')
        
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

        # Feed through transformer blocks
        encoded_features = self.encoder_blocks(image_tokens)

        # Layer normalization
        encoded_features = self.encoder_norm(encoded_features)
        
        return encoded_features