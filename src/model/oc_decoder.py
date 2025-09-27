"""
Object Centric Scene Representation Decoder
"""

import torch.nn as nn
import torch
from base.baseTransformer import baseTransformer
from CONFIG import config

class ObjectCentricDecoder(baseTransformer):
        
    def __init__(self):

        
        super().__init__(config=config)

        self.decoder_projection = self.get_projection('decoder')
        self.decoder_pos_embed = self.get_positional_encoder(self.decoder_embed_dim)
        self.decoder_blocks = self.get_transformer_blocks(self.decoder_embed_dim, self.decoder_depth)
        self.decoder_norm = self.get_ln(self.decoder_embed_dim)

        self.decoder_pred_image = nn.Linear(self.decoder_embed_dim, self.image_height * self.image_width * self.out_chans, bias=True)

        # Initialize weights
        self.initialize_weights()
        
        return

    def combine_objects_to_scene(self, object_frames):
        """
        Combine object frames back to full scene by summing them.
        
        Args:
            object_frames: [B, T, num_objects, C, H, W]
            
        Returns:
            scene: [B, T, C, H, W]
        """
        # Simple approach: sum all object frames (assumes objects don't overlap significantly)
        scene = torch.sum(object_frames, dim=2)  # [B, T, C, H, W]
        
        # Clamp to valid pixel range
        scene = torch.clamp(scene, 0.0, 1.0)
        
        return scene
    

    def forward_loss(self, target_patches, pred_patches):
        """
        Compute reconstruction loss for different modalities.
        
        Args:
            targets: [B, T, N, patch_dim] 
            pred: [B, T, N, patch_dim] - predicted patches
        """
        loss_fn = nn.MSELoss()
        loss = loss_fn(target_patches, pred_patches)
        
        return loss
    
 
    def forward(self, encoded_features, target=None):
        """
        Forward pass through decoder.     
        
        Args:
            encoded_features: [B, T, num_objects, embed_dim] - encoded features from encoder
            target: target images for loss computation
        
        Returns:
            predictions: Reconstructed images
            loss: Loss value
        """
        B, T, Num_objects, D = encoded_features.shape       

        # Project to decoder dimension
        x = self.decoder_projection(encoded_features)  # [B, T, Num_objects, decoder_embed_dim]
                
        # Add positional encoding
        x = self.decoder_pos_embed(x)
        
        # Apply decoder blocks
        for i, block in enumerate(self.decoder_blocks):
            x = block(x)
        
        # Final layer norm 
        x = self.decoder_norm(x)
        
        # Project to frame predictions
        pred_frames = self.decoder_pred_image(x)  # [B, T, Num_objects, H * W * out_chans]
        
        pred_frames = pred_frames.view(B, T, Num_objects, self.out_chans, self.image_height, self.image_width) # [B, T, Num_objects, C, H, W ]
        
        # Combine objects to reconstruct scene
        scene_recons = self.combine_objects_to_scene(pred_frames)  # [B, T, C, H, W]
        
        # Calculate the loss
        loss = None
        if target is not None:
            loss = self.forward_loss(target, scene_recons)
            
        return scene_recons, loss