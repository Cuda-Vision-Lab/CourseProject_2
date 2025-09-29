"""
Object Centric Scene Representation Encoder
"""

import torch
from base.baseTransformer import baseTransformer
from CONFIG import config

class ObjectCentricEncoder(baseTransformer):
    
    def __init__(self):


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

    def forward(self, images, masks=None, bboxes=None):
        """ 
        Forward pass
        """
        
        if self.use_masks and self.use_bboxes:
            raise ValueError("Either masks or bboxes must be provided! Not both!")
        
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
        
        # CNN-based projection
        object_frames = object_frames.view(B * T * num_frames, C, H, W)  
        cnn_features = self.patch_projection(object_frames)   # → [B*T*11, 512]
        object_tokens = cnn_features.view(B, T, num_frames, self.encoder_embed_dim) # [B, T, num_frames, 512]

        # # Reshape object frames to patches
        # object_patches = object_frames.view(B,T,num_frames,patch_size)
        
        # # # Project to transformer token dimension
        # object_tokens = self.patch_projection(object_patches)
        
        # Add positional encoding
        object_tokens = self.encoder_pos_embed(object_tokens)
        
        # Feed through transformer blocks
        object_tokens = self.encoder_blocks(object_tokens)
        
        # Layer normalization
        object_tokens = self.encoder_norm(object_tokens)

        return object_tokens # [B, T, num_objects, embed_dim]