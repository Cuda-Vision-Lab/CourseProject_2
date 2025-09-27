'''
Transformer Predictor for Holistic Scene Representation
'''

from base.baseTransformer import baseTransformer
from CONFIG import config


class HolisticTransformerPredictor(baseTransformer):
    
    """ 
        Foward pass through the Holistic transformer predictor module to predic the subsequent embeddings

        Args:
        -----
        encoder_history: torch Tensor
            Input embeddings from the previous time steps. Shape is (B, num_preds, num_patch_tokens, D) e.g., [B, 5, 64, 512]

        Returns:
        --------
        output: torch Tensor
            Predictor embeddings. Shape is (B, num_frames, num_objects, D). Returns exactly the same shape as the input. 
            In Wrapping Module, we only keep the last time-step, i.e., (B, -1, num_objects, D).
    """
    
    def __init__(self):
        
        super().__init__(config=config)
        
        # Predictor input and output projections
        self.mlp_in, self.mlp_out = self.get_projection('predictor')
        
        # Positional encoder for sequence modeling
        self.pe = self.get_positional_encoder(embed_dim=self.predictor_embed_dim)
                
        # Transformer blocks
        self.transformer_encoders = self.get_transformer_blocks(
                                                                embed_dim=self.predictor_embed_dim, 
                                                                depth=self.predictor_depth
                                                                )
        
        # Layer Normalization
        self.predictor_norm = self.get_ln(self.predictor_embed_dim)
        
        # Initialize weights
        self.initialize_weights()

        return



