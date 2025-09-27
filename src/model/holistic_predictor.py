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
        encoder_features: torch Tensor
            Input embeddings from the previous time steps. Shape is (B, num_preds, num_patch_tokens, D) e.g., [B, 5, 64, 512]

        Returns:
        --------
        output: torch Tensor
            Predictor embeddings. Shape is (B, num_preds, num_patch_tokens, D). Returns exactly the same shape as the input. 
            In Wrapping Module, we only keep the last time-step, i.e., (B, -1, num_patch_tokens, D). e.g., [B, -1, 64, 512]
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



    def forward(self, encoder_history):
        """
        Forward pass through the transformer predictor module
        """
        B, num_preds, num_patch_tokens, D = encoder_history.shape #[B, 5, 64, 512]
        
        # Map embeddings to token space
        token_input = self.mlp_in(encoder_history)  # (B, num_preds, num_patch_tokens, predictor_embed_dim)
        
        # Apply positional encoding
        token_input = self.pe(token_input)
        
        # Feed through transformer encoder blocks
        pred_tokens = self.transformer_encoders(token_input) # (B, num_preds, num_patch_tokens, predictor_embed_dim)
        
        pred_tokens = self.predictor_norm(pred_tokens)
        
        # Map back to embedding space
        output = self.mlp_out(pred_tokens)  # (B, num_preds, num_patch_tokens, encoder_embed_dim)
        
        if self.residual:
            output = output + encoder_history
            
        return output