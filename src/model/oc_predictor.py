"""
Transformer Predictor for Object Centric Scene Representation
"""

from base.baseTransformer import baseTransformer
from CONFIG import config

class ObjectCentricTransformerPredictor(baseTransformer):
    
    """ 
        Foward pass through the Object Centric transformer predictor module to predict the subsequent embeddings

        Args:
        -----
        encoder_features: torch Tensor
            Input embeddings from the previous time steps. Shape is (B, num_preds, num_objects, D) e.g., [B, 5, 11, 512]

        Returns:
        --------
        output: torch Tensor
            Predictor embeddings. Shape is (B, num_preds, num_objects, D). Returns exactly the same shape as the input. 
            In Wrapping Module, we only keep the last time-step prediction, i.e., (B, -1, num_objects, D). e.g., [B, -1, 11, 512]
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

        Args:
        -----
        encoder_features: torch Tensor
            Input embeddings from encoder for the whole sequence. Shape is (B, num_preds, num_objects, embed_dim)

        Returns:
        --------
        output: torch Tensor
            Predicted embeddings. Shape is (B, num_preds, num_objects, predictor_embed_dim)
        """
        B, num_preds, num_objects, D = encoder_history.shape   

        # Map embeddings to token space
        token_input = self.mlp_in(encoder_history)  # (B, num_preds, num_objects, predictor_embed_dim)
        
        # Apply positional encoding
        token_input = self.pe(token_input)

        # Feed through transformer encoder blocks
        pred_tokens = self.transformer_encoders(token_input) # (B, seq_len_keep, num_patch_tokens, predictor_embed_dim)
        
        pred_tokens = self.predictor_norm(pred_tokens)
        # pred_tokens = pred_tokens.reshape(B, T, N, self.predictor_embed_dim) # same as inputs

        # Map back to embedding space
        output = self.mlp_out(pred_tokens)  # (B, num_preds, num_objects, encoder_embed_dim)
        
        if self.residual:
            output = output + encoder_history
            
        return output
