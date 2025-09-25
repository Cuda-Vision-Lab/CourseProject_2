import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from .model_utils import MaskEncoder, BBoxEncoder, PositionalEncoding, TransformerBlock
from base.baseTransformer import baseTransformer
from CONFIG import config
import random

class PredictorWrapper(nn.Module):
    """
    Wrapper module that autoregressively applies any predictor module on a sequence of data
    
    Args:
    -----
    predictor: nn.Module
        Instantiated predictor module to wrap.
    """
    
    '''
    At each prediction step, the wrapper:

    Concatenates the newly predicted embedding to the current input buffer.
    Trims the buffer to the required size (oldest frames dropped if necessary).
    Feeds this buffer into the transformer to generate the next prediction.
    Repeats this for the number of prediction steps required.
    '''
    
    def __init__(self, predictor):
        """
        Module initializer
        """
        super().__init__(config=config)
        
        self.predictor = predictor
        self.num_preds = config['vit_cfg']['num_preds']
        self.predictor_window_size = config['vit_cfg']['predictor_window_size']
        # self.mode = mode
        
    
    def forward(self, encoder_history):
        """

        Args:
        -----
        encoder_history: torch Tensor
        the output of the encoder : Shape is (B, T, num_tokens, embed_dim) : num_tokens = N_total, no token masking here, 
        predictor receives N_total(total number of tokens in the sequence) embeddings for all tokens in the sequence
            Shape is (B, T, num_tokens, embed_dim)

        Returns:
        --------
        pred_embeds: torch Tensor
            Predicted subsequent embeddings. Shape is (B, num_preds, num_tokens, embed_dim)
        
        """
        B, T, N, D = encoder_history.shape

        # Dynamically slicing the input sequence 
        slicing_idx = random.randint(0, T - (2*self.num_preds+1))
        
        # Keep 5 consecutive tokens for prediction
        predictor_input = encoder_history[:, slicing_idx : slicing_idx+self.num_preds].clone()  # (B, num_frames, N, D)

        target = encoder_history[:, slicing_idx+self.num_preds:slicing_idx+(2*self.num_preds)].clone()  # Future frames for loss computation
        
        losses = []
        pred_embeds = []
        
        for t in range(self.num_preds):
            
            # Get prediction from current input buffer, keep only the last prediction
            cur_pred = self.predictor(predictor_input)[:, -1]# (B, N, D) -- only last prediction
            
            # Compute loss iin each time step - CHECK!! Is this correct?
            loss = self.forward_loss(target[:, t], cur_pred) 
            losses.append(loss)
            
            # Autoregressive: feed back last prediction
            predictor_input = torch.cat([predictor_input, cur_pred], dim=1)  # (B, num_frames+1, N, embed_dim)
            predictor_input = self._update_buffer_size(predictor_input)  # Shift window size (B, num_frames, N, embed_dim)
            pred_embeds.append(cur_pred) 
            
        pred_embeds = torch.stack(pred_embeds, dim=1)  # (B, num_preds, N, embed_dim)
        total_loss = torch.stack(losses).mean()  # mean over prediction steps 
        
        return pred_embeds, total_loss
    
    def forward_loss(self, target, pred):
        """
        Compute reconstruction loss .
        """
        loss = F.mse_loss(pred, target)
        return loss

    def _update_buffer_size(self, predictor_inputs):
        """
        Updating the inputs of a transformer model given the 'buffer_size'.
        We keep a moving window over the input tokens, dropping the oldest slots if the buffer
        size is exceeded.
        """
        num_inputs = predictor_inputs.shape[1]
        if num_inputs > self.predictor_window_size:
            extra_inputs = num_inputs - self.predictor_window_size
            predictor_inputs = predictor_inputs[:, extra_inputs:]
        return predictor_inputs


class TransformerPredictor(baseTransformer):
    
    """ 
        Foward pass through the transformer predictor module to predic the subsequent embeddings

        Args:
        -----
        inputs: torch Tensor
            Input embeddings from the previous time steps. Shape is (B, num_frames, num_tokens, embed_dim)

        Returns:
        --------
        output: torch Tensor
            Predictor embeddings. Shape is (B, num_frames, num_tokens, embed_dim), but we only care about
            the last time-step, i.e., (B, -1, num_tokens, embed_dim).
    
    """
    
    def __init__(self):
        
        super().__init__(config=config)
        
        self.mlp_in, self.mlp_out = self.get_projection('predictor')
        
        # Positional encoder for sequence modeling
        self.pe = self.get_positional_encoder(embed_dim=self.predictor_embed_dim)
                
        # Transformer blocks
        self.transformer_encoders = self.get_transformer_blocks(
                                                                embed_dim=self.predictor_embed_dim, 
                                                                depth=self.predictor_depth
                                                                )
        
        # Initialize weights
        self.initialize_weights()

        return


    def forward(self, inputs):
        """
        Forward pass through the transformer predictor module

        Args:
        -----
        inputs: torch Tensor
            Input embeddings from encoder for the whole sequence. Shape is (B, seq_len, num_patch_tokens, embed_dim)

        Returns:
        --------
        output: torch Tensor
            Predicted embeddings. Shape is (B, seq_len_keep, num_patch_tokens, predictor_embed_dim)
        """
        B, T, N, D = inputs.shape  # T = buffer_size 

        # Map embeddings to token space
        token_input = self.mlp_in(inputs)  # (B, seq_len_keep, num_patch_tokens, predictor_embed_dim)
        
        # Apply positional encoding
        token_input = self.pe(token_input)

        # Feed through transformer encoder blocks
        pred_tokens = self.transformer_encoders(token_input) # (B, seq_len_keep, num_patch_tokens, predictor_embed_dim)
        
        pred_tokens = pred_tokens.reshape(B, T, N, self.predictor_embed_dim) # same as inputs

        # Map back to embedding space
        output = self.mlp_out(pred_tokens)  # 
        
        if self.residual:
            output = output + inputs
            
        return output
