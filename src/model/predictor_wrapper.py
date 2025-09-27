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
        
    
    def forward(self, encoder_features):
        """

        Args:
        -----
        encoder_features: torch Tensor: Shape is [B, T, num_objects, D] e.g., [B, 24, 11, 512]
        predictor receives object-frame embeddings for frames in the sequence

        Returns:
        --------
        pred_embeds: torch Tensor
            Predicted subsequent embeddings. Shape is (B, num_preds, num_objects, D) e.g., [B, 5, 11, 512]
        
        """
        B, T, num_objects, D = encoder_features.shape

        # Dynamically slicing the input sequence 
        slicing_idx = random.randint(0, T - (2*self.num_preds+1))
        
        # Keep 5 consecutive tokens for prediction
        predictor_input = encoder_features[:, slicing_idx : slicing_idx+self.num_preds].clone()  # (B, num_preds, num_objects, D) e.g., [B, 5, 11, 512]

        target = encoder_features[:, slicing_idx+self.num_preds:slicing_idx+(2*self.num_preds)].clone()  # Future frames for loss computation
        
        losses = []
        pred_embeds = []
        
        for t in range(self.num_preds):
            
            # Get prediction from current input buffer, keep only the last prediction
            cur_pred = self.predictor(predictor_input)[:, -1]# (B, num_objects, D) -- only last prediction
            
            # Compute loss iin each time step - CHECK!! Is this correct?
            loss = self.forward_loss(target[:, t], cur_pred) 
            losses.append(loss)
            
            # Autoregressive: feed back last prediction
            predictor_input = torch.cat([predictor_input, cur_pred], dim=1)  # (B, num_frames+1, num_objects, D)
            predictor_input = self._update_buffer_size(predictor_input)  # Shift window size (B, num_frames, num_objects, D)
            pred_embeds.append(cur_pred) 
            
        pred_embeds = torch.stack(pred_embeds, dim=1)  # (B, num_preds, num_objects, D)
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

