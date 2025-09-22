import numpy as np
import torch.nn as nn
import torch
from .model_utils import MaskEncoder, BBoxEncoder
from base.baseTransformer import baseTransformer
from CONFIG import config

class PredictorWrapper(nn.Module):
    
    
    def __init__(self, exp_params, predictor):
        """
        Module initializer
        """
        super().__init__()
        self.predictor = predictor
        self.num_preds = config['vit_cfg']['num_preds']
        self.predictor_window_size = config['vit_cfg']['predictor_window_size']
        
    def _update_buffer_size(self, inputs):
        """
        Updating the inputs of a transformer model given the 'buffer_size'.
        We keep a moving window over the input tokens, dropping the oldest slots if the buffer
        size is exceeded.
        """
        num_inputs = inputs.shape[1]
        if num_inputs > self.input_buffer_size:
            extra_inputs = num_inputs - self.input_buffer_size
            inputs = inputs[:, extra_inputs:]
        return inputs
    
    def forward_transformer(self, slot_history):
        """
        Forward pass through any Transformer-based predictor module

        Args:
        -----
        slot_history: torch Tensor
            Decomposed slots form the seed and predicted images.
            Shape is (B, num_frames, num_slots, slot_dim)

        Returns:
        --------
        pred_slots: torch Tensor
            Predicted subsequent slots. Shape is (B, num_preds, num_slots, slot_dim)
        """
        first_slot_idx = 1 if self.skip_first_slot else 0
        predictor_input = slot_history[:, first_slot_idx:self.num_context].clone()  # inial token buffer

        pred_slots = []
        for t in range(self.num_preds):
            cur_preds = self.predictor(predictor_input)[:, -1]  # get predicted slots from step
            next_input = slot_history[:, self.num_context+t] if self.teacher_force else cur_preds
            predictor_input = torch.cat([predictor_input, next_input.unsqueeze(1)], dim=1)
            predictor_input = self._update_buffer_size(predictor_input)
            pred_slots.append(cur_preds)
        pred_slots = torch.stack(pred_slots, dim=1)  # (B, num_preds, num_slots, slot_dim)
        return pred_slots
        
     


class VanillaTransformerPredictor(nn.Module):
    """
    Vanilla Transformer Predictor module.
    It performs self-attention over all slots in the input buffer, jointly modelling
    the relational and temporal dimensions.

    Args:
    -----
    num_slots: int
        Number of slots per image. Number of inputs to Transformer is num_slots * num_imgs
    slot_dim: int
        Dimensionality of the input slots
    num_imgs: int
        Number of images to jointly process. Number of inputs to Transformer is num_slots * num_imgs
    token_dim: int
        Input slots are mapped to this dimensionality via a fully-connected layer
    hidden_dim: int
        Hidden dimension of the MLPs in the transformer blocks
    num_layers: int
        Number of transformer blocks to sequentially apply
    n_heads: int
        Number of attention heads in multi-head self attention
    residual: bool
        If True, a residual connection bridges across the predictor module
    input_buffer_size: int
        Maximum number of consecutive time steps that the transformer receives as input
    """

    def __init__(self, num_slots, slot_dim, num_imgs, token_dim=128, hidden_dim=256,
                 num_layers=2, n_heads=4, residual=False, input_buffer_size=5):
        """
        Module initializer
        """
        super().__init__()
        self.num_slots = num_slots
        self.num_imgs = num_imgs
        self.slot_dim = slot_dim
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.nhead = n_heads
        self.residual = residual
        self.input_buffer_size = input_buffer_size
        print_("Instanciating Vanilla Transformer Predictor:")
        print_(f"  --> num_layers: {self.num_layers}")
        print_(f"  --> input_dim: {self.slot_dim}")
        print_(f"  --> token_dim: {self.token_dim}")
        print_(f"  --> hidden_dim: {self.hidden_dim}")
        print_(f"  --> num_heads: {self.nhead}")
        print_(f"  --> residual: {self.residual}")
        print_("  --> batch_first: True")
        print_("  --> norm_first: True")
        print_(f"  --> input_buffer_size: {self.input_buffer_size}")

        # MLPs to map slot-dim into token-dim and back
        self.mlp_in = nn.Linear(self.encoder_embed_dim, token_dim)
        self.mlp_out = nn.Linear(token_dim, self.predictor_embed_dim)

        # embed_dim is split across num_heads, i.e., each head will have dimension embed_dim // num_heads)
        self.transformer_encoders = self.predictor_blocks
        # nn.Sequential(
        #     *[torch.nn.TransformerEncoderLayer(
        #             d_model=token_dim,
        #             nhead=self.nhead,
        #             batch_first=True,
        #             norm_first=True,
        #             dim_feedforward=hidden_dim
        #         ) for _ in range(num_layers)]
        #     )

        # Custom temrpoal encoding. All slots from the same time step share the encoding
        self.pe = PositionalEncoding(d_model=self.token_dim, max_len=input_buffer_size)
        return

    def forward(self, inputs):
        """
        Foward pass through the transformer predictor module to predic the subsequent object slots

        Args:
        -----
        inputs: torch Tensor
            Input object slots from the previous time steps. Shape is (B, num_imgs, num_slots, slot_dim)

        Returns:
        --------
        output: torch Tensor
            Predictor object slots. Shape is (B, num_imgs, num_slots, slot_dim), but we only care about
            the last time-step, i.e., (B, -1, num_slots, slot_dim).
        """
        B, num_imgs, num_patches, _ = inputs.shape

        # mapping slots to tokens, and applying temporal positional encoding
        token_input = self.mlp_in(inputs)
        time_encoded_input = self.pe(
                                    x=token_input,
                                    batch_size=B,
                                    num_slots=num_patches
                                    )

        # feeding through transformer encoder blocks
        token_output = time_encoded_input.reshape(B, num_imgs * num_patches, self.token_dim)
        for encoder in self.transformer_encoders:
            token_output = encoder(token_output)
        token_output = token_output.reshape(B, num_imgs, num_patches, self.token_dim)

        # mapping back to slot dimensionality
        output = self.mlp_out(token_output)
        output = output + inputs if self.residual else output
        return output



class Predictor(baseTransformer):
    
    """ 
    Vision Transformer for image reconstruction task
    """
    def __init__(self):
        
        module_name = 'predictor'
        
        super().__init__(config=config, module_name=module_name)
          
        self.pe = self.get_positional_encoder(module_name)
        
        self.mlp_in = nn.Linear(self.encoder_embed_dim, token_dim)
        self.mlp_out = nn.Linear(token_dim, self.predictor_embed_dim)
        
        self.transformer_encoders = self.transformer_encoder(module_name)
        
        
        # Initialize weights
        self.initialize_weights()

        return

    def forward_loss(self, imgs, pred, mask):
        """
        Compute reconstruction loss .
        """
        pass
    #     return loss

    
def forward(self, inputs):
        """
        Foward pass through the transformer predictor module to predic the subsequent object slots

        Args:
        -----
        inputs: torch Tensor
            Input object slots from the previous time steps. Shape is (B, num_imgs, num_slots, slot_dim)

        Returns:
        --------
        output: torch Tensor
            Predictor object slots. Shape is (B, num_imgs, num_slots, slot_dim), but we only care about
            the last time-step, i.e., (B, -1, num_slots, slot_dim).
        """
        B, num_imgs, num_patches, _ = inputs.shape

        # mapping slots to tokens, and applying temporal positional encoding
        token_input = self.mlp_in(inputs)
        time_encoded_input = self.pe(
                                    x=token_input,
                                    batch_size=B,
                                    num_slots=num_patches
                                    )

        # feeding through transformer encoder blocks
        token_output = time_encoded_input.reshape(B, num_imgs * num_patches, self.token_dim)
        for encoder in self.transformer_encoders:
            token_output = encoder(token_output)
        token_output = token_output.reshape(B, num_imgs, num_patches, self.token_dim)

        # mapping back to slot dimensionality
        output = self.mlp_out(token_output)
        output = output + inputs if self.residual else output
        return output
