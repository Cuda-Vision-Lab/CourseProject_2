import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR


class MultiHeadSelfAttention(nn.Module):
    """ 
    Self-Attention module

    Args:
    -----
    embed_dim: int
        Dimensionality of the tokens in the transformer
    inner_dim: int
        Dimensionality used for attention
    """

    def __init__(self, embed_dim, attn_dim, num_heads):
        """ """
        super().__init__()
        self.embed_dim = embed_dim #  Embedding size per token, here called D. N number of tokens
        self.attn_dim = attn_dim # the dimension of the attention vector
        self.num_heads = num_heads 
        assert num_heads >= 1 # multi-head attention
        assert attn_dim % num_heads == 0, f"attn_dim = {attn_dim} must be divisible by num_heads = {num_heads}..."
        self.head_dim = attn_dim // num_heads

        # query, key and value projections
        self.q = nn.Linear(embed_dim, attn_dim, bias=False) 
        self.k = nn.Linear(embed_dim, attn_dim, bias=False) 
        self.v = nn.Linear(embed_dim, attn_dim, bias=False) 

        # output projection
        self.out_proj = nn.Linear(attn_dim, embed_dim, bias=False) # back to the original input dimension
        return
    
    def attention(self, query, key, value):
        """
        Computing self-attention

        All (q,k,v).shape ~ (batch_size * seq_len * self.num_heads, num_tokens, self.head_dim)
        """
        scale = (query.shape[-1]) ** (-0.5) # smoothing gradiants to work better with softmax

        # similarity between each query and the keys
        similarity = torch.bmm(query, key.permute(0, 2, 1)) * scale  # ~(B, N, N) batch-wise matrix multiplication, permmute here acts as traspose for dimentions matching
        attention = similarity.softmax(dim=-1) # softmax across each row 
        self.attention_map = attention # for visualization \latter

        # attention * values
        output = torch.bmm(attention, value)
        return output

    def split_into_heads(self, x):  # TODO: check if this is correct
        """
        Splitting a vector into multiple heads
        """
        # print(f"Input x shape: {x.shape}")

        batch_size, seq_len, num_tokens, embed_dim = x.shape 
        # print(f'number of heads: {self.num_heads}')
        # print(f'head dim: {self.head_dim}')
        # print(f"Input x shape: {x.shape}")
        
        # Reshape to combine batch and sequence dimensions for processing
        x = x.reshape(batch_size * seq_len, num_tokens, embed_dim)  
        # print(f"Reshaped x shape: {x.shape}")
        
        # Split the token dimension into heads
        x = x.view(batch_size * seq_len, num_tokens, self.num_heads, self.head_dim)  
        # print(f"After view x shape: {x.shape}")
        
        # Permute to get heads dimension first for independent attention
        x = x.permute(0, 2, 1, 3) 
        # print(f"After permute x shape: {x.shape}")
        
        # Reshape to combine batch*seq and heads for batch processing
        y = x.reshape(batch_size * seq_len * self.num_heads, num_tokens, self.head_dim)  
        # print(f"Final y shape: {y.shape}")
        
        return y

    def merge_heads(self, x):
        """
        Rearranging heads back to original head structure
        """
        _, num_tokens, dim_head = x.shape 
        y = x.reshape(-1, self.num_heads, num_tokens, dim_head) 
        y = y.reshape(-1, num_tokens, self.num_heads * dim_head) 
        return y


    def forward(self, x):
        """ 
        Forward pass through Self-Attention module
        """
        # Store original shape to restore later
        original_shape = x.shape 
        batch_size, seq_len, num_tokens, embed_dim = original_shape
        
        # linear projections and splitting into heads:
        # (B, N, D) --> (B, N, Nh, Dh) --> (B * Nh, N, Dh)
        q, k, v = self.q(x), self.k(x), self.v(x)
        q = self.split_into_heads(q) 
        k = self.split_into_heads(k)
        v = self.split_into_heads(v)

        # applying attention equation
        vect = self.attention(query=q, key=k, value=v)
        # print(f"Vect shape: {vect.shape}")
        # rearranging heads: (B * Nh, N, Dh) --> (B*T, N, D)
        y = self.merge_heads(vect)  
        # print(f"Y SHAPE AFTER MERGE HEADS: {y.shape}")
        y = self.out_proj(y) #(B, N, embed_dim)
        # print(f"Y SHAPE AFTER OUT PROJ: {y.shape}")
        # Reshape back to original 4D shape
        y = y.reshape(batch_size, seq_len, num_tokens, embed_dim)  
        # print(f"Y SHAPE AFTER RESHAPE: {y.shape}")
        return y
    

class MLP(nn.Module):
    """
    2-Layer Multi-Layer Perceptron used in transformer blocks
    
    Args:
    -----
    in_dim: int
        Dimensionality of the input embeddings to the MLP
    hidden_dim: int
        Hidden dimensionality of the MLP
    """
    
    def __init__(self, in_dim, hidden_dim):
        """ MLP Initializer """
        super().__init__()
        self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),  ## NOTE: GELU activation function used in FCL for transformers!
                nn.Linear(hidden_dim, in_dim),
            )
        
    def forward(self, x):
        """ Forward """
        y = self.mlp(x)
        return y
    

class TransformerBlock(nn.Module):
    """
    Transformer block using self-attention

    Args:
    -----
    embed_dim: int
        Dimensionality of the input tokens
    attn_dim: int
        Inner dimensionality of the attention module. Must be divisible be num_heads
    num_heads: int
        Number of heads in the self-attention mechanism
    mlp_size: int
        Hidden dimension of the MLP module
    """

    def __init__(self, embed_dim, attn_dim, num_heads, mlp_size):
        """ Module initializer """
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp_size = mlp_size
        self.attn_dim = attn_dim
        self.num_heads = num_heads

        # MHA
        self.ln_att = nn.LayerNorm(embed_dim, eps=1e-6) # Layer normalization
        self.attn = MultiHeadSelfAttention(
                embed_dim=embed_dim,
                attn_dim=attn_dim,
                num_heads=num_heads
            ) # ---> [320,17,128]
        
        # MLP
        self.ln_mlp = nn.LayerNorm(embed_dim, eps=1e-6) # Layer normalization
        self.mlp = MLP(
                in_dim=embed_dim,
                hidden_dim=mlp_size,
            )
        return


    def forward(self, inputs):
        """
        Forward pass through transformer encoder block.
        We assume the more modern PreNorm design
        """
        # assert inputs.ndim == 3, f"Inputs to the transformer block must be of shape (B, N, D), but got {inputs.shape}"
        # print(f"INPUTS SHAPE: {inputs.shape}") 
 
        # Self-attention.
        x = self.ln_att(inputs)
        # print(f"X SHAPE BEFORE ATTENTION: {x.shape}")
        x = self.attn(x) 
        assert x.shape == inputs.shape, f"X shape: {x.shape} and inputs shape: {inputs.shape} MUST BE THE SAME (input and output of the attention block)"
        y = x + inputs # residual connection - both are now 4D 

        # MLP
        z = self.ln_mlp(y)
        z = self.mlp(z)
        z = z + y # residual connection

        return z


    def get_attention_masks(self):
        """ Fetching last computer attention masks """
        attn_masks = self.attn.attention_map
        N = attn_masks.shape[-1]
        attn_masks = attn_masks.reshape(-1, self.num_heads, N, N)
        return attn_masks



class Patchifier:
    """ 
    Module that splits an image into patches.
    We assumen square images and patches
    """

    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, img):
        """
        img: (B, seq_len, C, H, W)
        Returns: (B, seq_len, num_patches, patch_dim)
        """
        B, seq_len, C, H, W = img.shape
        assert H % self.patch_size == 0, f"H={H} not divisible by patch_size={self.patch_size}"
        assert W % self.patch_size == 0, f"W={W} not divisible by patch_size={self.patch_size}"
        num_patch_H = H // self.patch_size
        num_patch_W = W // self.patch_size

        patch_data = img.reshape(
            B, seq_len, C, num_patch_H, self.patch_size, num_patch_W, self.patch_size
        )
        # permute to bring patch grid together
        # -> (B, seq_len, num_patch_H, num_patch_W, C, patch_size, patch_size)
        patch_data = patch_data.permute(0, 1, 3, 5, 2, 4, 6)
        num_patches = num_patch_H * num_patch_W
        patch_dim = C * self.patch_size * self.patch_size
    
        patch_data = patch_data.reshape(B, seq_len, num_patches, patch_dim) # -> (B, seq_len, num_patches, patch_dim)

        return patch_data
    
# class PositionalEncoding(nn.Module):
#     """
#     Sinusoidal Positional encoding 

#     Args:
#     -----
#     d_model: int
#         Dimensionality of the tokens
#     max_len: int
#         Length of the sequence.
#     """

#     def __init__(self, d_model, max_len=64):
#         """
#         Initializing the positional encoding
#         """
#         super().__init__()
#         self.d_model = d_model #  The dimensionality of token embeddings
#         self.max_len = max_len #  Maximum sequence length the model can handle (default 64)
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         # initializing embedding
#         self.pe = self._get_pe()
#         return

#     def _get_pe(self):
#         """
#         Initializing the temporal positional encoding given the encoding mode
#         """
#         max_len = self.max_len
#         d_model = self.d_model
        
#         pe = torch.zeros(max_len, d_model) # Creates a zero tensor - one row per position
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term) # Even dimensions get sine
#         pe[:, 1::2] = torch.cos(position * div_term) # Odd dimensions get cosine
#         pe = pe.view(1, max_len, d_model)
#         return pe

#     def forward(self, x):
#         """
#         Adding the positional encoding to the input tokens of the transformer
#         """
#         if x.device != self.pe.device:
#             self.pe = self.pe.to(self.device)
#         batch_size, seq_len, num_tokens, token_dim = x.shape
#         # Repeat for batch and truncate to actual sequence length
#         cur_pe = self.pe.repeat(batch_size, seq_len, 1, 1)[:, :, :num_tokens, :]
#         print(f"Cur pe shape: {cur_pe.shape}")
#         print(f"X entering pe shape: {x.shape}")
#         y = x + cur_pe # Adding the positional encoding to the input tokens
#         return y        
   

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_spatial_len=8, max_temporal_len=24):
        super().__init__()
        self.d_model = d_model
        self.max_spatial_len = max_spatial_len
        self.max_temporal_len = max_temporal_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Precompute 2D spatial PE [1, 1, N, d_model//2]
        N = max_spatial_len ** 2
        d_spatial = d_model // 2
        pe_spatial = torch.zeros(1, 1, N, d_spatial)
        position_h = torch.arange(0, max_spatial_len, dtype=torch.float).unsqueeze(1).repeat(1, max_spatial_len).view(-1, 1)  # [N, 1]
        position_w = torch.arange(0, max_spatial_len, dtype=torch.float).unsqueeze(0).repeat(max_spatial_len, 1).view(-1, 1)  # [N, 1]

        d_half = d_spatial // 2  # d_model//4
        div_term = torch.exp(torch.arange(0, d_half, 2).float() * (-math.log(10000.0) / d_half))  # length d_half//2

        # H encodings: first d_half dims
        pe_spatial[0, 0, :, 0:d_half:2] = torch.sin(position_h * div_term)
        pe_spatial[0, 0, :, 1:d_half:2] = torch.cos(position_h * div_term)

        # W encodings: next d_half dims
        pe_spatial[0, 0, :, d_half:d_spatial:2] = torch.sin(position_w * div_term)
        pe_spatial[0, 0, :, d_half + 1:d_spatial:2] = torch.cos(position_w * div_term)
        self.pe_spatial = pe_spatial
        # 1D temporal PE [1, max_t, 1, d_model//2]
        d_temporal = d_model // 2
        pe_temporal = torch.zeros(1, max_temporal_len, 1, d_temporal)
        position_t = torch.arange(0, max_temporal_len, dtype=torch.float).unsqueeze(1)  # [max_t, 1]
        div_term_t = torch.exp(torch.arange(0, d_temporal, 2).float() * (-math.log(10000.0) / d_temporal))  # length d_temporal//2

        pe_temporal[0, :, 0, 0::2] = torch.sin(position_t * div_term_t)
        pe_temporal[0, :, 0, 1::2] = torch.cos(position_t * div_term_t)
        self.pe_temporal = pe_temporal
        # print(f"Pe spatial shape: {pe_temporal.shape}")

    def forward(self, x):
        B, T, N, D = x.shape        
        pe_spatial = self.pe_spatial.repeat(B, T, 1, 1)  # [B, T, N, D//2]
        # print(f"Pe spatial shape: {pe_spatial.shape}")
        pe_temporal = self.pe_temporal.repeat(B, 1, N, 1)[: , :T, :, :]  # Adjust for actual T
        # print(f"Pe temporal shape: {pe_temporal.shape}")
        pe = torch.cat([pe_spatial, pe_temporal], dim=-1).to(x.device)
        # print(f"Pe shape: {pe.shape}")
        # print(f"X in pe shape: {x.shape}")
        return x + pe
    
class MaskEncoder(nn.Module):
    """
    Encodes segmentation masks into patch embeddings.
    
    Args:
        patch_size: int, size of patches for mask patching
        embed_dim: int, embedding dimension
        in_chans: int, number of input channels (1 for grayscale masks)
    """
    
    def __init__(self, patch_size, embed_dim, in_chans=1):
        super().__init__()

        self.mask_patchifier = Patchifier(patch_size)
        
        # Projection for mask patches
        self.mask_projection = nn.Sequential(
            nn.LayerNorm(patch_size * patch_size * in_chans),
            nn.Linear(patch_size * patch_size * in_chans, embed_dim)
        )
        
    def forward(self, masks):
        """
        Args:
            masks: [B, T, H, W] - segmentation masks
        
        Returns:
            mask_embeddings: [B, T, num_patches, embed_dim]
        """
        B, T, H, W = masks.shape
        
        # Convert to float32 to match LayerNorm expectations
        masks = masks.float()
        
        # Add channel dimension 
        if masks.dim() == 4:
            masks = masks.unsqueeze(2)  # [B, T, 1, H, W]
        
        # Patchify masks
        mask_patches = self.mask_patchifier(masks)  # [B, T, num_patches, patch_dim]
        
        # Project to embedding space
        mask_embeddings = self.mask_projection(mask_patches)  # [B, T, num_patches, embed_dim]
        
        return mask_embeddings

class BBoxEncoder(nn.Module):
    """
    Encodes bounding boxes into embeddings that can be used with the transformer.
    
    Args:
        embed_dim: int, embedding dimension for the transformer
        max_objects: int, maximum number of objects per frame
        bbox_dim: int, dimension of bbox coordinates (usually 4 for x1,y1,x2,y2)
    """
    
    def __init__(self, embed_dim, max_objects=11):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Learnable embeddings for bbox coordinates
        self.bbox_projection = nn.Sequential(
            nn.Linear(4, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # Positional encoding for bbox order
        self.bbox_pos_encoding = nn.Parameter(torch.randn(max_objects, embed_dim // 4))
        
        # Final projection to combine all bbox features
        self.bbox_final_proj = nn.Linear(embed_dim + embed_dim // 4, embed_dim)
        
    def forward(self, bboxes):
        """
        Args:
            bboxes: [B, T, max_objects, bbox_dim] - bounding box coordinates
        
        Returns:
            bbox_embeddings: [B, T, max_objects, embed_dim]
        """
        B, T, max_objects, bbox_dim = bboxes.shape
        
        # Convert to float32 to match Linear layer expectations
        # bboxes = bboxes.float()
        
        # Project bbox coordinates to embedding space
        bbox_coords = bboxes.view(B * T * max_objects, bbox_dim)
        bbox_emb = self.bbox_projection(bbox_coords)  # [B*T*max_objects, embed_dim]
        bbox_emb = bbox_emb.view(B, T, max_objects, self.embed_dim)
        bbox_embeddings = bbox_emb
          
        return bbox_embeddings


def get_scheduler(optimizer, num_epochs, warmup_epochs):
    '''
    Getting a scheduler for the optimizer
    '''
    # --- Warmup scheduler ---
    def warmup_lambda(epoch):
        return float(epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

    # --- Cosine Annealing after warmup ---
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)

    # --- Combination of the two schedulers ---
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]  # switch after warmup_epochs
    )
    return scheduler