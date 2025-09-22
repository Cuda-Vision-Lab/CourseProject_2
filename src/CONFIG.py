"""
Global configurations
"""

import os

config = {
            'data': {
                    'dataset_path': '/home/nfs/inf6/data/datasets/MOVi/movi_c/',

                    'batch_size': 64,
                    
                    'patch_size': 32,
                    
                    'max_objects' : 11,
                    
                    'num_workers': 0,
                    },
 
            'training': {         
                        'train':{
                            'shuffle': True,
                            'transforms': None
                                },
                        'validation':{
                            'shuffle': False,
                            'transforms': None
                                    },
                        'num_epochs':50,
                        
                        'model_name' : 'maevit1',
                        
                        'lr' : 1e-4,
                        
                        'root' : '/home/user/soltania1/CudaVisionSS2025/src/CourseProject/src',
                        },
         
            'vit_cfg': {
                        'encoder_embed_dim' : 128,
                        
                        'decoder_embed_dim' : 128,
                        
                        'max_len' : 64,
                        
                        'in_out_channels' : 3,
                        
                        'mask_ratio': 0.75,
                        
                        'norm_pix_loss' : False,
                        
                        'use_masks': True,

                        'use_bboxes': True,
                        
                        'attn_dim' : 192 ,

                        'num_heads' : 4,

                        'mlp_size' : 512,
                        
                        'encoder_depth' : 4,
                        
                        'decoder_depth' : 2,
                        
                        'predictor_depth' : 4,
                        
                        'num_preds' : 5, # number of predictor predictions
                        
                        'predictor_window_size' : 5, #sliding window size for predictor input
                        
                        # 'use_predictor' : False,

                        },


         
         
                
}