"""
Global configurations
"""

import os

config = {
            'data': {
                    'dataset_path': '/home/nfs/inf6/data/datasets/MOVi/movi_c/',

                    'batch_size': 32,  # Reduced from 64 to improve memory efficiency
                    
                    'patch_size': 16,
                    
                    'max_objects' : 11,
                    
                    'num_workers': 8,  # Use 8 CPU cores for data loading
                    
                    'image_height' : 128,
                    
                    'image_width' : 128,
                    },
 
            'training': {         
                        'train':{
                            'shuffle': True,
                            'transforms': 'train'
                                },
                        'validation':{
                            'shuffle': False,
                            'transforms': 'validation'
                                    },
                        
                        'num_epochs':100,

                        'warmup_epochs': 5,

                        'early_stopping_patience': 10,
                        
                        'model_name' : '01_OC_AE_XL',
                        
                        'lr' : 1e-3,  # Reduced from 4e-3 for more stable training
                        
                        'save_frequency': 25,
                        
                        'use_scheduler': True,
                        
                        'use_early_stopping': True,
                        
                        'root' : '/home/user/soltania1/CourseProject_2/src',
                        },
         
            'vit_cfg': {
                        'encoder_embed_dim' : 512, # Increased from 64
                        
                        'decoder_embed_dim' : 384, # Increased to match encoder for better reconstruction
                        
                        'max_len' : 64,  
                        
                        'in_out_channels' : 3,
                        
                        'use_masks': True,

                        'use_bboxes': False,
                        
                        'attn_dim' : 128 ,

                        'num_heads' : 8, # Must divide embed_dim evenly (256 ÷ 8 = 32)

                        'mlp_size' : 1024, # Moderate increase (was 1024, now between 1024-2048)
                        
                        'encoder_depth' : 12, # Moderate increase (was 12, now between 12-24)
                        
                        'decoder_depth' : 8, # Moderate increase (was 6, now between 6-12)
                        
                        'predictor_depth' : 8,
                        
                        'num_preds' : 5, # number of predictor predictions
                        
                        'predictor_window_size' : 5, #sliding window size for predictor input
                        
                        'predictor_embed_dim' : 256,
                        
                        'residual' : True, # Residual connection in predictor
                        
                        
                        },

         
}