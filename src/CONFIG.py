"""
Global configurations
"""

import os

config = {
            'data': {
                    'dataset_path': '/home/nfs/inf6/data/datasets/MOVi/movi_c/',

                    'batch_size': 8,  # Further reduced to prevent OOM during decoder upsampling with AMP
                    
                    'patch_size': 16,
                    
                    'max_objects' : 11,
                    
                    'num_workers': 8,  # Use 8 CPU cores for data loading
                    
                    'image_height' : 64,
                    
                    'image_width' : 64,
                    },
 
            'training': {         
                        'train':{
                            'shuffle': False,
                            'transforms': 'train'
                                },
                        'validation':{
                            'shuffle': False,
                            'transforms': 'validation'
                                    },
                        
                        'num_epochs':300,

                        'warmup_epochs': 15,

                        'early_stopping_patience': 15,
                        
                        'model_name' : '01_OC_AE_XL_64_Full_CNN',
                        
                        'lr' : 4e-4,  # Further reduced for better convergence with larger model
                        
                        'save_frequency': 25,
                        
                        'use_scheduler': True,
                        
                        'use_early_stopping': True,
                        
                        'use_transforms': False,  ## TOOD: change to True for training
                        
                        'use_amp': True,  # Enable Mixed Precision Training
                        
                        'root' : '/home/user/soltania1/CourseProject_2/src',
                        },
         
            'vit_cfg': {
                        'encoder_embed_dim' : 512, # Increased significantly for better representation capacity
                        
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