"""
Global configurations
"""

import os

config = {
            'data': {
                    'dataset_path': '/home/nfs/inf6/data/datasets/MOVi/movi_c/',

                    'batch_size': 32,  # Reduced from 64 to improve memory efficiency
                    
                    'patch_size': 32,
                    
                    'max_objects' : 11,
                    
                    'num_workers': 8,  # Use 8 CPU cores for data loading
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
                        'num_epochs':300,

                        'warmup_epochs': 30,
                        
                        'model_name' : 'holistic_large_mask_0.25',
                        
                        'lr' : 4e-4,
                        
                        'save_frequency': 50,
                        
                        'root' : '/home/user/soltania1/CourseProject_2/src',
                        },
         
            'vit_cfg': {
                        'encoder_embed_dim' : 256, # Increased from 64
                        
                        'decoder_embed_dim' : 256, # Increased from 64
                        
                        'max_len' : 256,
                        
                        'in_out_channels' : 3,
                        
                        'mask_ratio': 0.25, # Decreased from 0.75
                        
                        'norm_pix_loss' : True,
                        
                        'use_masks': True,

                        'use_bboxes': True,
                        
                        'attn_dim' : 256 ,

                        'num_heads' : 8, # Increased from 4

                        'mlp_size' : 1024, # Increased from 512
                        
                        'encoder_depth' : 12, # Increased from 4
                        
                        'decoder_depth' : 6, # Increased from 2
                        
                        'predictor_depth' : 4,
                        
                        'num_preds' : 5, # number of predictor predictions
                        
                        'predictor_window_size' : 5, #sliding window size for predictor input
                        
                        # 'use_predictor' : False,

                        },


         
         
                
}