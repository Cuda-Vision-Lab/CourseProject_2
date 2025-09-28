"""
Trainer for Autoencoder, Predictor, and Inference modes.
"""

from base.baseTrainer import baseTrainer
from CONFIG import config
import argparse
import logging
import setproctitle
from model.ocvp import TransformerAutoEncoder, TransformerPredictor, OCVP
from model.holistic_encoder import HolisticEncoder
from model.holistic_decoder import HolisticDecoder
from model.holistic_predictor import HolisticTransformerPredictor
from model.predictor_wrapper import PredictorWrapper
from model.oc_encoder import ObjectCentricEncoder
from model.oc_decoder import ObjectCentricDecoder
from model.oc_predictor import ObjectCentricTransformerPredictor
from utils.utils import load_model, count_model_params
  
def get_encoder(scene_rep):
    """
    Get the encoder for the given scene representation
    """
    if scene_rep == 'holistic':
        return HolisticEncoder()  
    elif scene_rep == 'oc':
        return ObjectCentricEncoder()
    else:
        raise ValueError(f"Invalid scene representation type: {scene_rep}")

def get_decoder(scene_rep):
    """
    Get the decoder for the given scene representation
    """
    if scene_rep == 'holistic':
        return HolisticDecoder()
    elif scene_rep == 'oc':
        return ObjectCentricDecoder()
    else:
        raise ValueError(f"Invalid scene representation type: {scene_rep}")


def get_predictor(scene_rep):
    """
    Get the predictor for the given scene representation
    """
    if scene_rep == 'holistic':
        return PredictorWrapper(HolisticTransformerPredictor())
    elif scene_rep == 'oc':
        return PredictorWrapper(ObjectCentricTransformerPredictor())
    else:
        raise ValueError(f"Invalid scene representation type: {scene_rep}")

def show_logs():
    """
    Show the logs
    """
    logging.info(f"Training configuration:")
    logging.info(f"  - Epochs: {config['training']['num_epochs']}")
    logging.info(f"  - Batch size: {config['data']['batch_size']}")
    logging.info(f"  - Learning rate: {config['training']['lr']}")
    logging.info(f"  - Patch size: {config['data']['patch_size']}")
    logging.info(f"  - Attention dimension: {config['vit_cfg']['attn_dim']}")
    logging.info(f"  - Number of heads: {config['vit_cfg']['num_heads']}")
    logging.info(f"  - MLP size: {config['vit_cfg']['mlp_size']}")
    logging.info(f"  - Encoder depth: {config['vit_cfg']['encoder_depth']}")
    logging.info(f"  - Decoder depth: {config['vit_cfg']['decoder_depth']}")
    logging.info(f"  - Use masks: {config['vit_cfg']['use_masks']}")
    logging.info(f"  - Use bboxes: {config['vit_cfg']['use_bboxes']}")

    return
    
if __name__ == "__main__":
    
    # Set CUDA memory allocation configuration for better memory management
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    parser = argparse.ArgumentParser( description="Trainer for Autoencoder, Predictor, and Inference modes. Options:\n")
    parser.add_argument('-a',   '--ae', action='store_true', help='Enable autoencoder training mode')
    parser.add_argument('-p',   '--predictor', action='store_true', help='Enable predictor training mode')
    parser.add_argument('-i',   '--inference', action='store_true', help='Enable end-to-end inference mode')
    parser.add_argument('-ac',  '--ackpt', help='Checkpoint path to pretrained autoencoder')
    parser.add_argument('-pc',  '--pckpt', help='Checkpoint path to pretrained predictor')
    parser.add_argument('-s',   '--scene_rep', default='holistic', choices=['holistic', 'oc'], help='Scene representation type (default: holistic)')
    
    # parser.print_help()

    args = parser.parse_args()

    logging.info(f"Started setting up the trainer --->")
    
    trainer = baseTrainer(config)
    
    model_name = config['training']['model_name']
 
    if args.ae:
        
        '''Autoencoder training mode'''
        setproctitle.setproctitle(f"{model_name}")
        print()
        logging.info(f"AUTOENCODER TRAINING MODE --> Scene Representation: {args.scene_rep}")
        print()
        if not args.scene_rep:
            print()
            logging.warning("No specified scene representation type. By default will be considered as holistic.")
            print()
     
        # Create autoencoder model
        # mask_ratio = config['vit_cfg']['mask_ratio']
        encoder = get_encoder(scene_rep = args.scene_rep)
        decoder = get_decoder(scene_rep = args.scene_rep)
        
        model = TransformerAutoEncoder(encoder, decoder)
        
        logging.info(f"NUMBER OF MODEL PARAMETERS: {count_model_params(model)}")
        print()
        
        show_logs()
        
        training_mode = "Autoencoder"

        trainer.setup_model(model=model, mode=training_mode)
        trainer.train_model()
              
    elif args.predictor:
        
        """Predictor training mode"""
        show_logs()
        logging.info(f"  - Predictor depth: {config['vit_cfg']['predictor_depth']}")
        logging.info(f"  - Number of predictions: {config['vit_cfg']['num_preds']}")
        logging.info(f"  - Predictor window size: {config['vit_cfg']['predictor_window_size']}")
        print()
        logging.info(f"PREDICTOR TRAINING MODE --> Scene Representation: {args.scene_rep}")
        print()
        setproctitle.setproctitle(f"{model_name}")
        
        if not args.ackpt:
            raise FileNotFoundError("Please specify the checkpoint to the pretrained AutoEncoder model")
            
        encoder = get_encoder(scene_rep = args.scene_rep)
        decoder = get_decoder(scene_rep = args.scene_rep)
        predictor = get_predictor(scene_rep = args.scene_rep)
              
        model = TransformerPredictor(encoder, decoder, predictor)
        
        # Load AE weights, freeze encoder and decoder
        model = load_model(model, mode="predictor_training", savepath= args.ackpt) 
        
        if not( any(p.requires_grad for p in model.encoder.parameters()) or (any(p.requires_grad for p in model.decoder.parameters()))):
            logging.info("Encoder and decoder parameters are frozen. Proceeding to train the predictor ...")
        else:
            logging.error( "Encoder and decoder parameters are NOT FROZEN!! Please freeze them before training the predictor")
            
        training_mode = "Predictor"
        
        #Train and save predictor checkpoints
        trainer.setup_model(model=model, mode=training_mode)
        trainer.train_model()
        

    elif args.inference:
        
        if (not args.ackpt) or (not args.pckpt):
            raise FileNotFoundError("Please specify the checkpoint to both pretrained AutoEncoder and Predictor models")
        
        encoder = get_encoder(scene_rep = args.scene_rep)          
        decoder = get_decoder(scene_rep = args.scene_rep)
        predictor = get_predictor(scene_rep = args.scene_rep)
        
        model = OCVP(encoder, decoder, predictor)
        
        model = load_model(model, mode="inference", savepath= args.ackpt) 
        
        # do inference and save results somewhere. 
        # some inference.py that takes the above models and do the inference

    else:
        raise ValueError("Please specify a valid mode: --ae, --predictor, or --inference")
        