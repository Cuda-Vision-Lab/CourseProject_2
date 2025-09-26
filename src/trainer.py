from base.baseTrainer import baseTrainer
from CONFIG import config
import argparse
import logging
import setproctitle
from model.ocvp import TransformerAutoEncoder, TransformerPredictor, OCVP
# from model.encoder import MultiModalVitEncoder
from model.decoder import VitDecoder
from model.predictor import Predictor
from model.holistic_encoder import HolisticEncoder
from model.oc_encoder import ObjectCentricEncoder
from utils.utils import load_model, count_model_params
from torch.utils.tensorboard import SummaryWriter
  
def get_encoder(scene_rep, mode ):
    if scene_rep == 'holistic':
        return HolisticEncoder(mode= mode)
    
    elif scene_rep == 'oc':
        return ObjectCentricEncoder(mode= mode)
    else:
        raise ValueError(f"Invalid scene representation type: {scene_rep}")
    
    
if __name__ == "__main__":
    
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
        setproctitle.setproctitle(f"{model_name}_AE")
        logging.info(f"AUTOENCODER TRAINING MODE --> Scene Representation: {args.scene_rep}")
        
        if not args.ackpt:
            print()
            logging.warning("No specified scene representation type. By default will be considered as holistic.")
            print()
     
        # Create autoencoder model
        mask_ratio = config['vit_cfg']['mask_ratio']
        encoder = get_encoder(scene_rep = args.scene_rep, mode= 'training')
        decoder = VitDecoder(mode= 'training')
        
        model = TransformerAutoEncoder(encoder, decoder)
        
        logging.info(f"NUMBER OF MODEL PARAMETERS: {count_model_params(model)}")
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
        logging.info(f"  - Mask ratio: {config['vit_cfg']['mask_ratio']}")
        logging.info(f"  - Use masks: {config['vit_cfg']['use_masks']}")
        logging.info(f"  - Use bboxes: {config['vit_cfg']['use_bboxes']}")
        
        training_mode = "Autoencoder"

        trainer.setup_model(model=model, mode=training_mode)
        trainer.train_model()
              
    elif args.predictor:
        logging.info(f"  - Predictor depth: {config['vit_cfg']['predictor_depth']}")
        logging.info(f"  - Number of predictions: {config['vit_cfg']['num_preds']}")
        logging.info(f"  - Predictor window size: {config['vit_cfg']['predictor_window_size']}")
        setproctitle.setproctitle(f"{model_name}_predictor")
        logging.info(f"PREDICTOR TRAINING MODE --> Scene Representation: {args.scene_rep}")
        
        if not args.ackpt:
            raise FileNotFoundError("Please specify the checkpoint to the pretrained AutoEncoder model")
            
        encoder = get_encoder(scene_rep = args.scene_rep, mode= 'predictor')
        
        # Load AE weights
        encoder, decoder,_,_= load_model(model= encoder, savepath= args.ackpt) # TODO: also pass optimizer here
        
        predictor = Predictor()
        
        model = TransformerPredictor(encoder, predictor)
        model.encoder.requires_grad_(False)
        model.decoder.requires_grad_(False)
        
        training_mode = "Predictor"
        
        # Load AE weights
        # ae_checkpoint = torch.load("ae_checkpoint.pth", map_location=device)
        # model_full.encoder.load_state_dict(ae_checkpoint['encoder_state_dict'])
        # model_full.decoder.load_state_dict(ae_checkpoint['decoder_state_dict'])

        # # Freeze encoder and decoder
        # model_full.encoder.requires_grad_(False)
        # model_full.decoder.requires_grad_(False)

        # # Training setup for predictor
        # optimizer = torch.optim.Adam(
        #     filter(lambda p: p.requires_grad, model_full.parameters()),  # Only predictor params
        #     lr=0.001
        # )
        
        #Train and save predictor checkpoints
        # trainer.setup_model(model=model)
        # trainer.train_model()
        

    elif args.inference:
        
        if (not args.ackpt) or (not args.pckpt):
            raise FileNotFoundError("Please specify the checkpoint to both pretrained AutoEncoder and Predictor models")
        

        encoder = get_encoder(scene_rep = args.scene_rep, mode= 'inference')          
        decoder = VitDecoder(mode='inference')
        predictor = Predictor()
         
        encoder,_,_,_= load_model(model= encoder, savepath= args.ackpt) # TODO: also pass optimizer here
        decoder,_,_,_= load_model(model= decoder, savepath= args.ackpt) # TODO: also pass optimizer here
        predictor,_,_,_= load_model(model= predictor, savepath= args.pckpt) # TODO: also pass optimizer here
        
        model = OCVP(encoder, decoder, predictor)
        
        # do inference and save results somewhere. 
        # some inference.py that takes the above models and do the inference

    else:
        raise ValueError("Please specify a valid mode: --ae, --predictor, or --inference")
        